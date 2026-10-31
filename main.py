import os
import json
import time
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
from upstash_vector import Index
from upstash_redis import Redis

app = FastAPI(title="AI Agent Tool-Chain Cache Gateway - SaaS Edition", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)

VECTOR_URL = os.getenv("UPSTASH_VECTOR_URL", "") or os.getenv("UPSTASH_VECTOR_REST_URL", "")
VECTOR_TOKEN = os.getenv("UPSTASH_VECTOR_TOKEN", "") or os.getenv("UPSTASH_VECTOR_REST_TOKEN", "")
REDIS_URL = os.getenv("UPSTASH_REDIS_URL", "") or os.getenv("UPSTASH_REDIS_REST_URL", "")
REDIS_TOKEN = os.getenv("UPSTASH_REDIS_TOKEN", "") or os.getenv("UPSTASH_REDIS_REST_TOKEN", "")

REAL_LLM_API_KEY = os.getenv("REAL_LLM_API_KEY", "")
REAL_LLM_BASE_URL = os.getenv("REAL_LLM_BASE_URL", "https://api.deepseek.com")
DEFAULT_CHAT_MODEL = os.getenv("DEFAULT_CHAT_MODEL", "deepseek-chat")

vector_index = Index(url=VECTOR_URL, token=VECTOR_TOKEN) if VECTOR_URL and VECTOR_TOKEN else None
redis_client = Redis(url=REDIS_URL, token=REDIS_TOKEN) if REDIS_URL and REDIS_TOKEN else None

def init_default_tenants():
    if not redis_client:
        return
    defaults = {
        "sk-gateway-demo001": {"name": "企业客户A-电商业务部", "plan": "pro", "expires_at": 9999999999},
        "sk-gateway-vip002": {"name": "企业客户B-智能客服部", "plan": "pro", "expires_at": 9999999999},
        "sk-gateway-free": {"name": "免费体验用户", "plan": "trial", "expires_at": 9999999999},
        "sk-gw-client-vip888": {"name": "某某科技创新公司", "plan": "pro", "expires_at": 9999999999}
    }
    for key, info in defaults.items():
        redis_key = f"gateway:key:{key}"
        if not redis_client.exists(redis_key):
            redis_client.set(redis_key, json.dumps(info))

init_default_tenants()

def get_tenant_from_credentials(credentials: HTTPAuthorizationCredentials | None) -> str:
    if not credentials:
        return "public-default-tenant"
    token = credentials.credentials
    if not redis_client:
        return "default-tenant"
    try:
        tenant_data_str = redis_client.get(f"gateway:key:{token}")
        if not tenant_data_str:
            return f"tenant-{token[-6:]}" if len(token) > 6 else "unknown-tenant"
        if isinstance(tenant_data_str, str):
            tenant_info = json.loads(tenant_data_str)
        else:
            tenant_info = tenant_data_str
        if tenant_info.get("expires_at", 0) < time.time():
            raise HTTPException(status_code=403, detail="API Key has expired.")
        return tenant_info.get("name", "paid-tenant")
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"🔥 Auth error: {e}")
        return "default-tenant"

def calculate_saved_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    utc_now = datetime.now(timezone.utc)
    hour_decimal = utc_now.hour + utc_now.minute / 60.0
    is_peak = 9.5 <= hour_decimal <= 23.5
    is_pro = "pro" in model_name.lower() or "coder" in model_name.lower()
    if is_pro:
        input_rate = 9.0 if is_peak else 4.5
        output_rate = 27.0 if is_peak else 13.5
    else:
        input_rate = 2.0 if is_peak else 1.0
        output_rate = 8.0 if is_peak else 4.0
    cost = (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000
    return round(cost, 6)

def update_metrics(tenant: str, hit: bool, saved_cost: float = 0.0):
    if not redis_client:
        return
    try:
        total_key = f"tenant:{tenant}:total_requests"
        hits_key = f"tenant:{tenant}:cache_hits"
        cost_key = f"tenant:{tenant}:saved_cost"
        redis_client.incr(total_key)
        if hit:
            redis_client.incr(hits_key)
            redis_client.incrbyfloat(cost_key, saved_cost)
        print(f"📊 Metrics updated for [{tenant}] | Hit: {hit} | Saved: ¥{saved_cost:.4f}")
    except Exception as e:
        print(f"🔥 Metrics update error: {e}")

@app.get("/v1/models")
@app.get("/models")
@app.get("/v1/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "deepseek-chat", "object": "model", "owned_by": "gateway"},
            {"id": "deepseek-pro", "object": "model", "owned_by": "gateway"},
            {"id": DEFAULT_CHAT_MODEL, "object": "model", "owned_by": "gateway"}
        ]
    }

async def handle_chat_completion(body: dict, tenant: str):
    messages = body.get("messages", [])
    
    # 提取用户输入的 Prompt
    user_prompt = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content_field = msg.get("content", "")
            if isinstance(content_field, str):
                user_prompt = content_field
            elif isinstance(content_field, list):
                for part in content_field:
                    if isinstance(part, dict) and part.get("type") == "text":
                        user_prompt = part.get("text", "")
            break

    # 检查是否为 Agent 工具内部循环的返回结果步骤
    last_message = messages[-1] if messages else {}
    is_agent_loop = isinstance(last_message, dict) and last_message.get("role") in ["tool", "function"]

    is_stream = body.get("stream", False)
    client_model = body.get("model", DEFAULT_CHAT_MODEL)

    print(f"📥 收到 Agent 任务 [租户: {tenant}] | Prompt: '{user_prompt[:50]}...' | Stream: {is_stream} | AgentLoop: {is_agent_loop}")

    # 1. 尝试匹配已缓存的 Agent 工具链 / 执行计划
    cached_data = None
    if vector_index and user_prompt and not is_agent_loop:
        try:
            results = vector_index.query(data=user_prompt, top_k=1, include_metadata=True)
            if results and results[0].score >= 0.88:  # 稍微放宽一点阈值以更好匹配复杂意图
                meta = results[0].metadata
                if meta and meta.get("tenant") == tenant:
                    cached_data = meta
                    is_tool_chain_hit = meta.get("is_tool_chain", False)
                    print(f"🎯 {'[AGENT TOOL-CHAIN] ' if is_tool_chain_hit else ''}Cache HIT for tenant [{tenant}] prompt: '{user_prompt}' (Score: {results[0].score})")
            else:
                score_val = results[0].score if results else 0.0
                print(f"🔍 Cache MISS (Score {score_val:.2f} < 0.88) for prompt: '{user_prompt}'")
        except Exception as e:
            print(f"🔥 Vector search error: {e}")
    else:
        print(f"⏭️ 跳过语义检索 (vector_index有效: {bool(vector_index)}, 有效Prompt: {bool(user_prompt)}, 非AgentLoop: {not is_agent_loop})")

    if cached_data:
        assistant_message_json = cached_data.get("assistant_message")
        prompt_tokens = int(cached_data.get("prompt_tokens", 1200)) # 复杂任务通常消耗更多 Token
        completion_tokens = int(cached_data.get("completion_tokens", 400))
        
        saved_amount = calculate_saved_cost(client_model, prompt_tokens, completion_tokens)
        update_metrics(tenant, hit=True, saved_cost=saved_amount)

        try:
            assistant_message = json.loads(assistant_message_json)
        except Exception:
            assistant_message = {"role": "assistant", "content": assistant_message_json}

        has_tools = bool(assistant_message.get("tool_calls"))
        proper_finish_reason = "tool_calls" if has_tools else "stop"

        if is_stream:
            async def cached_stream():
                delta_payload = {}
                if assistant_message.get("content"):
                    delta_payload["content"] = assistant_message["content"]
                if has_tools:
                    formatted_tool_calls = []
                    for idx, tc in enumerate(assistant_message["tool_calls"]):
                        formatted_tool_calls.append({
                            "index": idx,
                            "id": tc.get("id", f"call_cached_{idx}"),
                            "type": tc.get("type", "function"),
                            "function": tc.get("function", {})
                        })
                    delta_payload["tool_calls"] = formatted_tool_calls

                chunk_data = {
                    "id": "chatcmpl-agent-cached",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": client_model,
                    "choices": [{"index": 0, "delta": delta_payload, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                
                end_chunk = {
                    "id": "chatcmpl-agent-cached",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": client_model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": proper_finish_reason}]
                }
                yield f"data: {json.dumps(end_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(cached_stream(), media_type="text/event-stream")
        else:
            return {
                "id": "chatcmpl-agent-cached",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": client_model,
                "choices": [{"index": 0, "message": assistant_message, "finish_reason": proper_finish_reason}]
            }

    # 2. 未命中，转发给 DeepSeek 官方并实时捕获其工具调度链
    update_metrics(tenant, hit=False, saved_cost=0.0)
    headers = {
        "Authorization": f"Bearer {REAL_LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    
    async def proxy_stream():
        full_content = ""
        collected_tool_calls = {}
        has_sent_finish = False
        token_tracker = {"prompt": max(200, len(user_prompt))}
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", f"{REAL_LLM_BASE_URL.rstrip('/')}/chat/completions", json=body, headers=headers) as response:
                if response.status_code != 200:
                    error_detail = await response.aread()
                    err_msg = error_detail.decode()
                    print(f"🔥 Upstream DeepSeek error [{response.status_code}]: {err_msg}")
                    yield f"data: {json.dumps({'error': err_msg})}\n\n"
                    return
                
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    line_str = line.strip()
                    if line_str.startswith("data: "):
                        data_content = line_str[6:].strip()
                        if data_content == "[DONE]":
                            continue
                        try:
                            data = json.loads(data_content)
                            if "usage" in data and data["usage"]:
                                token_tracker["prompt"] = data["usage"].get("prompt_tokens", token_tracker["prompt"])

                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    full_content += content
                                
                                t_calls = delta.get("tool_calls")
                                if t_calls:
                                    for tc in t_calls:
                                        idx = tc.get("index", 0)
                                        if idx not in collected_tool_calls:
                                            collected_tool_calls[idx] = {"id": tc.get("id"), "type": tc.get("type", "function"), "function": {"name": "", "arguments": ""}}
                                        if tc.get("id"):
                                            collected_tool_calls[idx]["id"] = tc.get("id")
                                        func = tc.get("function", {})
                                        if func.get("name"):
                                            collected_tool_calls[idx]["function"]["name"] += func.get("name")
                                        if func.get("arguments"):
                                            collected_tool_calls[idx]["function"]["arguments"] += func.get("arguments")

                                if choices[0].get("finish_reason"):
                                    has_sent_finish = True
                        except Exception:
                            pass
                        yield f"data: {data_content}\n\n"
                    else:
                        yield f"{line_str}\n\n"

        if not has_sent_finish:
            finish_reason_fallback = "tool_calls" if collected_tool_calls else "stop"
            finish_chunk = {
                "id": "chatcmpl-proxy-end",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": client_model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason_fallback}]
            }
            yield f"data: {json.dumps(finish_chunk)}\n\n"
        
        yield "data: [DONE]\n\n"
            
        # 任务完成后的工具链与执行计划持久化缓存
        print(f"📝 准备持久化 Agent 工具链 [租户: {tenant}] -> user_prompt: '{user_prompt}', 包含Tools: {bool(collected_tool_calls)}")
        if vector_index and user_prompt:
            assistant_msg = {"role": "assistant"}
            if full_content:
                assistant_msg["content"] = full_content
            if collected_tool_calls:
                assistant_msg["tool_calls"] = list(collected_tool_calls.values())

            p_tokens = token_tracker["prompt"]
            c_tokens = max(100, len(full_content) // 2 + len(str(collected_tool_calls)))

            if assistant_msg.get("content") or assistant_msg.get("tool_calls"):
                try:
                    vector_index.upsert(
                        vectors=[{
                            "id": f"agent-plan-{tenant}-{int(time.time()*1000)}",
                            "data": user_prompt,
                            "metadata": {
                                "tenant": tenant, 
                                "is_tool_chain": bool(collected_tool_calls),
                                "assistant_message": json.dumps(assistant_msg, ensure_ascii=False),
                                "prompt_tokens": p_tokens,
                                "completion_tokens": c_tokens
                            }
                        }]
                    )
                    print(f"💾 Agent Tool-Chain Plan Cached successfully! [Tenant: {tenant}] (Tools: {bool(collected_tool_calls)}, P:{p_tokens}, C:{c_tokens})")
                except Exception as e:
                    print(f"🔥 Vector upsert error: {e}")
            else:
                print("⚠️ assistant_msg 内容为空，放弃 Agent 缓存写入")
        else:
            print(f"⚠️ 跳过写入缓存：vector_index有效={bool(vector_index)}, 有效Prompt={bool(user_prompt)}")

    if is_stream:
        return StreamingResponse(proxy_stream(), media_type="text/event-stream")
    else:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{REAL_LLM_BASE_URL.rstrip('/')}/chat/completions", json=body, headers=headers)
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            data = response.json()
            try:
                choice = data["choices"][0]
                message = choice.get("message", {})
                usage = data.get("usage", {})
                p_tokens = usage.get("prompt_tokens", max(200, len(user_prompt)))
                c_tokens = usage.get("completion_tokens", max(100, len(message.get("content", "")) // 2))

                has_tc = bool(message.get("tool_calls"))
                print(f"📝 非流式 Agent 任务持久化 [租户: {tenant}] -> 包含Tools: {has_tc}")
                if vector_index and user_prompt:
                    vector_index.upsert(
                        vectors=[{
                            "id": f"agent-plan-{tenant}-{int(time.time()*1000)}",
                            "data": user_prompt,
                            "metadata": {
                                "tenant": tenant, 
                                "is_tool_chain": has_tc,
                                "assistant_message": json.dumps(message, ensure_ascii=False),
                                "prompt_tokens": p_tokens,
                                "completion_tokens": c_tokens
                            }
                        }]
                    )
                    print(f"💾 Non-stream Agent Tool-Chain Cached successfully for tenant [{tenant}]")
            except Exception as e:
                print(f"🔥 Non-stream cache upsert error: {e}")
            return data

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
@app.post("/v1/v1/chat/completions")
async def chat_completions(request: Request, credentials: HTTPAuthorizationCredentials | None = Security(security)):
    tenant = get_tenant_from_credentials(credentials)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    return await handle_chat_completion(body, tenant)

@app.get("/stats")
def get_stats(tenant: str = "public-default-tenant"):
    if not redis_client:
        return {"metrics": {"total_requests": 0, "cache_hits": 0, "cache_misses": 0, "hit_rate": "0%", "miss_rate": "0%", "estimated_saved_cost": "¥0.00"}}
    try:
        total = int(redis_client.get(f"tenant:{tenant}:total_requests") or 0)
        hits = int(redis_client.get(f"tenant:{tenant}:cache_hits") or 0)
        misses = max(0, total - hits)
        cost = float(redis_client.get(f"tenant:{tenant}:saved_cost") or 0.0)
        
        hit_rate = f"{(hits / total * 100):.1f}%" if total > 0 else "0%"
        miss_rate = f"{(misses / total * 100):.1f}%" if total > 0 else "0%"
        
        return {
            "tenant": tenant,
            "metrics": {
                "total_requests": total,
                "cache_hits": hits,
                "cache_misses": misses,
                "hit_rate": hit_rate,
                "miss_rate": miss_rate,
                "estimated_saved_cost": f"¥{cost:.4f}"
            }
        }
    except Exception as e:
        raise HTTPException(status_status=500, detail=str(e))

@app.post("/admin/create_tenant")
def create_tenant(api_key: str, name: str, plan: str = "pro", days_valid: int = 30):
    if not redis_client:
        raise HTTPException(status_code=500, detail="Redis not connected")
    try:
        expires_at = time.time() + (days_valid * 86400)
        tenant_info = {"name": name, "plan": plan, "expires_at": expires_at}
        redis_client.set(f"gateway:key:{api_key}", json.dumps(tenant_info))
        return {"status": "success", "message": f"Tenant {name} created successfully", "api_key": api_key, "expires_at": expires_at}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))