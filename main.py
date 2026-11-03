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

app = FastAPI(title="AI Agent Step-Level Cache & Loop-Guard Gateway", version="3.6.1")

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

def detect_infinite_loop(messages: list) -> bool:
    """网关级死循环检测：检查最近两次 assistant 回复是否完全一致"""
    assistant_signatures = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content", "") or ""
            tool_calls = msg.get("tool_calls", []) or []
            signature = f"{content.strip()}:{json.dumps(tool_calls, sort_keys=True)}"
            assistant_signatures.append(signature)
    
    if len(assistant_signatures) >= 2 and assistant_signatures[-1] == assistant_signatures[-2]:
        if assistant_signatures[-1] != ":" : # 排除纯空回复
            return True
    return False

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
    
    # 🛑 1. 核心护栏：检测死循环，若陷入死循环则主动向上下文中注入系统熔断警告
    if detect_infinite_loop(messages):
        print(f"🚨 [LOOP GUARD ACTIVATED] 检测到租户 [{tenant}] 的 Agent 陷入死循环！正在注入强制熔断警告...")
        messages.append({
            "role": "system",
            "content": "【网关强制护栏】检测到你正在陷入重复输出或重复调用相同工具的死循环中。请立刻停止当前死循环，绝对不要再次调用任何工具，请根据已有线索直接向用户给出最终结论与总结报告。"
        })
        body["messages"] = messages

    current_step_query = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") in ["user", "tool"]:
            content_field = msg.get("content", "")
            if isinstance(content_field, str) and content_field.strip():
                current_step_query = content_field
                break
            elif isinstance(content_field, list):
                for part in content_field:
                    if isinstance(part, dict) and part.get("type") == "text":
                        current_step_query = part.get("text", "")
                        break
            if current_step_query:
                break

    # 🛑 2. 过滤 DSH 后台系统辅助请求
    background_keywords = [
        "Generate the session title", 
        "Time sampled while preparing", 
        "session title from this JSON"
    ]
    if any(kw in current_step_query for kw in background_keywords):
        print(f"🛡️ 拦截并放行后台系统辅助请求: '{current_step_query[:40]}...'")
        headers = {"Authorization": f"Bearer {REAL_LLM_API_KEY}", "Content-Type": "application/json"}
        if body.get("stream", False):
            async def bg_stream():
                async with httpx.AsyncClient(timeout=60.0) as client:
                    async with client.stream("POST", f"{REAL_LLM_BASE_URL.rstrip('/')}/chat/completions", json=body, headers=headers) as response:
                        async for line in response.aiter_lines():
                            if line: yield f"{line}\n\n"
            return StreamingResponse(bg_stream(), media_type="text/event-stream")
        else:
            async with httpx.AsyncClient(timeout=60.0) as client:
                res = await client.post(f"{REAL_LLM_BASE_URL.rstrip('/')}/chat/completions", json=body, headers=headers)
                return res.json()

    is_stream = body.get("stream", False)
    client_model = body.get("model", DEFAULT_CHAT_MODEL)

    print(f"📥 收到步骤级请求 [租户: {tenant}] | Query片段: '{current_step_query[:50]}...'")

    # 3. 步骤级原子向量检索 (阈值 0.75，兼顾灵活性)
    cached_data = None
    if vector_index and current_step_query:
        try:
            results = vector_index.query(data=current_step_query, top_k=1, include_metadata=True)
            if results and results[0].score >= 0.75:
                meta = results[0].metadata
                if meta and meta.get("tenant") == tenant:
                    cached_data = meta
                    print(f"🎯 [STEP-LEVEL HIT] 成功命中原子工具步骤！租户 [{tenant}] | 得分: {results[0].score:.2f} | 绑定工具: {meta.get('tool_names')}")
            else:
                score_val = results[0].score if results else 0.0
                print(f"🔍 Step Cache MISS (最高得分 {score_val:.2f} < 0.75)")
        except Exception as e:
            print(f"🔥 Vector search error: {e}")

    if cached_data:
        assistant_message_json = cached_data.get("assistant_message")
        prompt_tokens = int(cached_data.get("prompt_tokens", 600))
        completion_tokens = int(cached_data.get("completion_tokens", 150))
        
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
                            "id": tc.get("id", f"call_step_{idx}"),
                            "type": tc.get("type", "function"),
                            "function": tc.get("function", {})
                        })
                    delta_payload["tool_calls"] = formatted_tool_calls

                chunk_data = {
                    "id": "chatcmpl-step-cached",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": client_model,
                    "choices": [{"index": 0, "delta": delta_payload, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                
                end_chunk = {
                    "id": "chatcmpl-step-cached",
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
                "id": "chatcmpl-step-cached",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": client_model,
                "choices": [{"index": 0, "message": assistant_message, "finish_reason": proper_finish_reason}]
            }

    # 4. 未命中，转发给 DeepSeek 官方
    update_metrics(tenant, hit=False, saved_cost=0.0)
    headers = {"Authorization": f"Bearer {REAL_LLM_API_KEY}", "Content-Type": "application/json"}
    
    async def proxy_stream():
        full_content = ""
        collected_tool_calls = {}
        has_sent_finish = False
        token_tracker = {"prompt": max(100, len(current_step_query))}
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", f"{REAL_LLM_BASE_URL.rstrip('/')}/chat/completions", json=body, headers=headers) as response:
                if response.status_code != 200:
                    error_detail = await response.aread()
                    yield f"data: {json.dumps({'error': error_detail.decode()})}\n\n"
                    return
                
                async for line in response.aiter_lines():
                    if not line: continue
                    line_str = line.strip()
                    if line_str.startswith("data: "):
                        data_content = line_str[6:].strip()
                        if data_content == "[DONE]": continue
                        try:
                            data = json.loads(data_content)
                            if "usage" in data and data["usage"]:
                                token_tracker["prompt"] = data["usage"].get("prompt_tokens", token_tracker["prompt"])
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                if delta.get("content"):
                                    full_content += delta.get("content")
                                for tc in delta.get("tool_calls", []):
                                    idx = tc.get("index", 0)
                                    if idx not in collected_tool_calls:
                                        collected_tool_calls[idx] = {"id": tc.get("id"), "type": tc.get("type", "function"), "function": {"name": "", "arguments": ""}}
                                    if tc.get("id"): collected_tool_calls[idx]["id"] = tc.get("id")
                                    func = tc.get("function", {})
                                    if func.get("name"): collected_tool_calls[idx]["function"]["name"] += func.get("name")
                                    if func.get("arguments"): collected_tool_calls[idx]["function"]["arguments"] += func.get("arguments")
                                if choices[0].get("finish_reason"):
                                    has_sent_finish = True
                        except Exception:
                            pass
                        yield f"data: {data_content}\n\n"
                    else:
                        yield f"{line_str}\n\n"

        if not has_sent_finish:
            finish_chunk = {
                "id": "chatcmpl-proxy-end", "object": "chat.completion.chunk", "created": int(time.time()),
                "model": client_model, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if collected_tool_calls else "stop"}]
            }
            yield f"data: {json.dumps(finish_chunk)}\n\n"
        yield "data: [DONE]\n\n"
            
        tool_names = [tc["function"]["name"] for tc in collected_tool_calls.values() if "function" in tc and "name" in tc["function"]]
        if vector_index and current_step_query and tool_names:
            assistant_msg = {"role": "assistant"}
            if full_content: assistant_msg["content"] = full_content
            if collected_tool_calls: assistant_msg["tool_calls"] = list(collected_tool_calls.values())

            try:
                vector_index.upsert(
                    vectors=[{
                        "id": f"step-{tenant}-{int(time.time()*1000)}",
                        "data": current_step_query,
                        "metadata": {
                            "tenant": tenant,
                            "tool_names": json.dumps(tool_names, ensure_ascii=False),
                            "assistant_message": json.dumps(assistant_msg, ensure_ascii=False),
                            "prompt_tokens": token_tracker["prompt"],
                            "completion_tokens": max(50, len(str(collected_tool_calls)) // 2)
                        }
                    }]
                )
                print(f"💾 原子步骤工具缓存写入成功 [Tenant: {tenant}] | Tools: {tool_names}")
            except Exception as e:
                print(f"🔥 Step vector upsert error: {e}")

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
                tool_names = [tc["function"]["name"] for tc in message.get("tool_calls", []) if "function" in tc and "name" in tc["function"]]
                
                if vector_index and current_step_query and tool_names:
                    vector_index.upsert(
                        vectors=[{
                            "id": f"step-{tenant}-{int(time.time()*1000)}",
                            "data": current_step_query,
                            "metadata": {
                                "tenant": tenant,
                                "tool_names": json.dumps(tool_names, ensure_ascii=False),
                                "assistant_message": json.dumps(message, ensure_ascii=False),
                                "prompt_tokens": usage.get("prompt_tokens", 500),
                                "completion_tokens": usage.get("completion_tokens", 100)
                            }
                        }]
                    )
                    print(f"💾 非流式原子步骤缓存写入成功 [{tenant}]")
            except Exception as e:
                print(f"🔥 Non-stream step cache error: {e}")
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
        raise HTTPException(status_code=500, detail=str(e))

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