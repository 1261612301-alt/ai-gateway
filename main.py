import os
import json
import time
from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
from upstash_vector import Index
from upstash_redis import Redis

app = FastAPI(title="AI Semantic Cache Gateway", version="2.4.0")

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

API_KEY_TO_TENANT = {
    "sk-gateway-demo001": "企业客户A-电商业务部",
    "sk-gateway-vip002": "企业客户B-智能客服部",
    "sk-gateway-free": "免费体验用户"
}

def get_tenant_from_credentials(credentials: HTTPAuthorizationCredentials | None) -> str:
    if not credentials:
        return "public-default-tenant"
    token = credentials.credentials
    return API_KEY_TO_TENANT.get(token, f"tenant-{token[-6:]}" if len(token) > 6 else "default-tenant")

def update_metrics(tenant: str, hit: bool, saved_cost: float = 0.002):
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
        print(f"✅ Metrics updated successfully for tenant [{tenant}] | Hit: {hit}")
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
            {"id": "deepseek-coder", "object": "model", "owned_by": "gateway"},
            {"id": DEFAULT_CHAT_MODEL, "object": "model", "owned_by": "gateway"}
        ]
    }

async def handle_chat_completion(body: dict, tenant: str):
    messages = body.get("messages", [])
    user_prompt = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content_field = msg.get("content", "")
            if isinstance(content_field, str):
                user_prompt = content_field
            elif isinstance(content_field, list):
                # 兼容多模态或富文本消息
                for part in content_field:
                    if isinstance(part, dict) and part.get("type") == "text":
                        user_prompt = part.get("text", "")
            break

    is_stream = body.get("stream", False)
    client_model = body.get("model", DEFAULT_CHAT_MODEL)

    # 1. 尝试语义缓存命中
    cached_message_json = None
    if vector_index and user_prompt:
        try:
            results = vector_index.query(data=user_prompt, top_k=1, include_metadata=True)
            if results and results[0].score >= 0.92:
                meta = results[0].metadata
                if meta and meta.get("tenant") == tenant:
                    cached_message_json = meta.get("assistant_message")
        except Exception as e:
            print(f"Vector search error: {e}")

    if cached_message_json:
        update_metrics(tenant, hit=True)
        try:
            assistant_message = json.loads(cached_message_json)
        except Exception:
            assistant_message = {"role": "assistant", "content": cached_message_json}

        if is_stream:
            async def cached_stream():
                # 构造兼容流式输出，支持 content 和 tool_calls
                delta_payload = {}
                if assistant_message.get("content"):
                    delta_payload["content"] = assistant_message["content"]
                if assistant_message.get("tool_calls"):
                    delta_payload["tool_calls"] = assistant_message["tool_calls"]

                chunk_data = {
                    "id": "chatcmpl-cached",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": client_model,
                    "choices": [{"index": 0, "delta": delta_payload, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                
                end_chunk = {
                    "id": "chatcmpl-cached",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": client_model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(end_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(cached_stream(), media_type="text/event-stream")
        else:
            return {
                "id": "chatcmpl-cached",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": client_model,
                "choices": [{"index": 0, "message": assistant_message, "finish_reason": "stop"}]
            }

    # 2. 未命中，转发给 DeepSeek 官方
    update_metrics(tenant, hit=False)
    headers = {
        "Authorization": f"Bearer {REAL_LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    
    async def proxy_stream():
        full_content = ""
        collected_tool_calls = {}
        has_sent_finish = False
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", f"{REAL_LLM_BASE_URL.rstrip('/')}/chat/completions", json=body, headers=headers) as response:
                if response.status_code != 200:
                    error_detail = await response.aread()
                    err_msg = error_detail.decode()
                    print(f"Upstream DeepSeek error [{response.status_code}]: {err_msg}")
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
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    full_content += content
                                
                                # 收集流式传输中的 tool_calls
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
            finish_chunk = {
                "id": "chatcmpl-proxy-end",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": client_model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(finish_chunk)}\n\n"
        
        yield "data: [DONE]\n\n"
            
        # 异步写入语义缓存：支持同时缓存文本和工具调用指令
        if vector_index and user_prompt:
            assistant_msg = {"role": "assistant"}
            if full_content:
                assistant_msg["content"] = full_content
            if collected_tool_calls:
                assistant_msg["tool_calls"] = list(collected_tool_calls.values())

            if assistant_msg.get("content") or assistant_msg.get("tool_calls"):
                try:
                    vector_index.upsert(
                        vectors=[{
                            "id": f"t-{tenant}-{int(time.time()*1000)}",
                            "data": user_prompt,
                            "metadata": {
                                "tenant": tenant, 
                                "assistant_message": json.dumps(assistant_msg, ensure_ascii=False)
                            }
                        }]
                    )
                    print(f"💾 Successfully cached response with tool_calls for tenant [{tenant}]")
                except Exception as e:
                    print(f"Vector upsert error: {e}")

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
                if vector_index and user_prompt:
                    vector_index.upsert(
                        vectors=[{
                            "id": f"t-{tenant}-{int(time.time()*1000)}",
                            "data": user_prompt,
                            "metadata": {
                                "tenant": tenant, 
                                "assistant_message": json.dumps(message, ensure_ascii=False)
                            }
                        }]
                    )
            except Exception as e:
                print(f"Non-stream cache upsert error: {e}")
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
        return {"metrics": {"total_requests": 0, "cache_hits": 0, "hit_rate": "0%", "estimated_saved_cost": "$0.00"}}
    try:
        total = int(redis_client.get(f"tenant:{tenant}:total_requests") or 0)
        hits = int(redis_client.get(f"tenant:{tenant}:cache_hits") or 0)
        cost = float(redis_client.get(f"tenant:{tenant}:saved_cost") or 0.0)
        hit_rate = f"{(hits / total * 100):.1f}%" if total > 0 else "0%"
        return {
            "tenant": tenant,
            "metrics": {
                "total_requests": total,
                "cache_hits": hits,
                "hit_rate": hit_rate,
                "estimated_saved_cost": f"${cost:.4f}"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))