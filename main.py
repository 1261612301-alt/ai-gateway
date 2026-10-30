import os
import json
import time
from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from upstash_vector import Index
from upstash_redis import Redis

app = FastAPI(title="AI Semantic Cache Gateway", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)

# 初始化 Upstash 客户端 (从环境变量读取)
VECTOR_URL = os.getenv("UPSTASH_VECTOR_REST_URL", "")
VECTOR_TOKEN = os.getenv("UPSTASH_VECTOR_REST_TOKEN", "")
REDIS_URL = os.getenv("UPSTASH_REDIS_REST_URL", "")
REDIS_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")

# 真实大模型后端配置 (例如 Gemini / DeepSeek 兼容接口)
REAL_LLM_API_KEY = os.getenv("REAL_LLM_API_KEY", "")
REAL_LLM_BASE_URL = os.getenv("REAL_LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
DEFAULT_CHAT_MODEL = os.getenv("DEFAULT_CHAT_MODEL", "gemini-2.5-flash")

vector_index = Index(url=VECTOR_URL, token=VECTOR_TOKEN) if VECTOR_URL and VECTOR_TOKEN else None
redis_client = Redis(url=REDIS_URL, token=REDIS_TOKEN) if REDIS_URL and REDIS_TOKEN else None

# API Key 与租户映射关系
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

class ChatRequest(BaseModel):
    model: str = DEFAULT_CHAT_MODEL
    messages: list
    stream: bool = False
    temperature: float = 0.7

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
    except Exception as e:
        print(f"Metrics update error: {e}")

# ==================== 兼容性路由 (防止 DSH 等框架报 404) ====================
@app.get("/v1/models")
@app.get("/models")
@app.get("/v1/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "deepseek-chat", "object": "model", "owned_by": "gateway"},
            {"id": DEFAULT_CHAT_MODEL, "object": "model", "owned_by": "gateway"},
            {"id": "gemini-3.1-flash-lite", "object": "model", "owned_by": "gateway"},
            {"id": "gemini-2.5-flash", "object": "model", "owned_by": "gateway"}
        ]
    }

async def handle_chat_completion(request: ChatRequest, tenant: str):
    user_prompt = ""
    for msg in reversed(request.messages):
        if msg.get("role") == "user":
            user_prompt = msg.get("content", "")
            break

    # 1. 尝试语义缓存命中
    cached_response = None
    if vector_index and user_prompt:
        try:
            results = vector_index.query(data=user_prompt, top_k=1, include_metadata=True)
            if results and results[0].score >= 0.92:
                meta = results[0].metadata
                if meta and meta.get("tenant") == tenant:
                    cached_response = meta.get("response")
        except Exception as e:
            print(f"Vector search error: {e}")

    if cached_response:
        update_metrics(tenant, hit=True)
        if request.stream:
            async def cached_stream():
                chunk_data = {
                    "id": "chatcmpl-cached",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"content": cached_response}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(cached_stream(), media_type="text/event-stream")
        else:
            return {
                "id": "chatcmpl-cached",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": cached_response}, "finish_reason": "stop"}]
            }

    # 2. 未命中，请求真实大模型
    update_metrics(tenant, hit=False)
    headers = {
        "Authorization": f"Bearer {REAL_LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = request.dict()
    
    async def proxy_stream():
        full_content = ""
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", f"{REAL_LLM_BASE_URL.rstrip('/')}/chat/completions", json=payload, headers=headers) as response:
                if response.status_code != 200:
                    yield f"data: {json.dumps({'error': 'upstream error'})}\n\n"
                    return
                async for line in response.aiter_lines():
                    if line:
                        yield line + "\n\n"
                        if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                            try:
                                data = json.loads(line[6:])
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    full_content += content
                            except Exception:
                                pass
        # 异步写入语义缓存
        if vector_index and user_prompt and full_content:
            try:
                vector_index.upsert(
                    vectors=[{
                        "id": f"t-{tenant}-{int(time.time()*1000)}",
                        "data": user_prompt,
                        "metadata": {"tenant": tenant, "response": full_content}
                    }]
                )
            except Exception as e:
                print(f"Vector upsert error: {e}")

    if request.stream:
        return StreamingResponse(proxy_stream(), media_type="text/event-stream")
    else:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{REAL_LLM_BASE_URL.rstrip('/')}/chat/completions", json=payload, headers=headers)
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            data = response.json()
            try:
                full_content = data["choices"][0]["message"]["content"]
                if vector_index and user_prompt and full_content:
                    vector_index.upsert(
                        vectors=[{
                            "id": f"t-{tenant}-{int(time.time()*1000)}",
                            "data": user_prompt,
                            "metadata": {"tenant": tenant, "response": full_content}
                        }]
                    )
            except Exception:
                pass
            return data

# ==================== 多路径对话核心路由 ====================
@app.post("/v1/chat/completions")
@app.post("/chat/completions")
@app.post("/v1/v1/chat/completions")
async def chat_completions(request: ChatRequest, credentials: HTTPAuthorizationCredentials | None = Security(security)):
    tenant = get_tenant_from_credentials(credentials)
    return await handle_chat_completion(request, tenant)

# ==================== 租户统计接口 ====================
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