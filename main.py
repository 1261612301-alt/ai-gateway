import os
import time
import uuid
import json
import asyncio

from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from google import genai
from pydantic import BaseModel, ConfigDict
from upstash_redis import Redis
from upstash_vector import Index

# ==================== 1. 配置信息 ====================
def require_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    pretty = " 或 ".join(names)
    raise RuntimeError(
        f"启动失败：未检测到环境变量 {pretty}。"
        "请在云平台（或本地）配置后再启动服务。"
    )

GEMINI_API_KEY = require_env("GEMINI_API_KEY")
UPSTASH_REDIS_URL = require_env("UPSTASH_REDIS_URL")
UPSTASH_REDIS_TOKEN = require_env("UPSTASH_REDIS_TOKEN")
UPSTASH_VECTOR_URL = require_env("UPSTASH_VECTOR_URL", "UPSTASH_VECTOR_REST_URL")
UPSTASH_VECTOR_TOKEN = require_env("UPSTASH_VECTOR_TOKEN", "UPSTASH_VECTOR_REST_TOKEN")

EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
DEFAULT_CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.1-flash-lite")
CHAT_FALLBACKS = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-3.8-flash",
]
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.92"))

# ==================== 2. 初始化 ====================
app = FastAPI(
    title="AI Semantic Cache Gateway API - Multi-Tenant",
    version="2.2.0",
    description="支持 API Key 多租户隔离、语义检索与流式传输的 AI 降本网关",
)

ai_client = genai.Client(api_key=GEMINI_API_KEY)
redis_client = Redis(url=UPSTASH_REDIS_URL, token=UPSTASH_REDIS_TOKEN)
vector_index = Index(url=UPSTASH_VECTOR_URL, token=UPSTASH_VECTOR_TOKEN)
_INDEX_DIMENSION: int | None = None

security = HTTPBearer(auto_error=False)

# 模拟多租户 API Key 数据库（实际生产中可存入 Redis 或数据库）
# 格式: {"api_key": "tenant_name"}
VALID_CLIENTS = {
    "sk-gateway-demo001": "企业客户A-电商业务部",
    "sk-gateway-vip002": "企业客户B-智能客服部",
    "sk-gateway-free": "免费体验用户"
}

def get_tenant_by_key(credentials: HTTPAuthorizationCredentials | None) -> str:
    """验证 API Key 并返回对应的租户名称，未提供或错误则归为默认/或报错"""
    if not credentials:
        # 为了兼容前期测试，如果没有传 Bearer Token，暂时归为默认公共租户
        return "public-default-tenant"
    
    token = credentials.credentials
    if token in VALID_CLIENTS:
        return VALID_CLIENTS[token]
    
    # 如果是以 sk- 开头的新客户 Key，也可以动态放行并自动注册
    if token.startswith("sk-"):
        return f"client-{token[-6:]}"
        
    raise HTTPException(status_code=401, detail="无效的 API Key，请检查 Authorization 请求头")


def get_index_dimension() -> int:
    global _INDEX_DIMENSION
    if _INDEX_DIMENSION is None:
        info = vector_index.info()
        _INDEX_DIMENSION = int(getattr(info, "dimension", 0) or 0)
        print(f"Upstash Vector 维度: {_INDEX_DIMENSION}")
    return _INDEX_DIMENSION


def fit_vector(values: list[float], dimension: int) -> list[float]:
    if dimension <= 0 or len(values) == dimension:
        return values
    if len(values) > dimension:
        return values[:dimension]
    return values + [0.0] * (dimension - len(values))


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: str | list | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = DEFAULT_CHAT_MODEL
    messages: list[Message]
    stream: bool | None = False


def extract_user_prompt(messages: list[Message]) -> str:
    for msg in reversed(messages):
        if msg.role != "user":
            continue
        content = msg.content
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str) and item.strip():
                    parts.append(item.strip())
                elif isinstance(item, dict) and str(item.get("text") or "").strip():
                    parts.append(str(item["text"]).strip())
            if parts:
                return "\n".join(parts)
    raise HTTPException(status_code=400, detail="messages 中未找到 role=user 的有效 content")


def embed_prompt(prompt: str) -> list[float]:
    dimension = get_index_dimension()
    embed_kwargs: dict = {"model": EMBED_MODEL, "contents": prompt}
    if dimension in {768, 1536, 3072}:
        embed_kwargs["config"] = {"output_dimensionality": dimension}
    result = ai_client.models.embed_content(**embed_kwargs)
    values: list[float] | None = None
    if getattr(result, "embeddings", None):
        values = list(result.embeddings[0].values)
    else:
        embedding = getattr(result, "embedding", None)
        if embedding is not None:
            values = list(embedding.values)
    if not values:
        raise RuntimeError("嵌入模型未返回向量")
    return fit_vector(values, dimension)


def query_semantic_cache(query_vector: list[float]):
    results = vector_index.query(
        vector=query_vector,
        top_k=1,
        include_metadata=True,
    )
    if not results:
        return None, 0.0
    top = results[0]
    score = float(getattr(top, "score", 0.0) or 0.0)
    metadata = getattr(top, "metadata", None) or {}
    return metadata, score


def save_semantic_cache(prompt: str, ai_response: str, query_vector: list[float], tenant: str) -> None:
    vector_index.upsert(
        vectors=[
            {
                "id": str(uuid.uuid4()),
                "vector": query_vector,
                "metadata": {
                    "prompt": prompt,
                    "response": ai_response,
                    "ai_response": ai_response,
                    "tenant": tenant
                },
            }
        ]
    )


# ==================== 3. 核心多租户语义缓存路由 ====================
@app.post("/v1/chat/completions")
def chat_gateway(request: ChatRequest, credentials: HTTPAuthorizationCredentials | None = Security(security)):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages 不能为空")

    tenant_name = get_tenant_by_key(credentials)
    start_time = time.time()
    user_prompt = extract_user_prompt(request.messages)

    # 按租户隔离的 Redis 统计 Key
    total_key = f"stats:total:{tenant_name}"
    hits_key = f"stats:hits:{tenant_name}"

    try:
        redis_client.incr("stats:total_requests")  # 全局总数
        redis_client.incr(total_key)              # 租户总数
        
        query_vector = embed_prompt(user_prompt)
        metadata, similarity_score = query_semantic_cache(query_vector)
        cached_response = None
        if metadata:
            cached_response = metadata.get("ai_response") or metadata.get("response")

        # ----------------------------------------------------
        # 情况 A：命中语义缓存
        # ----------------------------------------------------
        if cached_response and similarity_score >= SIMILARITY_THRESHOLD:
            redis_client.incr("stats:cache_hits")  # 全局命中
            redis_client.incr(hits_key)           # 租户命中
            
            if request.stream:
                async def fake_stream_generator():
                    chunk_size = 2
                    for i in range(0, len(cached_response), chunk_size):
                        chunk = cached_response[i:i+chunk_size]
                        chunk_data = {
                            "id": "chatcmpl-gateway",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}]
                        }
                        yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                        await asyncio.sleep(0.01)
                    
                    final_chunk = {
                        "id": "chatcmpl-gateway",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                    }
                    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(fake_stream_generator(), media_type="text/event-stream")

            latency_ms = round((time.time() - start_time) * 1000, 2)
            return {
                "id": "chatcmpl-gateway",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": cached_response},
                        "finish_reason": "stop",
                    }
                ],
                "gateway_meta": {
                    "tenant": tenant_name,
                    "cached": True,
                    "similarity": round(similarity_score, 4),
                    "latency_ms": f"{latency_ms} ms",
                    "message": f"【语义缓存命中】相似度 {round(similarity_score * 100, 2)}%",
                },
            }

        # ----------------------------------------------------
        # 情况 B：未命中缓存，调用大模型
        # ----------------------------------------------------
        candidates: list[str] = []
        if request.model and request.model.startswith("gemini") and "2.5" not in request.model:
            candidates.append(request.model)
        for name in CHAT_FALLBACKS:
            if name not in candidates:
                candidates.append(name)

        if request.stream:
            async def llm_stream_generator():
                full_text = ""
                used_model = request.model
                success = False
                last_exc = None

                for model in candidates:
                    try:
                        used_model = model
                        response_stream = ai_client.models.generate_content_stream(
                            model=model, contents=user_prompt
                        )
                        for chunk in response_stream:
                            text_chunk = getattr(chunk, "text", "")
                            if text_chunk:
                                full_text += text_chunk
                                chunk_data = {
                                    "id": "chatcmpl-gateway",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": used_model,
                                    "choices": [{"index": 0, "delta": {"content": text_chunk}, "finish_reason": None}]
                                }
                                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                        success = True
                        break
                    except Exception as exc:
                        last_exc = exc
                        err_str = str(exc).upper()
                        if any(token in err_str for token in ("404", "NOT_FOUND", "NO LONGER AVAILABLE", "503", "UNAVAILABLE", "429")):
                            continue
                        break

                if not success:
                    err_chunk = {
                        "id": "chatcmpl-gateway",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{"index": 0, "delta": {"content": f"\n[网关错误: Gemini 调用失败: {last_exc}]"}, "finish_reason": "stop"}]
                    }
                    yield f"data: {json.dumps(err_chunk, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                if full_text:
                    try:
                        save_semantic_cache(user_prompt, full_text, query_vector, tenant_name)
                    except Exception as e:
                        print(f"⚠️ 缓存写入异常: {e}")

                final_chunk = {
                    "id": "chatcmpl-gateway",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": used_model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(llm_stream_generator(), media_type="text/event-stream")

        # 非流式调用 Gemini
        last_error = None
        ai_response = ""
        used_model = request.model
        for model in candidates:
            try:
                completion = ai_client.models.generate_content(model=model, contents=user_prompt)
                ai_response = (completion.text or "").strip()
                if ai_response:
                    used_model = model
                    break
            except Exception as exc:
                last_error = exc
                continue
        
        if not ai_response:
            raise RuntimeError(f"Gemini 调用失败: {last_error}")

        save_semantic_cache(user_prompt, ai_response, query_vector, tenant_name)
        latency_ms = round((time.time() - start_time) * 1000, 2)
        return {
            "id": "chatcmpl-gateway",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": used_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ai_response},
                    "finish_reason": "stop",
                }
            ],
            "gateway_meta": {
                "tenant": tenant_name,
                "cached": False,
                "similarity": round(similarity_score, 4),
                "latency_ms": f"{latency_ms} ms",
                "message": "【未命中缓存】已调用大模型，并写入 Upstash Vector",
            },
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"网关处理失败: {exc}") from exc


# ==================== 4. 统计与探针 ====================
@app.get("/stats")
def get_stats(tenant: str | None = None):
    """获取统计数据。若传入 tenant 参数，则返回该租户的专属数据；否则返回全局数据。"""
    try:
        if tenant:
            total_requests = int(redis_client.get(f"stats:total:{tenant}") or 0)
            cache_hits = int(redis_client.get(f"stats:hits:{tenant}") or 0)
        else:
            total_requests = int(redis_client.get("stats:total_requests") or 0)
            cache_hits = int(redis_client.get("stats:cache_hits") or 0)

        hit_rate = f"{round((cache_hits / total_requests) * 100, 2)}%" if total_requests else "0%"
        return {
            "status": "success",
            "tenant": tenant or "global",
            "metrics": {
                "total_requests": total_requests,
                "cache_hits": cache_hits,
                "hit_rate": hit_rate,
                "estimated_saved_cost": f"${round(cache_hits * 0.002, 4)}",
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"获取统计失败: {exc}") from exc


@app.get("/")
def read_root():
    return {
        "service": "AI Semantic Cache Gateway - Multi-Tenant",
        "status": "ok",
        "docs": "/docs",
        "chat": "POST /v1/chat/completions",
        "stats": "GET /stats?tenant=租户名",
    }

@app.get("/v1/models")
def list_models():
    """兼容 OpenAI 协议的模型列表探针，供 DSH 等框架初始化检查"""
    return {
        "object": "list",
        "data": [
            {
                "id": "deepseek-chat",
                "object": "model",
                "owned_by": "gateway"
            },
            {
                "id": DEFAULT_CHAT_MODEL,
                "object": "model",
                "owned_by": "gateway"
            },
            {
                "id": "gemini-3.1-flash-lite",
                "object": "model",
                "owned_by": "gateway"
            },
            {
                "id": "gemini-3.5-flash-lite",
                "object": "model",
                "owned_by": "gateway"
            }
        ]
    }