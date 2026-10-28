import os
import time
import uuid

from fastapi import FastAPI, HTTPException
from google import genai
from pydantic import BaseModel, ConfigDict
from upstash_redis import Redis
from upstash_vector import Index

# ==================== 1. 配置信息（仅从环境变量读取，禁止硬编码密钥） ====================
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
STATS_TOTAL_KEY = "stats:total_requests"
STATS_HITS_KEY = "stats:cache_hits"

# ==================== 2. 初始化 ====================
app = FastAPI(
    title="AI Semantic Cache Gateway API",
    version="2.0.0",
    description="基于 Upstash Vector 语义向量检索的 AI 降本网关",
)

ai_client = genai.Client(api_key=GEMINI_API_KEY)
redis_client = Redis(url=UPSTASH_REDIS_URL, token=UPSTASH_REDIS_TOKEN)
vector_index = Index(url=UPSTASH_VECTOR_URL, token=UPSTASH_VECTOR_TOKEN)
_INDEX_DIMENSION: int | None = None


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


def ask_gemini(prompt: str, preferred_model: str) -> tuple[str, str]:
    candidates: list[str] = []
    if preferred_model and preferred_model.startswith("gemini") and "2.5" not in preferred_model:
        candidates.append(preferred_model)
    for name in CHAT_FALLBACKS:
        if name not in candidates:
            candidates.append(name)

    last_error: Exception | None = None
    for model in candidates:
        try:
            completion = ai_client.models.generate_content(model=model, contents=prompt)
            text = (completion.text or "").strip()
            if text:
                return text, model
        except Exception as exc:
            last_error = exc
            text = str(exc).upper()
            if any(token in text for token in ("404", "NOT_FOUND", "NO LONGER AVAILABLE", "503", "UNAVAILABLE", "429")):
                continue
            raise
    raise RuntimeError(f"Gemini 调用失败: {last_error}")


def save_semantic_cache(prompt: str, ai_response: str, query_vector: list[float]) -> None:
    vector_index.upsert(
        vectors=[
            {
                "id": str(uuid.uuid4()),
                "vector": query_vector,
                "metadata": {
                    "prompt": prompt,
                    "response": ai_response,
                    "ai_response": ai_response,
                },
            }
        ]
    )


# ==================== 3. 核心语义缓存路由 ====================
@app.post("/v1/chat/completions")
def chat_gateway(request: ChatRequest):
    if request.stream:
        raise HTTPException(status_code=400, detail="当前网关暂不支持 stream=true")
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages 不能为空")

    start_time = time.time()
    user_prompt = extract_user_prompt(request.messages)

    try:
        redis_client.incr(STATS_TOTAL_KEY)
        query_vector = embed_prompt(user_prompt)
        metadata, similarity_score = query_semantic_cache(query_vector)
        cached_response = None
        if metadata:
            cached_response = metadata.get("ai_response") or metadata.get("response")

        if cached_response and similarity_score >= SIMILARITY_THRESHOLD:
            latency_ms = round((time.time() - start_time) * 1000, 2)
            redis_client.incr(STATS_HITS_KEY)
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
                    "cached": True,
                    "similarity": round(similarity_score, 4),
                    "latency_ms": f"{latency_ms} ms",
                    "message": f"【语义缓存命中】相似度 {round(similarity_score * 100, 2)}%",
                },
            }

        ai_response, used_model = ask_gemini(user_prompt, request.model)
        save_semantic_cache(user_prompt, ai_response, query_vector)
        latency_ms = round((time.time() - start_time) * 1000, 2)
        return {
            "id": "chatcmpl-gateway",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": used_model or request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ai_response},
                    "finish_reason": "stop",
                }
            ],
            "gateway_meta": {
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
def get_stats():
    try:
        total_requests = int(redis_client.get(STATS_TOTAL_KEY) or 0)
        cache_hits = int(redis_client.get(STATS_HITS_KEY) or 0)
        hit_rate = f"{round((cache_hits / total_requests) * 100, 2)}%" if total_requests else "0%"
        return {
            "status": "success",
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
        "service": "AI Semantic Cache Gateway",
        "status": "ok",
        "docs": "/docs",
        "chat": "POST /v1/chat/completions",
        "stats": "GET /stats",
    }
