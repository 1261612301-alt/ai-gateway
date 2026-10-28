from openai import OpenAI, APIStatusError

client = OpenAI(
    api_key="anything",
    base_url="http://127.0.0.1:8000/v1",
    timeout=120.0,
    max_retries=0,
)

try:
    response = client.chat.completions.create(
        model="gemini-3.1-flash-lite",
        messages=[{"role": "user", "content": "怎么修改微信密码？"}],
    )
except APIStatusError as exc:
    print("HTTP", exc.status_code)
    print(exc.response.text if exc.response is not None else exc)
    raise

print("AI 回答:", response.choices[0].message.content)
print("gateway_meta:", getattr(response, "gateway_meta", None) or response.model_extra)
