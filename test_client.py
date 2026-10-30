import requests
import json

url = "https://ai-gateway-ardy.onrender.com/v1/chat/completions"
payload = {
    "model": "gemini-3.1-flash-lite",
    "messages": [{"role": "user", "content": "请用一句话介绍什么是语义缓存"}],
    "stream": True  # 开启流式传输
}

print("🤖 正在发送请求...")
response = requests.post(url, json=payload, stream=True)

for line in response.iter_lines():
    if line:
        decoded = line.decode('utf-8')
        if decoded.startswith("data: "):
            data_str = decoded[6:]
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
                content = data["choices"][0]["delta"].get("content", "")
                print(content, end="", flush=True)
            except Exception:
                pass
print("\n")