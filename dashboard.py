import streamlit as st
import requests
import json

# 页面配置
st.set_page_config(
    page_title="AI 语义缓存降本网关 - 商业化控制台",
    page_icon="⚡",
    layout="wide"
)

# 侧边栏配置
st.sidebar.title("⚙️ 网关控制台")
api_base_url = st.sidebar.text_input(
    "后端网关 API 地址 (Render URL)", 
    value="https://ai-gateway-ardy.onrender.com"  
).strip()

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **商业化亮点**：\n"
    "- **毫秒级响应**：缓存命中时 50ms 秒回。\n"
    "- **极致降本**：高频重复问题绕过大模型，省下 100% Token 费用。"
)

# 主界面 Tabs
tab1, tab2 = st.tabs(["💬 AI 智能对话测试", "📊 降本增效数据大屏"])

# ==================== Tab 1: AI 智能对话演练场 ====================
with tab1:
    st.header("💬 语义缓存与流式对话网关演示")
    st.markdown("体验大模型流式吐字，尝试**连续发送两次相同或语义相近的问题**，观察第二次的极致速度！")

    # 初始化对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 渲染历史消息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 用户输入
    if prompt := st.chat_input("输入你的问题（例如：公司年假怎么休？）..."):
        # 记录用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 助理响应框
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # 构造符合 OpenAI 格式的流式请求
                payload = {
                    "model": "gemini-3.1-flash-lite",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True
                }
                url = f"{api_base_url.rstrip('/')}/v1/chat/completions"
                
                # 发起 HTTP Stream 请求
                response = requests.post(url, json=payload, stream=True, timeout=30)
                
                if response.status_code != 200:
                    message_placeholder.error(f"网关响应错误: {response.status_code} - {response.text}")
                else:
                    # 逐行解析 SSE 流数据
                    for line in response.iter_lines():
                        if line:
                            decoded = line.decode('utf-8')
                            if decoded.startswith("data: "):
                                data_str = decoded[6:]
                                if data_str == "[DONE]":
                                    break
                                try:
                                    data = json.loads(data_str)
                                    delta = data["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        full_response += content
                                        # 实时展现打字机效果
                                        message_placeholder.markdown(full_response + "▌")
                                except Exception:
                                    pass
                    
                    # 最终去除光标
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
            except Exception as e:
                message_placeholder.error(f"连接后端网关失败: {e}")

# ==================== Tab 2: 降本增效数据大屏 ====================
with tab2:
    st.header("📊 实时降本与缓存命中大屏")
    st.markdown("通过 Upstash Redis 与 Vector 实时追踪网关的经济效益。")

    if st.button("🔄 刷新统计数据", type="primary"):
        pass

    try:
        stats_url = f"{api_base_url.rstrip('/')}/stats"
        res = requests.get(stats_url, timeout=5)
        
        if res.status_code == 200:
            stats_data = res.json().get("metrics", {})
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("总请求数 (Total Requests)", stats_data.get("total_requests", 0))
            col2.metric("缓存命中数 (Cache Hits)", stats_data.get("cache_hits", 0))
            col3.metric("缓存命中率 (Hit Rate)", stats_data.get("hit_rate", "0%"))
            col4.metric("预估节省成本 (Saved Cost)", stats_data.get("estimated_saved_cost", "$0"))
            
            st.success("✨ 数据已与 Render 后端及 Upstash 实时同步！")
        else:
            st.warning("⚠️ 暂时无法获取后端统计数据，请检查 API 地址是否正确。")
    except Exception as e:
        st.warning(f"⚠️ 无法连接到统计接口: {e}")