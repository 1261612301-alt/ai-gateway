import streamlit as st
import requests
import json

# 页面配置
st.set_page_config(
    page_title="AI 语义缓存降本网关 - 多租户商业控制台",
    page_icon="⚡",
    layout="wide"
)

# 侧边栏：租户认证与配置
st.sidebar.title("🔐 租户控制台")
api_base_url = st.sidebar.text_input(
    "后端网关 API 地址 (Render URL)", 
    value="https://ai-gateway-ardy.onrender.com"
).strip()

api_key = st.sidebar.text_input(
    "API Key (租户密钥)", 
    value="sk-gateway-demo001",
    type="password",
    help="输入你的专属 API Key，例如 sk-gateway-demo001 或 sk-gateway-vip002"
).strip()

tenant_mapping = {
    "sk-gateway-demo001": "企业客户A-电商业务部",
    "sk-gateway-vip002": "企业客户B-智能客服部",
    "sk-gateway-free": "免费体验用户"
}
current_tenant = tenant_mapping.get(api_key, f"client-{api_key[-6:]}" if len(api_key) > 6 else "public-default-tenant")

st.sidebar.markdown("---")
st.sidebar.info(
    f"👤 **当前登录租户**：\n`{current_tenant}`\n\n"
    "💡 **商业化提示**：\n"
    "- 若初次请求较慢，可能是 Render 免费版正在唤醒（约需 30 秒）。"
)

tab1, tab2 = st.tabs(["💬 AI 智能对话测试", "📊 专属降本数据大屏"])

# ==================== Tab 1: AI 智能对话演练场 ====================
with tab1:
    st.header("💬 语义缓存与流式对话网关演示")
    st.markdown("体验大模型流式吐字，尝试**连续发送两次相同或语义相近的问题**，观察第二次的极致速度！")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("输入你的问题（例如：公司年假怎么休？）..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                payload = {
                    "model": "gemini-3.1-flash-lite",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True
                }
                url = f"{api_base_url.rstrip('/')}/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                with st.spinner("🤖 正在连接网关并生成回答（若服务器刚唤醒可能需要数秒）..."):
                    response = requests.post(url, json=payload, headers=headers, stream=True, timeout=60)
                
                if response.status_code != 200:
                    message_placeholder.error(f"网关响应错误 [{response.status_code}]: {response.text}")
                else:
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
                                        message_placeholder.markdown(full_response + "▌")
                                except Exception:
                                    pass
                    
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
            except Exception as e:
                message_placeholder.error(f"连接后端网关失败（可能服务器正在冷启动，请稍后重试）: {e}")

# ==================== Tab 2: 降本增效数据大屏 ====================
with tab2:
    st.header(f"📊 租户专属降本与缓存命中大屏 [{current_tenant}]")
    st.markdown("实时追踪当前 API Key 对应的资源消耗与降本收益。")

    if st.button("🔄 刷新统计数据", type="primary"):
        pass

    try:
        stats_url = f"{api_base_url.rstrip('/')}/stats?tenant={current_tenant}"
        with st.spinner("📊 正在同步云端统计数据..."):
            res = requests.get(stats_url, timeout=15)
        
        if res.status_code == 200:
            stats_data = res.json().get("metrics", {})
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("总请求数 (Total Requests)", stats_data.get("total_requests", 0))
            col2.metric("缓存命中数 (Cache Hits)", stats_data.get("cache_hits", 0))
            col3.metric("缓存命中率 (Hit Rate)", stats_data.get("hit_rate", "0%"))
            col4.metric("为您节省成本 (Saved Cost)", stats_data.get("estimated_saved_cost", "$0"))
            
            st.success(f"✨ 数据已成功与租户 [{current_tenant}] 的云端隔离存储同步！")
        else:
            st.warning("⚠️ 暂时无法获取后端统计数据，请检查 API 地址是否正确。")
    except Exception as e:
        st.warning(f"⚠️ 无法连接到统计接口（可能正在唤醒）: {e}")