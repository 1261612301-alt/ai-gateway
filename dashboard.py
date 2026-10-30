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
    "💡 **商业化亮点**：\n"
    "- **多租户隔离**：各企业数据与配额完全独立。\n"
    "- **毫秒级响应**：缓存命中时 50ms 秒回。\n"
    "- **极致降本**：高频重复问题省下 100% Token 费用。"
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