import streamlit as st
import requests
import plotly.express as px
import pandas as pd
import time

st.set_page_config(
    page_title="AI Gateway 监控看板",
    layout="wide",
    initial_sidebar_state="expanded"
)

GATEWAY_STATS_URL = "https://ai-gateway-ardy.onrender.com/stats"

st.title("🚀 AI Gateway 实时语义缓存与降本监控看板")
st.markdown("实时监控云端网关的请求量、缓存命中率以及降本效果。")

st.sidebar.header("⚙️ 控制面板")
auto_refresh = st.sidebar.checkbox("开启自动刷新 (5秒)", value=False)
if st.sidebar.button("🔄 立即刷新数据"):
    st.rerun()

def fetch_stats():
    try:
        # 加上 timestamp 参数防止 HTTP 缓存
        response = requests.get(f"{GATEWAY_STATS_URL}?_t={time.time()}", timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"无法获取数据，HTTP 状态码: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"连接云端网关失败: {e}")
        return None

res = fetch_stats()

if res:
    # 嵌套提取 metrics 里的真实数据
    metrics = res.get("metrics", res)
    
    total_requests = metrics.get("total_requests", 0)
    cache_hits = metrics.get("cache_hits", 0)
    cache_misses = total_requests - cache_hits if total_requests >= cache_hits else 0
    
    hit_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0.0
    
    saved_usd = cache_hits * 0.002
    saved_cny = saved_usd * 7.2

    st.markdown("---")

    # 第一排：核心 KPI 卡片
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="📊 总请求量 (Total)", value=f"{total_requests:,}")
    with col2:
        st.metric(label="🎯 缓存命中数 (Hits)", value=f"{cache_hits:,}")
    with col3:
        st.metric(label="⚡ 缓存命中率 (Hit Rate)", value=f"{hit_rate:.1f}%")
    with col4:
        st.metric(label="💰 预估节省成本", value=f"${saved_usd:.3f}", delta=f"≈ ¥{saved_cny:.2f}")

    st.markdown("---")

    # 第二排：图表分析
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("🍰 缓存命中 vs 真实 API 调用占比")
        if total_requests > 0:
            df_pie = pd.DataFrame({
                "类别": ["缓存拦截 (Cache Hit)", "LLM 真实调用 (Cache Miss)"],
                "请求数": [cache_hits, cache_misses]
            })
            fig_pie = px.pie(
                df_pie, 
                values="请求数", 
                names="类别", 
                color="类别",
                color_discrete_map={"缓存拦截 (Cache Hit)": "#00CC96", "LLM 真实调用 (Cache Miss)": "#EF553B"},
                hole=0.4
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("暂无请求数据，请先发起测试请求。")

    with chart_col2:
        st.subheader("⚡ 响应延迟对比估算 (Latency)")
        df_bar = pd.DataFrame({
            "响应来源": ["语义缓存 (Cache Hit)", "Gemini API 真实调用"],
            "平均延迟 (秒)": [0.05, 1.20]
        })
        fig_bar = px.bar(
            df_bar, 
            x="响应来源", 
            y="平均延迟 (秒)", 
            color="响应来源",
            color_discrete_sequence=["#636EFA", "#AB63FA"],
            text_auto='.2f'
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    if auto_refresh:
        time.sleep(5)
        st.rerun()