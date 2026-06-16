import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
import google.generativeai as genai

# ==========================================
# 1. 基础配置与轻量状态管理
# ==========================================
st.set_page_config(page_title="SKU-Doctor 智能数据集体检与决策系统", layout="wide")

# 【核心防错】用 session_state 锁死数据，一旦计算成功，除非重新上传，否则绝不触发二次重复计算
if "analyzed" not in st.session_state: st.session_state.analyzed = False
if "df_cleaned" not in st.session_state: st.session_state.df_cleaned = None
if "df_hierarchical" not in st.session_state: st.session_state.df_hierarchical = None
if "diagnostic_report" not in st.session_state: st.session_state.diagnostic_report = {}
if "chat_history" not in st.session_state: st.session_state.chat_history = []

COLUMN_MAP = {
    'product_id': ['Product_ID', 'StockCode', 'product_id', '产品ID', '商品编码', 'sku'],
    'category': ['Category', 'product_category_name', '产品品类', '品类', 'category_name_1'],
    'price': ['Final_Price(Rs.)', 'UnitPrice', 'Price (Rs.)', 'price', '单价', '销售额'],
    'sales': ['order_item_id', 'Quantity', '销量', '数量', 'qty_ordered']
}

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. 纯本地核心计算引擎 (加入严格的沙盒隔离，根治 KeyError)
# ==========================================
def process_dataset_pure_local(uploaded_file):
    # 强制重新读取原始 CSV，断绝上一次重绘带来的缓存污染
    uploaded_file.seek(0)
    df_raw = pd.read_csv(uploaded_file, low_memory=False)
    
    # 显式初始化，防止 Scope 溢出
    df_mapped = pd.DataFrame()
    
    # 1. 严格映射
    for logic, candidates in COLUMN_MAP.items():
        match = [c for c in df_raw.columns if c in candidates]
        if match: 
            df_mapped[logic] = df_raw[match[0]].copy()
        
    # 【根治 KeyError】如果实在没匹配上，强制给予保底列，防止后续 groupby 报错
    if 'category' not in df_mapped.columns:
        df_mapped['category'] = '未分类未知品类'
    else:
        df_mapped['category'] = df_mapped['category'].fillna('未分类未知品类')
        
    if 'price' not in df_mapped.columns: df_mapped['price'] = 0.0
    if df_mapped['price'].dtype == 'object':
        df_mapped['price'] = df_mapped['price'].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df_mapped['price'] = pd.to_numeric(df_mapped['price'], errors='coerce')
    df_mapped['price'] = df_mapped['price'].fillna(0.0)
    
    if 'sales' in df_mapped.columns: 
        df_mapped['actual_sales'] = df_mapped['sales']
    else: 
        df_mapped['actual_sales'] = 1

    # 3. 聚合与 ABC 分析
    df_product = df_mapped.groupby(['category', 'product_id']).agg(
        total_revenue=('price', 'sum'), 
        sales_count=('actual_sales', 'count' if 'sales' not in df_mapped.columns else 'sum')
    ).reset_index()
    
    df_category = df_product.groupby('category').agg(
        total_revenue=('total_revenue', 'sum'), 
        sales_count=('sales_count', 'sum')
    ).reset_index().sort_values(by='total_revenue', ascending=False)
    
    df_category['cum_pct'] = df_category['total_revenue'].cumsum() / df_category['total_revenue'].sum()
    df_category['rank'] = df_category['cum_pct'].apply(lambda x: 'A' if x <= 0.8 else ('B' if x <= 0.95 else 'C'))
    
    # 4. 构建穿透大表
    hierarchical_rows = []
    for _, cat_row in df_category.iterrows():
        cat_name = cat_row['category']
        hierarchical_rows.append({
            '数据层级': '【品类大类】', '名称/编码ID': cat_name,
            '总销售额(利润)': cat_row['total_revenue'], '总销量': cat_row['sales_count'], '决策分级': cat_row['rank']
        })
        cat_products = df_product[df_product['category'] == cat_name].sort_values(by='total_revenue', ascending=False)
        for _, prod_row in cat_products.iterrows():
            hierarchical_rows.append({
                '数据层级': '  └─ 具体单品 SKU', '名称/编码ID': prod_row['product_id'],
                '总销售额(利润)': prod_row['total_revenue'], '总销量': prod_row['sales_count'], '决策分级': '单品穿透'
            })
            
    df_hierarchical = pd.DataFrame(hierarchical_rows)
    
    # 5. 动态警告阈值
    a_cats = df_category[df_category['rank'] == 'A']
    c_cats = df_category[df_category['rank'] == 'C']
    avg_rev_A = a_cats['total_revenue'].mean() if not a_cats.empty else 1
    avg_rev_C = c_cats['total_revenue'].mean() if not c_cats.empty else 0
    ratio = avg_rev_C / avg_rev_A if avg_rev_A > 0 else 0
    
    if ratio <= 0.10:
        warning = f"🚨 【品类裁剪预警】C类均值仅为A类的 {ratio*100:.1f}%，大类间两极分化严重。建议：果断放弃或整体裁剪整个C类大类。"
    elif ratio >= 0.90:
        warning = f"⚖️ 【微观结构预警】C类均值达A类的 {ratio*100:.1f}%，表现大体平衡。建议：不宜盲目砍掉品类，请执行底层具体单品淘汰。"
    else:
        warning = f"📊 【混合协同预警】C类均值为A类的 {ratio*100:.1f}%，处于中游。建议：维持宏观大类现状，精准微调低效单品。"

    report_dict = {
        "warning": warning,
        "top_cat": df_category.head(3), "bottom_cat": df_category.tail(3),
        "top_prod": df_product.sort_values(by='total_revenue', ascending=False).head(3)
    }
    return df_category, df_hierarchical, report_dict

# ==========================================
# 3. AI 调用路由
# ==========================================
def call_ai_consultant(provider, api_key, prompt):
    try:
        if provider == "DeepSeek (国内直连)":
            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            response = client.chat.completions.create(
                model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0.3
            )
            return response.choices[0].message.content
        else:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            return response.text
    except Exception as e:
        return f"🔑 AI 接入失败。错误信息：{str(e)}"

# ==========================================
# 4. 主界面渲染
# ==========================================
with st.sidebar:
    st.markdown("### 📁 1. 数据源导入")
    # 【核心防错】对于大文件上传，显式通知限制与类型
    uploaded_file = st.file_uploader("请上传 CSV 销售流水数据集 (大文件推荐本地模式运行)", type=["csv"])
    
    if uploaded_file and st.button("🚀 开始自动化数据体检", use_container_width=True):
        with st.spinner("本地算法正在极速洗数..."):
            df_cat, df_hier, r_dict = process_dataset_pure_local(uploaded_file)
            # 将结果永久锁在 session_state 里，哪怕网页刷新，只要不点这个按钮，数据就不会丢
            st.session_state.df_cleaned = df_cat
            st.session_state.df_hierarchical = df_hier
            st.session_state.diagnostic_report = r_dict
            st.session_state.analyzed = True
            st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("### 🤖 2. 决策 AI 专家外挂 (可选)")
    ai_provider = st.selectbox("选择 AI 引擎", ["DeepSeek (国内直连)", "Google Gemini"])
    api_key = st.text_input(f"配置 {ai_provider} 密钥", type="password", placeholder="留空则不启用AI")

st.markdown("<h2 style='text-align: center; color: #1E3A8A;'>⚕️ SKU-Doctor 智能数据集体检与决策系统</h2>", unsafe_allow_html=True)

if not st.session_state.analyzed:
    st.info("💡 欢迎使用 SKU-Doctor。请在左侧导入您需要体检的原始 CSV 格式数据集。")
else:
    col_left, col_right = st.columns([11, 9])
    
    # 左侧：硬核本地清洗产物（完全读取缓存状态，永不报错）
    with col_left:
        st.markdown("### 📊 本地数智清洗与核心结论")
        st.warning(st.session_state.diagnostic_report["warning"])
        
        col_g1, col_g2 = st.columns(2)
        df_c = st.session_state.df_cleaned
        with col_g1:
            fig1, ax1 = plt.subplots(figsize=(6, 4.5))
            ax1.pie(df_c['total_revenue'], labels=df_c['category'], autopct='%1.1f%%', startangle=90)
            ax1.set_title("各大类整体利润贡献占比")
            st.pyplot(fig1)
            plt.close(fig1)
        with col_g2:
            fig2, ax2 = plt.subplots(figsize=(6, 4.5))
            comp_df = pd.concat([st.session_state.diagnostic_report["top_cat"], st.session_state.diagnostic_report["bottom_cat"]])
            ax2.barh(comp_df['category'], comp_df['total_revenue'], color=['#2ca02c']*len(st.session_state.diagnostic_report["top_cat"]) + ['#d62728']*len(st.session_state.diagnostic_report["bottom_cat"]))
            ax2.set_title("最优 Top 3 与 最劣 Bottom 3 对比")
            ax2.invert_yaxis()
            st.pyplot(fig2)
            plt.close(fig2)
            
        st.markdown("#### 📥 标准化成果一键下载")
        col_d1, col_d2 = st.columns(2)
        col_d1.download_button(
            label="💾 导出：清洗后文件.csv", 
            data=st.session_state.df_cleaned.to_csv(index=False).encode('utf-8-sig'),
            file_name="清洗后文件.csv", mime="text/csv", use_container_width=True
        )
        col_d2.download_button(
            label="💾 导出：分析后穿透大表.csv", 
            data=st.session_state.df_hierarchical.to_csv(index=False).encode('utf-8-sig'),
            file_name="分析后穿透大表.csv", mime="text/csv", use_container_width=True
        )
        st.dataframe(st.session_state.df_hierarchical, height=350, use_container_width=True)

    # 右侧：可选 AI 解惑栏
    with col_right:
        st.markdown("### 💬 结论不理解？唤醒 AI 智能解惑")
        if not api_key:
            st.info("🔒 当前处于纯本地完全离线模式。如需激活右侧 AI 智囊，请在左侧侧边栏配置 API Key。")
        else:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]): st.markdown(msg["content"])
            
            if chat_input := st.chat_input("针对左侧图表、ABC分析结论，向 AI 咨询或请求模拟答辩...", key="sku_doctor_chat_v2"):
                with st.chat_message("user"): st.markdown(chat_input)
                st.session_state.chat_history.append({"role": "user", "content": chat_input})
                
                with st.chat_message("assistant"):
                    with st.spinner("AI 智囊正在透视底层商业故事..."):
                        context = f"""
                        你是一个高级商业咨询专家。用户正在查看 SKU 诊断报告：
                        - 本地系统的警告结论是：{st.session_state.diagnostic_report['warning']}
                        - 表现最好的前三个大类是：{st.session_state.diagnostic_report['top_cat']['category'].tolist()}
                        请回答用户的问题：{chat_input}
                        """
                        reply = call_ai_consultant(ai_provider, api_key, context)
                        st.markdown(reply)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.rerun()
