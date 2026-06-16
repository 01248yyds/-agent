import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from openai import OpenAI
import google.generativeai as genai

# ==========================================
# 1. 基础配置与轻量状态管理
# ==========================================
st.set_page_config(page_title="SKU-Doctor 智能数据集体检与决策系统", layout="wide")

# 核心状态持久化
if "analyzed" not in st.session_state: st.session_state.analyzed = False
if "df_cleaned" not in st.session_state: st.session_state.df_cleaned = None
if "df_hierarchical" not in st.session_state: st.session_state.df_hierarchical = None
if "diagnostic_report" not in st.session_state: st.session_state.diagnostic_report = {}
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# 100%继承原 main_青春版V2 的自适应智能映射词条字典
COLUMN_MAP = {
    'product_id': ['Product_ID', 'StockCode', 'product_id', '产品ID', '商品编码', 'sku'],
    'category': ['Category', 'product_category_name', '产品品类', '品类', 'category_name_1'],
    'price': ['Final_Price(Rs.)', 'UnitPrice', 'Price (Rs.)', 'price', '单价', '销售额'],
    'sales': ['order_item_id', 'Quantity', '销量', '数量', 'qty_ordered']
}

# 设置图表中文字体，杜绝中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. 纯本地核心计算引擎 (完美继承原本地 main 代码逻辑)
# ==========================================
def process_dataset_pure_local(uploaded_file):
    """
    完全在本地内存中运行的硬核清洗算法。
    不经过任何外部网络，速度以毫秒计，确保绝对隐私。
    """
    df_raw = pd.read_csv(uploaded_file, low_memory=False)
    df_mapped = pd.DataFrame()
    
    # 1. 智能映射列名
    for logic, candidates in COLUMN_MAP.items():
        match = [c for c in df_raw.columns if c in candidates]
        if match: df_mapped[logic] = df_raw[match[0]]
        
    # 2. 基础清洗与安全保护保护
    if 'category' not in df_mapped.columns: df_mapped['category'] = '未分类未知品类'
    else: df_mapped['category'] = df_mapped['category'].fillna('未分类未知品类')
    
    if 'price' not in df_mapped.columns: df_mapped['price'] = 0.0
    if df_mapped['price'].dtype == 'object':
        df_mapped['price'] = df_mapped['price'].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df_mapped['price'] = pd.to_numeric(df_mapped['price'], errors='coerce')
    df_mapped['price'] = df_mapped['price'].fillna(0.0)
    
    if 'sales' in df_mapped.columns: df_mapped['actual_sales'] = df_mapped['sales']
    else: df_mapped['actual_sales'] = 1

    # 3. 级联聚合：大类与单品
    df_product = df_mapped.groupby(['category', 'product_id']).agg(
        total_revenue=('price', 'sum'), 
        sales_count=('actual_sales', 'count' if 'sales' not in df_mapped.columns else 'sum')
    ).reset_index()
    
    df_category = df_product.groupby('category').agg(
        total_revenue=('total_revenue', 'sum'), 
        sales_count=('sales_count', 'sum')
    ).reset_index().sort_values(by='total_revenue', ascending=False)
    
    # 4. ABC 矩阵经典分级
    df_category['cum_pct'] = df_category['total_revenue'].cumsum() / df_category['total_revenue'].sum()
    df_category['rank'] = df_category['cum_pct'].apply(lambda x: 'A' if x <= 0.8 else ('B' if x <= 0.95 else 'C'))
    
    # 5. 构建“另起一行”的多层级穿透大表 (分析后文件.csv)
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
    
    # 6. 100%原汁原味的动态风险警示线逻辑 (10% 与 90% 判定门槛)
    a_cats = df_category[df_category['rank'] == 'A']
    c_cats = df_category[df_category['rank'] == 'C']
    avg_rev_A = a_cats['total_revenue'].mean() if not a_cats.empty else 1
    avg_rev_C = c_cats['total_revenue'].mean() if not c_cats.empty else 0
    ratio = avg_rev_C / avg_rev_A if avg_rev_A > 0 else 0
    
    if ratio <= 0.10:
        warning = f"🚨 【品类裁剪预警】C类品类平均利润仅为A类的 {ratio*100:.1f}%，大类间两极分化极度严重。系统建议：可考虑果断放弃或整体裁剪整个C类大类。"
    elif ratio >= 0.90:
        warning = f"⚖️ 【微观结构预警】C类品类平均利润达A类的 {ratio*100:.1f}%，各大类表现基本均衡。系统建议：不宜盲目砍掉整个品类，请转而进入下方单品穿透层级，精准淘汰低效SKU。"
    else:
        warning = f"📊 【混合协同预警】C类品类平均利润为A类的 {ratio*100:.1f}%，处于中游状态。系统建议：维持宏观大类现状，实行精准下架、去库存单品的局部微调。"

    report_dict = {
        "warning": warning,
        "top_cat": df_category.head(3), "bottom_cat": df_category.tail(3),
        "top_prod": df_product.sort_values(by='total_revenue', ascending=False).head(3)
    }
    
    return df_category, df_hierarchical, report_dict

# ==========================================
# 3. AI 辅助分析中转站
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
        return f"🔑 AI 专家接入失败，请检查侧边栏 API Key。错误信息：{str(e)}"

# ==========================================
# 4. 极简专业化界面渲染
# ==========================================
# 侧边栏
with st.sidebar:
    st.markdown("### 📁 1. 数据源导入")
    uploaded_file = st.file_uploader("请上传平台导出的原始 CSV 销售流水/财务数据集", type=["csv"])
    
    # 修复二：这里将 St.spinner 改为了正确的全小写 st.spinner
    if uploaded_file and st.button("🚀 开始自动化数据体检", use_container_width=True):
        with st.spinner("本地核心算法正在极速穿透清洗..."):
            df_cat, df_hier, r_dict = process_dataset_pure_local(uploaded_file)
            st.session_state.df_cleaned = df_cat
            st.session_state.df_hierarchical = df_hier
            st.session_state.diagnostic_report = r_dict
            st.session_state.analyzed = True
            st.session_state.chat_history = [] # 重置历史
        st.rerun()

    st.markdown("---")
    st.markdown("### 🤖 2. 答辩与决策 AI 专家外挂 (可选)")
    ai_provider = st.selectbox("选择 AI 引擎", ["DeepSeek (国内直连)", "Google Gemini"])
    api_key = st.text_input(f"配置 {ai_provider} 密钥 (可选)", type="password", placeholder="留空则不启用AI功能")

# 右侧主看板
st.markdown("<h2 style='text-align: center; color: #1E3A8A;'>⚕️ SKU-Doctor 智能数据集体检与决策系统</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>三大基本功能：多维智能清洗 ｜ 级联纵向穿透分析 ｜ 一键标准化成果下载</p>", unsafe_allow_html=True)

if not st.session_state.analyzed:
    st.info("💡 欢迎使用 SKU-Doctor。请在左侧导入您需要体检的原始 CSV 格式数据集，系统将通过纯本地计算立刻给出诊断成果。")
else:
    # 核心看板分左右两栏：左侧硬核数据（原本功能），右侧AI咨询解答（可选外挂）
    col_left, col_right = st.columns([11, 9])
    
    # ------------------ 左侧栏：硬核本地数据成果 ------------------
    with col_left:
        st.markdown("### 📊 本地数智清洗与核心结论")
        st.warning(st.session_state.diagnostic_report["warning"])
        
        # 本地极速绘图渲染
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
            ax2.barh(comp_df['category'], comp_df['total_revenue'], color=['#2ca02c']*3 + ['#d62728']*3)
            ax2.set_title("最优 Top 3 与 最劣 Bottom 3 品类对比")
            ax2.invert_yaxis()
            st.pyplot(fig2)
            plt.close(fig2)
            
        st.markdown("#### 📥 标准化清洗成果下载归档")
        col_d1, col_d2 = st.columns(2)
        col_d1.download_button(
            label="💾 导出格式化：清洗后文件.csv", 
            data=st.session_state.df_cleaned.to_csv(index=False).encode('utf-8-sig'),
            file_name="清洗后文件.csv", mime="text/csv", use_container_width=True
        )
        col_d2.download_button(
            label="💾 导出穿透大表：分析后文件.csv", 
            data=st.session_state.df_hierarchical.to_csv(index=False).encode('utf-8-sig'),
            file_name="分析后文件.csv", mime="text/csv", use_container_width=True
        )
        
        st.markdown("#### 📂 级联纵向穿透大表预览")
        st.dataframe(st.session_state.df_hierarchical, height=380, use_container_width=True)

    # ------------------ 右侧栏：解惑与 AI 答辩模拟 ------------------
    with col_right:
        st.markdown("### 💬 结论不理解？唤醒 AI 智能解惑")
        
        if not api_key:
            st.info("🔒 提示：当前未配置 AI 密钥。系统正处于『100%纯本地离线计算模式』。若对左侧的诊断图表或预警存在疑问，可在左侧侧边栏配置 API Key 激活外挂大脑。")
        else:
            # 渲染历史对话
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]): 
                    st.markdown(msg["content"])
            
            # 用户交互解惑（为输入框指定唯一 Key）
            if chat_input := st.chat_input("针对左侧图表、ABC分析结论，向 AI 咨询或请求模拟答辩...", key="pure_sku_chat"):
                with st.chat_message("user"): 
                    st.markdown(chat_input)
                st.session_state.chat_history.append({"role": "user", "content": chat_input})
                
                with st.chat_message("assistant"):
                    with st.spinner("AI 智囊正在透视数据背后故事..."):
                        context = f"""
                        你是一个极度高级的商业咨询专家与高校答辩评委。
                        当前用户正在看由我们的 Python 本地算法计算出来的 SKU 诊断报告。
                        - 本地系统的硬核诊断警告线结论是：{st.session_state.diagnostic_report['warning']}
                        - 表现最好的前三个大类是：{st.session_state.diagnostic_report['top_cat']['category'].tolist()}
                        - 利润最高的爆款单品ID为：{st.session_state.diagnostic_report['top_prod']['product_id'].tolist()}
                        
                        用户目前对这个本地计算出来的结论有些不理解、不清楚，或者正在准备应对教授的质疑。
                        请基于精益创业商业逻辑和严谨的财务常识，为用户解答疑惑，或者提供极具说服力的答辩话术：
                        用户提问：{chat_input}
                        """
                        reply = call_ai_consultant(ai_provider, api_key, context)
                        st.markdown(reply)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.rerun()
