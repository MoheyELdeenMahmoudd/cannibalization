import streamlit as st
import pandas as pd
import numpy as np
import io
import datetime
import re
import urllib.parse
from urllib.parse import urlparse
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# ==========================================
# 🎨 1. UI CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="Almaster Tech | SEO Command Center", page_icon="🦁", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif !important; }
    
    .stApp { background-color: #0e1117; color: #ffffff; }
    
    /* Custom Headers */
    .header-box {
        background: linear-gradient(90deg, #1e293b, #0f172a);
        padding: 20px; border-radius: 12px; border-left: 5px solid #38bdf8;
        margin-bottom: 25px; text-align: center;
    }
    
    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: #1f2937; border: 1px solid #374151;
        border-radius: 10px; padding: 15px; direction: rtl;
    }
    
    /* Dataframes */
    .stDataFrame { border: 1px solid #374151; border-radius: 5px; }
    
    /* RTL Adjustments */
    .rtl { direction: rtl; text-align: right; }
    .stSelectbox, .stTextInput, .stSlider { direction: rtl; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-box">
    <h1 style="color:white; margin:0;">ALMASTER <span style="color:#38bdf8;">TECH</span></h1>
    <p style="color:#94a3b8; font-size:16px;">Advanced Cannibalization Logic Engine v2.0</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# ⚙️ 2. ADVANCED CONFIGURATION
# ==========================================
class Config:
    # Logic Thresholds
    DOMINANCE_TOP_POS = 3.5  # If winner is better than this...
    DOMINANCE_SECOND_POS = 6.0 # And loser is better than this... It's Dominance.
    
    MIN_IMPRESSIONS = 20     # Ignore noise
    MIN_CLICKS = 0           # Can detect issues even with 0 clicks if impressions are high
    
    # Intent Dictionaries (Expanded)
    COMM_TERMS = ['buy', 'price', 'cost', 'service', 'hire', 'agency', 'shop', 'store', 
                  'شراء', 'سعر', 'اسعار', 'تكلفة', 'خدمة', 'شركة', 'وكالة', 'متجر', 'طلب', 'حجز', 'عيادة', 'دكتور']
    
    INFO_TERMS = ['how', 'what', 'guide', 'tips', 'best', 'review', 'vs', 'difference', 'signs', 'symptoms',
                  'كيف', 'ما هو', 'دليل', 'شرح', 'نصائح', 'افضل', 'مقارنة', 'الفرق', 'اعراض', 'علاج', 'اسباب', 'طريقة']

# ==========================================
# 🧠 3. CORE LOGIC ENGINE (THE BRAIN)
# ==========================================

def get_page_intent(url, query):
    """Detects intent based on URL structure and Query terms."""
    url_lower = str(url).lower()
    query_lower = str(query).lower()
    
    score_comm = 0
    score_info = 0
    
    # Check Query
    if any(t in query_lower for t in Config.COMM_TERMS): score_comm += 2
    if any(t in query_lower for t in Config.INFO_TERMS): score_info += 2
    
    # Check URL Patterns
    if any(x in url_lower for x in ['/product', '/cart', '/checkout', '/services']): score_comm += 3
    if any(x in url_lower for x in ['/blog', '/article', '/news', '/wiki', '/guide']): score_info += 3
    
    if score_comm > score_info: return "Commercial"
    if score_info > score_comm: return "Informational"
    return "Ambiguous"

def classify_cannibalization(row, brands):
    """
    The Senior SEO Logic:
    Distinguishes between 'Bad Cannibalization' and 'Good Dominance'.
    """
    winner_pos = row['Winner_Pos']
    loser_pos = row['Loser_Pos']
    winner_intent = row['Winner_Intent']
    loser_intent = row['Loser_Intent']
    query = row['Query']
    
    # 1. Brand Check
    is_brand = any(b.lower() in query.lower() for b in brands)
    if is_brand:
        return "Brand Dominance (Safe)", "🟢", "Monitor"

    # 2. Dominance Check (e.g. Ranking #1 and #2)
    if winner_pos <= Config.DOMINANCE_TOP_POS and loser_pos <= Config.DOMINANCE_SECOND_POS:
        return "SERP Dominance (Good)", "🟢", "Monitor - Do Not Touch"

    # 3. Intent Mismatch Check
    if winner_intent != "Ambiguous" and loser_intent != "Ambiguous" and winner_intent != loser_intent:
        return "Intent Conflict", "🟠", "Content Split / De-optimize Loser"

    # 4. True Cannibalization (The bad stuff)
    # High Severity: Loser is stealing significant traffic or dragging winner down
    if row['Overlap_Score'] > 0.6: 
        return "Critical Cannibalization", "🔴", "Merge / 301 Redirect"
    
    return "Moderate Cannibalization", "🟡", "Review Content Diff"

def analyze_gsc_data(df_raw, brands):
    # 1. Cleaning
    df = df_raw.copy()
    df['page_clean'] = df['page'].apply(lambda x: str(x).split('?')[0].split('#')[0].rstrip('/'))
    
    # 2. Aggregation (Merge split URLs)
    df_agg = df.groupby(['query', 'page_clean']).agg({
        'clicks': 'sum',
        'impressions': 'sum',
        'ctr': 'mean',
        'position': 'mean'
    }).reset_index()
    
    # 3. Filter Noise
    df_agg = df_agg[df_agg['impressions'] >= Config.MIN_IMPRESSIONS]
    
    # 4. Identify Queries with Multiple Pages
    query_counts = df_agg['query'].value_counts()
    cannibal_queries = query_counts[query_counts > 1].index.tolist()
    
    if not cannibal_queries:
        return pd.DataFrame()
    
    df_cannibal = df_agg[df_agg['query'].isin(cannibal_queries)]
    
    results = []
    
    for query, group in df_cannibal.groupby('query'):
        # Sort by Clicks (primary) then Impressions
        group = group.sort_values(['clicks', 'impressions'], ascending=[False, False])
        
        winner = group.iloc[0]
        losers = group.iloc[1:]
        
        # Calculate Winner Intent
        w_intent = get_page_intent(winner['page_clean'], query)
        
        for _, loser in losers.iterrows():
            l_intent = get_page_intent(loser['page_clean'], query)
            
            # Overlap Score (How much is the loser eating?)
            overlap = (loser['impressions'] / winner['impressions']) if winner['impressions'] > 0 else 0
            
            # Traffic Loss Estimation (Simplified CTR Curve model)
            # Assuming if Loser didn't exist, Winner would get 80% of Loser's Impressions converted at Winner's CTR
            traffic_loss = int(loser['impressions'] * winner['ctr'])
            
            row_data = {
                'Query': query,
                'Winner_Page': winner['page_clean'],
                'Winner_Pos': round(winner['position'], 1),
                'Winner_Intent': w_intent,
                'Loser_Page': loser['page_clean'],
                'Loser_Pos': round(loser['position'], 1),
                'Loser_Intent': l_intent,
                'Loser_Imps': loser['impressions'],
                'Overlap_Score': overlap,
                'Traffic_Loss': traffic_loss
            }
            
            # Apply Logic
            status, icon, action = classify_cannibalization(row_data, brands)
            
            row_data.update({
                'Status': status,
                'Icon': icon,
                'Action': action,
                'Priority': traffic_loss  # Sort key
            })
            
            results.append(row_data)
            
    return pd.DataFrame(results).sort_values('Priority', ascending=False)

# ==========================================
# 🔌 4. GSC CONNECTIVITY
# ==========================================
@st.cache_resource
def authenticate_gsc(auth_code):
    try:
        flow = InstalledAppFlow.from_client_secrets_file(
            "client_secret.json", 
            ['https://www.googleapis.com/auth/webmasters.readonly']
        )
        flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
        flow.fetch_token(code=auth_code)
        return build('searchconsole', 'v1', credentials=flow.credentials)
    except Exception as e:
        return None

def fetch_data(service, site_url, days):
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days)
    
    request = {
        'startDate': start_date.isoformat(),
        'endDate': end_date.isoformat(),
        'dimensions': ['query', 'page'],
        'rowLimit': 25000 
    }
    
    response = service.searchanalytics().query(siteUrl=site_url, body=request).execute()
    rows = response.get('rows', [])
    
    if not rows: return pd.DataFrame()
    
    data = []
    for row in rows:
        data.append({
            'query': row['keys'][0],
            'page': row['keys'][1],
            'clicks': row['clicks'],
            'impressions': row['impressions'],
            'ctr': row['ctr'],
            'position': row['position']
        })
    return pd.DataFrame(data)

# ==========================================
# 🖥️ 5. MAIN DASHBOARD
# ==========================================
with st.sidebar:
    st.header("⚙️ الإعدادات")
    
    # Auth
    uploaded_file = st.file_uploader("ملف JSON (client_secret)", type="json")
    
    if 'creds' in st.session_state:
        st.success("✅ متصل بنجاح")
        sites = st.session_state.get('sites', [])
        if not sites:
            try:
                site_list = st.session_state.creds.sites().list().execute()
                sites = [s['siteUrl'] for s in site_list.get('siteEntry', [])]
                st.session_state.sites = sites
            except: pass
        
        selected_site = st.selectbox("اختر الموقع", sites)
    else:
        selected_site = st.text_input("رابط الموقع (يدوي)", "https://example.com")

    days = st.slider("فترة التحليل (أيام)", 7, 90, 28)
    
    st.markdown("---")
    st.subheader("🛡️ إعدادات البراند")
    brands_input = st.text_area("كلمات البراند (افصل بفاصلة)", "almaster, المستر, ماستر")
    brands = [b.strip() for b in brands_input.split(',')]

# Login Flow
if uploaded_file and 'creds' not in st.session_state:
    with open("client_secret.json", "wb") as f: f.write(uploaded_file.getbuffer())
    flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", ['https://www.googleapis.com/auth/webmasters.readonly'])
    flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
    auth_url, _ = flow.authorization_url()
    st.markdown(f"[🔗 اضغط هنا للحصول على الكود]({auth_url})")
    code = st.text_input("أدخل كود المصادقة:")
    if code:
        srv = authenticate_gsc(code)
        if srv:
            st.session_state.creds = srv
            st.rerun()

# Analysis Trigger
if st.button("🚀 تشغيل التحليل (Deep Scan)", type="primary"):
    if 'creds' in st.session_state:
        with st.spinner("جاري سحب البيانات وتحليل الـ SERP Logic..."):
            raw_df = fetch_data(st.session_state.creds, selected_site, days)
            if not raw_df.empty:
                report_df = analyze_gsc_data(raw_df, brands)
                st.session_state.report = report_df
            else:
                st.error("لا توجد بيانات كافية للفترة المحددة.")
    else:
        st.warning("يرجى تسجيل الدخول أولاً.")

# Reporting View
if 'report' in st.session_state and not st.session_state.report.empty:
    df = st.session_state.report
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    critical = df[df['Status'].str.contains('Critical')]
    dominance = df[df['Status'].str.contains('Dominance')]
    
    col1.metric("🔴 تضارب حرج (يجب الإصلاح)", len(critical))
    col2.metric("🟢 هيمنة (ممتاز)", len(dominance))
    col3.metric("🟠 تضارب نوايا", len(df[df['Status'].str.contains('Intent')]))
    col4.metric("📉 زيارات محتملة ضائعة", f"{critical['Traffic_Loss'].sum():,}")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2 = st.tabs(["📋 تقرير العمليات (Actionable)", "📊 البيانات الكاملة"])
    
    with tab1:
        st.subheader("أولويات الإصلاح")
        st.info("💡 هذا الجدول يظهر فقط المشاكل التي تتطلب تدخلاً (تم استبعاد الهيمنة الإيجابية).")
        
        action_df = df[~df['Status'].str.contains('Dominance')].copy()
        
        st.dataframe(
            action_df[['Icon', 'Query', 'Status', 'Action', 'Winner_Page', 'Loser_Page', 'Traffic_Loss']],
            column_config={
                "Winner_Page": st.column_config.LinkColumn("الصفحة الفائزة"),
                "Loser_Page": st.column_config.LinkColumn("الصفحة الخاسرة"),
                "Traffic_Loss": st.column_config.ProgressColumn("حجم الخسارة", format="%d", min_value=0, max_value=int(action_df['Traffic_Loss'].max()))
            },
            use_container_width=True,
            height=500
        )
        
    with tab2:
        st.dataframe(df, use_container_width=True)

    # Excel Export (The Professional Way)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Full Report', index=False)
        critical.to_excel(writer, sheet_name='CRITICAL ACTIONS', index=False)
        dominance.to_excel(writer, sheet_name='Dominance Wins', index=False)
    
    st.download_button(
        label="📥 تحميل تقرير Excel احترافي",
        data=output.getvalue(),
        file_name=f"Almaster_SEO_Audit_{datetime.date.today()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

elif 'report' in st.session_state:
    st.success("🎉 نظيف تماماً! لا يوجد أي تضارب (Cannibalization) في موقعك.")
