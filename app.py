import streamlit as st
import pandas as pd
import numpy as np
import io
import datetime
import re
import requests
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
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
    
    /* Status Badges */
    .badge { padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }
    .badge-red { background-color: #7f1d1d; color: #fca5a5; }
    .badge-green { background-color: #14532d; color: #86efac; }
    
    /* RTL Adjustments */
    .rtl { direction: rtl; text-align: right; }
    .stSelectbox, .stTextInput, .stSlider { direction: rtl; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-box">
    <h1 style="color:white; margin:0;">ALMASTER <span style="color:#38bdf8;">TECH</span></h1>
    <p style="color:#94a3b8; font-size:16px;">SEO Cannibalization Engine v3.0 (with Live Validation)</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# ⚙️ 2. ADVANCED CONFIGURATION
# ==========================================
class Config:
    # Logic Thresholds
    DOMINANCE_TOP_POS = 3.5  
    DOMINANCE_SECOND_POS = 6.0 
    
    MIN_IMPRESSIONS = 15     # Ignore insignificant noise
    
    # Intent Dictionaries
    COMM_TERMS = ['buy', 'price', 'cost', 'service', 'hire', 'agency', 'shop', 'store', 'book',
                  'شراء', 'سعر', 'اسعار', 'تكلفة', 'خدمة', 'شركة', 'وكالة', 'متجر', 'طلب', 'حجز', 'عيادة', 'دكتور']
    
    INFO_TERMS = ['how', 'what', 'guide', 'tips', 'best', 'review', 'vs', 'signs', 'symptoms', 'causes',
                  'كيف', 'ما هو', 'دليل', 'شرح', 'نصائح', 'افضل', 'مقارنة', 'الفرق', 'اعراض', 'علاج', 'اسباب', 'طريقة']

# ==========================================
# 🕵️ 3. LIVE VALIDATOR ENGINE
# ==========================================
def get_live_status(url):
    """
    Sends a HEAD request to check if page is live, redirected (301), or gone (404).
    """
    try:
        # User-Agent to avoid blocking
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AlmasterTechBot/1.0'}
        response = requests.head(url, allow_redirects=True, timeout=3, headers=headers)
        
        was_redirected = False
        if response.history:
            was_redirected = True
            
        return response.status_code, was_redirected, response.url
    except:
        return 0, False, url

# ==========================================
# 🧠 4. CORE LOGIC ENGINE
# ==========================================

def get_page_intent(url, query):
    url_lower = str(url).lower()
    query_lower = str(query).lower()
    
    score_comm = 0
    score_info = 0
    
    if any(t in query_lower for t in Config.COMM_TERMS): score_comm += 2
    if any(t in query_lower for t in Config.INFO_TERMS): score_info += 2
    
    if any(x in url_lower for x in ['/product', '/cart', '/checkout', '/services']): score_comm += 3
    if any(x in url_lower for x in ['/blog', '/article', '/news', '/wiki', '/guide']): score_info += 3
    
    if score_comm > score_info: return "Commercial"
    if score_info > score_comm: return "Informational"
    return "Ambiguous"

def classify_cannibalization(row, brands):
    winner_pos = row['Winner_Pos']
    loser_pos = row['Loser_Pos']
    winner_intent = row['Winner_Intent']
    loser_intent = row['Loser_Intent']
    query = row['Query']
    
    # 1. Brand Check
    is_brand = any(b.lower() in query.lower() for b in brands)
    if is_brand:
        return "Brand Dominance (Safe)", "🟢", "Monitor"

    # 2. Dominance Check
    if winner_pos <= Config.DOMINANCE_TOP_POS and loser_pos <= Config.DOMINANCE_SECOND_POS:
        return "SERP Dominance (Good)", "🟢", "Monitor - Do Not Touch"

    # 3. Intent Mismatch
    if winner_intent != "Ambiguous" and loser_intent != "Ambiguous" and winner_intent != loser_intent:
        return "Intent Conflict", "🟠", "Content Split / De-optimize Loser"

    # 4. True Cannibalization
    if row['Overlap_Score'] > 0.6: 
        return "Critical Cannibalization", "🔴", "Merge / 301 Redirect"
    
    return "Moderate Cannibalization", "🟡", "Review Content Diff"

def analyze_gsc_data(df_raw, brands):
    # 1. Clean & Aggregate
    df = df_raw.copy()
    df['page_clean'] = df['page'].apply(lambda x: str(x).split('?')[0].split('#')[0].rstrip('/'))
    
    df_agg = df.groupby(['query', 'page_clean']).agg({
        'clicks': 'sum',
        'impressions': 'sum',
        'ctr': 'mean',
        'position': 'mean'
    }).reset_index()
    
    # Filter Noise
    df_agg = df_agg[df_agg['impressions'] >= Config.MIN_IMPRESSIONS]
    
    # Identify Cannibal Queries
    query_counts = df_agg['query'].value_counts()
    cannibal_queries = query_counts[query_counts > 1].index.tolist()
    
    if not cannibal_queries:
        return pd.DataFrame()
    
    df_cannibal = df_agg[df_agg['query'].isin(cannibal_queries)]
    results = []
    pages_to_check = []
    
    # 2. Initial Analysis Loop
    for query, group in df_cannibal.groupby('query'):
        group = group.sort_values(['clicks', 'impressions'], ascending=[False, False])
        winner = group.iloc[0]
        losers = group.iloc[1:]
        
        w_intent = get_page_intent(winner['page_clean'], query)
        
        for _, loser in losers.iterrows():
            l_intent = get_page_intent(loser['page_clean'], query)
            overlap = (loser['impressions'] / winner['impressions']) if winner['impressions'] > 0 else 0
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
            
            status, icon, action = classify_cannibalization(row_data, brands)
            
            # Flag for live check if Critical
            check_live = False
            if "Critical" in status or "Intent" in status or "Moderate" in status:
                check_live = True
                pages_to_check.append(loser['page_clean'])
            
            row_data.update({
                'Status': status, 'Icon': icon, 'Action': action, 
                'Priority': traffic_loss, 'Check_Live': check_live
            })
            results.append(row_data)

    # 3. Parallel Live Validation (Threaded)
    unique_pages = list(set(pages_to_check))
    status_map = {}
    
    if unique_pages:
        status_text = st.empty()
        status_text.info(f"🕵️ جاري التحقق الحي من {len(unique_pages)} صفحة لاستبعاد الـ Redirects...")
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {executor.submit(get_live_status, url): url for url in unique_pages}
            for future in future_to_url:
                try:
                    url = future_to_url[future]
                    code, redirected, final_url = future.result()
                    status_map[url] = {'code': code, 'redirected': redirected}
                except:
                    status_map[url] = {'code': 0, 'redirected': False}
        
        status_text.empty()

    # 4. Final Data Refinement
    final_results = []
    for row in results:
        url = row['Loser_Page']
        
        if row['Check_Live'] and url in status_map:
            live_data = status_map[url]
            code = str(live_data['code'])
            
            # Logic: If 301/308 or Redirected -> Resolved
            if live_data['redirected'] or code.startswith('3'):
                row['Status'] = "Resolved (Redirected)"
                row['Icon'] = "✅"
                row['Action'] = "None (Fixed)"
                row['Priority'] = -1 
                
            # Logic: If 404/410 -> Resolved
            elif code in ['404', '410']:
                 row['Status'] = "Resolved (Page Gone)"
                 row['Icon'] = "🗑️"
                 row['Action'] = "None"
                 row['Priority'] = -1
        
        final_results.append(row)
            
    return pd.DataFrame(final_results).sort_values('Priority', ascending=False)

# ==========================================
# 🔌 5. GSC CONNECTIVITY
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
    try:
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
    except Exception as e:
        st.error(f"GSC Error: {str(e)}")
        return pd.DataFrame()

# ==========================================
# 🖥️ 6. MAIN DASHBOARD UI
# ==========================================
with st.sidebar:
    st.header("⚙️ الإعدادات")
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
        selected_site = st.text_input("رابط الموقع (للمعاينة)", "https://example.com")

    days = st.slider("فترة التحليل (أيام)", 7, 90, 28)
    
    st.markdown("---")
    st.subheader("🛡️ إعدادات البراند")
    brands_input = st.text_area("كلمات البراند", "almaster, المستر, ماستر")
    brands = [b.strip() for b in brands_input.split(',')]

# Auth Logic
if uploaded_file and 'creds' not in st.session_state:
    with open("client_secret.json", "wb") as f: f.write(uploaded_file.getbuffer())
    flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", ['https://www.googleapis.com/auth/webmasters.readonly'])
    flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
    auth_url, _ = flow.authorization_url()
    st.markdown(f"[🔗 اضغط هنا للحصول على كود المصادقة]({auth_url})")
    code = st.text_input("أدخل الكود هنا:")
    if code:
        srv = authenticate_gsc(code)
        if srv:
            st.session_state.creds = srv
            st.rerun()

# Execute Analysis
if st.button("🚀 تشغيل التحليل الشامل (Deep Scan)", type="primary"):
    if 'creds' in st.session_state:
        with st.spinner("جاري سحب البيانات، تحليل الـ SERP، وفحص الـ Redirects الحية..."):
            raw_df = fetch_data(st.session_state.creds, selected_site, days)
            if not raw_df.empty:
                report_df = analyze_gsc_data(raw_df, brands)
                st.session_state.report = report_df
            else:
                st.error("لا توجد بيانات كافية للفترة المحددة.")
    else:
        st.warning("يرجى تسجيل الدخول أولاً.")

# Display Results
if 'report' in st.session_state and not st.session_state.report.empty:
    df = st.session_state.report
    
    # Filter Actions (Remove Resolved and Dominance from Metrics)
    actionable = df[~df['Status'].str.contains('Resolved|Dominance')]
    critical = actionable[actionable['Status'].str.contains('Critical')]
    resolved = df[df['Status'].str.contains('Resolved')]
    dominance = df[df['Status'].str.contains('Dominance')]
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔴 تضارب حقيقي (Action)", len(critical))
    col2.metric("🟢 هيمنة (Good)", len(dominance))
    col3.metric("🧹 تم حله (Redirects)", len(resolved))
    col4.metric("📉 زيارات مهددة", f"{critical['Traffic_Loss'].sum():,}")
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["🚀 خطة العمل", "🛡️ الهيمنة (السيطرة)", "📊 البيانات الكاملة"])
    
    with tab1:
        st.subheader("🔥 مشاكل تتطلب تدخلاً فورياً")
        st.info("تم استبعاد الحالات التي تم حلها (301) أو حالات السيطرة.")
        
        st.dataframe(
            actionable[['Icon', 'Query', 'Status', 'Action', 'Winner_Page', 'Loser_Page', 'Traffic_Loss']],
            column_config={
                "Winner_Page": st.column_config.LinkColumn("الصفحة الفائزة"),
                "Loser_Page": st.column_config.LinkColumn("الصفحة الخاسرة (المشكلة)"),
                "Traffic_Loss": st.column_config.ProgressColumn("الخسارة المتوقعة", format="%d", min_value=0, max_value=int(df['Traffic_Loss'].max()))
            },
            use_container_width=True
        )
        
    with tab2:
        st.subheader("✅ كلمات تسيطر عليها بأكثر من نتيجة")
        st.success("هذه النتائج جيدة! لا تلمسها.")
        st.dataframe(dominance[['Query', 'Winner_Pos', 'Loser_Pos', 'Winner_Page', 'Loser_Page']], use_container_width=True)
        
    with tab3:
        st.dataframe(df, use_container_width=True)

    # Excel Export
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Full Report', index=False)
        actionable.to_excel(writer, sheet_name='Action Plan', index=False)
        resolved.to_excel(writer, sheet_name='Resolved Issues', index=False)
        dominance.to_excel(writer, sheet_name='Dominance Wins', index=False)
    
    st.download_button(
        label="📥 تحميل التقرير الشامل (Excel)",
        data=output.getvalue(),
        file_name=f"Almaster_SEO_Audit_{datetime.date.today()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

elif 'report' in st.session_state:
    st.balloons()
    st.success("موقعك نظيف تماماً! لا يوجد أي تضارب حالياً.")
