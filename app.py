import streamlit as st
import pandas as pd
import numpy as np
import math
import re
import os
import datetime
from urllib.parse import urlparse
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

# ==========================================
# 🎨 1. THEME & UI SETUP (نفس تصميم ملف HTML)
# ==========================================
st.set_page_config(page_title="Almaster Tech - Cannibalization Hunter", page_icon="🕵️", layout="wide")

# حقن نفس الـ CSS والألوان والخطوط من ملفك
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');
    
    /* Global Reset */
    * { font-family: 'Cairo', sans-serif !important; }
    
    /* Background similar to your file */
    .stApp {
        background: url('https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop') no-repeat center center fixed;
        background-size: cover;
    }
    
    /* Dark Overlay */
    .stApp::before {
        content: ""; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(1, 1, 114, 0.75); z-index: -1; pointer-events: none;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.96);
        border-radius: 16px;
        padding: 30px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        border: 1px solid rgba(255,255,255,0.5);
        margin-bottom: 20px;
    }

    /* Headers */
    h1, h2, h3 { color: #010172 !important; font-weight: 800 !important; }
    
    /* Custom Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #2F45FF, #010172);
        color: white; border: none; border-radius: 12px;
        height: 50px; font-weight: 700; width: 100%;
        box-shadow: 0 10px 20px -5px rgba(1, 1, 114, 0.3);
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 30px -5px rgba(1, 1, 114, 0.4);
        color: white;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    div[data-testid="stMetricLabel"] { color: #666; font-size: 14px; }
    div[data-testid="stMetricValue"] { color: #2F45FF; font-weight: 800; font-size: 24px; }

    /* Alert Boxes */
    .stAlert { border-radius: 10px; }
    
    /* Logo Area */
    .logo-text { font-size: 28px; font-weight: 800; color: white; text-align: center; margin-bottom: 20px; }
    .logo-text span { color: #4dabf7; }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="logo-text">ALMASTER <span>TECH</span> <br><span style='font-size:16px; opacity:0.8'>GSC Cannibalization Hunter v8.0</span></div>
""", unsafe_allow_html=True)

# ==========================================
# ⚙️ 2. LOGIC CONFIGURATION (V8 Enterprise)
# ==========================================
class Config:
    MIN_IMP_QUANTILE = 0.2
    MIN_IMP_ABSOLUTE = 10
    MAX_DEPTH_PENALTY = 6
    WEIGHT_CLICKS = 0.4
    WEIGHT_POS = 0.3
    WEIGHT_IMPS = 0.2
    WEIGHT_DEPTH = 0.1
    SEVERITY_CRITICAL = 0.8
    SEVERITY_HIGH = 0.5
    SEVERITY_MEDIUM = 0.3
    EXPECTED_CTR = {1: 30.0, 2: 15.0, 3: 10.0, 4: 7.0, 5: 5.0, 6: 3.5, 7: 3.0, 8: 2.5, 9: 2.0, 10: 1.5}
    URL_PATTERNS = {
        'Commercial': {'terms': ['/product', '/service', '/shop', 'cart', 'checkout', '/pricing', 'booking'], 'weight': 3},
        'Informational': {'terms': ['/blog', '/news', '/article', '/guide', '/wiki', 'learn'], 'weight': 2}
    }
    COMM_QUERY_TERMS = ['buy', 'price', 'cost', 'service', 'company', 'agency', 'hire', 'شراء', 'سعر', 'تكلفة', 'شركة', 'خدمة']
    INFO_QUERY_TERMS = ['how', 'what', 'guide', 'tutorial', 'tips', 'why', 'review', 'vs', 'best', 'top', 'كيف', 'دليل', 'شرح', 'نصائح']

# ==========================================
# 🧠 3. HELPER FUNCTIONS
# ==========================================
def get_expected_ctr(position):
    pos = int(round(max(position, 1)))
    return Config.EXPECTED_CTR.get(pos, 1.0)

def identify_market_segment(url):
    try:
        parsed = urlparse(str(url))
        path = parsed.path.strip('/')
        parts = path.split('/')
        first_segment = parts[0].lower() if parts else ""
        if re.match(r'^[a-z]{2}-[a-z]{2}$', first_segment): return first_segment.upper()
        if re.match(r'^[a-z]{2}$', first_segment): return f"Global-{first_segment.upper()}"
        return "Global-EN"
    except: return "Unknown"

def classify_query_type(query, brand_list):
    q = str(query).lower()
    for brand in brand_list:
        if brand.lower() in q: return "Brand"
    if len(q.split()) >= 4: return "Long-Tail"
    return "Generic"

def detect_page_intent_weighted(url):
    url_lower = str(url).lower()
    scores = {'Commercial': 0, 'Informational': 0, 'General': 0.1}
    for intent, data in Config.URL_PATTERNS.items():
        for term in data['terms']:
            if term in url_lower: scores[intent] += data['weight']
    return max(scores, key=scores.get)

def detect_intents_batch(df):
    def _process_row(row):
        q = str(row['query']).lower()
        c_score = sum(1 for t in Config.COMM_QUERY_TERMS if t in q)
        i_score = sum(1 for t in Config.INFO_QUERY_TERMS if t in q)
        q_intent = 'Commercial' if c_score > i_score else 'Informational' if i_score > c_score else 'Navigational/General'
        p_intent = detect_page_intent_weighted(row['page_clean'])
        return pd.Series([q_intent, p_intent])
    return df.apply(_process_row, axis=1)

def calculate_vectorized_score(df_group):
    max_imp = max(df_group['impressions'].max(), 1)
    max_click = max(df_group['clicks'].max(), 1)
    norm_imp = df_group['impressions'] / max_imp
    norm_click = df_group['clicks'] / max_click
    pos_score = 10 / (np.log(df_group['position'] + 1) + 1)
    depths = df_group['page_clean'].str.count('/').clip(upper=Config.MAX_DEPTH_PENALTY)
    depth_factor = 1 / (depths + 1)
    return (norm_click * Config.WEIGHT_CLICKS) + (pos_score * Config.WEIGHT_POS) + \
           (norm_imp * Config.WEIGHT_IMPS) + (depth_factor * Config.WEIGHT_DEPTH)

def generate_action_plan(severity, w_type, l_type, q_type, ctr_perf):
    if q_type == 'Brand': return "🛡️ BRAND PROTECTION"
    if severity == "🔥 Critical" and w_type == l_type: return "⛔ MERGE / 301"
    if w_type != l_type: return "✂️ SPLIT INTENT"
    if ctr_perf < 0.6: return "🎨 FIX SNIPPET (Low CTR)"
    return "✅ MONITOR"

# ==========================================
# 🔌 4. AUTHENTICATION & DATA FETCHING
# ==========================================
def authenticate_gsc(uploaded_client_secret):
    SCOPES = ['https://www.googleapis.com/auth/webmasters.readonly']
    if uploaded_client_secret:
        # For Streamlit Cloud: Create a temp file
        with open("client_secret.json", "wb") as f:
            f.write(uploaded_client_secret.getbuffer())
        
        flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", SCOPES)
        # Use a fixed Redirect URI for Cloud usually, but for simple use:
        flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob" 
        
        auth_url, _ = flow.authorization_url(prompt='consent')
        
        st.markdown(f"""
        <div class="glass-card" style="text-align:center">
            <h3>🔐 مطلوب المصادقة</h3>
            <p>1. اضغط على الرابط أدناه لتسجيل الدخول بحساب جوجل.</p>
            <a href="{auth_url}" target="_blank" class="stButton" style="text-decoration:none; color:white; background:#010172; padding:10px 20px; border-radius:10px;">🔗 فتح رابط المصادقة</a>
            <p style="margin-top:10px">2. انسخ الكود الذي سيظهر لك وضعه في الخانة بالأسفل.</p>
        </div>
        """, unsafe_allow_html=True)
        
        auth_code = st.text_input("أدخل كود المصادقة هنا:")
        
        if auth_code:
            try:
                flow.fetch_token(code=auth_code)
                return build('searchconsole', 'v1', credentials=flow.credentials)
            except Exception as e:
                st.error(f"خطأ في المصادقة: {e}")
                return None
    return None

@st.cache_data(show_spinner=False)
def run_analysis(df, brand_names_input):
    df.columns = [c.lower().strip() for c in df.columns]
    df['page_clean'] = df['page'].astype(str).str.split('?').str[0].str.split('#').str[0].str.rstrip('/')
    df['market'] = df['page_clean'].apply(identify_market_segment)
    df['query_type'] = df['query'].apply(lambda x: classify_query_type(x, brand_names_input))
    df[['q_intent', 'p_intent']] = detect_intents_batch(df)
    df['weighted_pos'] = df['position'] * df['impressions']
    
    group_cols = ['query', 'market', 'query_type', 'q_intent', 'page_clean', 'p_intent']
    df_grouped = df.groupby(group_cols).agg({'clicks': 'sum', 'impressions': 'sum', 'weighted_pos': 'sum'}).reset_index()
    df_grouped['position'] = df_grouped['weighted_pos'] / df_grouped['impressions']
    df_grouped['ctr'] = (df_grouped['clicks'] / df_grouped['impressions'].replace(0, 1)) * 100
    
    quantile_thresh = df_grouped['impressions'].quantile(Config.MIN_IMP_QUANTILE)
    min_imp = max(int(quantile_thresh), Config.MIN_IMP_ABSOLUTE)
    df_filtered = df_grouped[df_grouped['impressions'] >= min_imp].copy()
    
    report_data = []
    for (query, market), group in df_filtered.groupby(['query', 'market']):
        if len(group) < 2: continue
        group = group.copy()
        group['score'] = calculate_vectorized_score(group)
        group = group.sort_values(by='score', ascending=False)
        winner = group.iloc[0]
        losers = group.iloc[1:]
        top_loser = losers.iloc[0]
        overlap = top_loser['score'] / winner['score'] if winner['score'] > 0 else 0
        base_severity = overlap * (1.2 if winner['query_type'] == 'Brand' else 1.0)
        
        severity = "Low"
        if base_severity > Config.SEVERITY_CRITICAL: severity = "🔥 Critical"
        elif base_severity > Config.SEVERITY_HIGH: severity = "High"
        elif base_severity > Config.SEVERITY_MEDIUM: severity = "Medium"
        
        exp_ctr = get_expected_ctr(winner['position'])
        ctr_perf = winner['ctr'] / exp_ctr if exp_ctr > 0 else 0
        action = generate_action_plan(severity, winner['p_intent'], top_loser['p_intent'], winner['query_type'], ctr_perf)
        wasted_imps = losers['impressions'].sum()
        est_traffic_loss = round(wasted_imps * (winner['ctr'] / 100))
        
        traffic_factor = min(math.log(wasted_imps + 1) * 10, 50)
        sev_points = { "🔥 Critical": 40, "High": 25, "Medium": 10, "Low": 5 }.get(severity, 5)
        brand_points = 10 if winner['query_type'] == 'Brand' else 0
        priority_score = min(traffic_factor + sev_points + brand_points, 100)
        
        difficulty = "Medium"
        if "MERGE" in action or "SNIPPET" in action: difficulty = "Low"
        elif "SPLIT" in action: difficulty = "High"

        report_data.append({
            'Market': market, 'Query': query, 'Severity': severity, 'Priority_Score': int(priority_score),
            'Action_Plan': action, 'Difficulty': difficulty, 'Est_Traffic_Loss': est_traffic_loss,
            'Winner': winner['page_clean'], 'Winner_CTR': round(winner['ctr'], 2),
            'Loser': top_loser['page_clean'], 'Overlap': round(overlap * 100, 1)
        })
    return pd.DataFrame(report_data)

# ==========================================
# 🖥️ 5. APP LAYOUT
# ==========================================
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### ⚙️ الإعدادات", unsafe_allow_html=True)
    
    uploaded_secret = st.file_uploader("📂 ملف المصادقة (client_secret.json)", type="json")
    
    site_url = st.text_input("رابط الموقع (GSC Property)", value="sc-domain:almaster.tech")
    days = st.slider("فترة التحليل (أيام)", 7, 90, 30)
    brands = st.text_area("أسماء البراند (للحماية)", "almaster, المستر, ماستر")
    
    run_btn = st.button("🚀 بدء الفحص الشامل")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if run_btn and uploaded_secret:
        service = authenticate_gsc(uploaded_secret)
        
        if service:
            with st.spinner('⏳ جاري سحب البيانات من جوجل...'):
                end_date = datetime.date.today()
                start_date = end_date - datetime.timedelta(days=days)
                req = {'startDate': start_date.isoformat(), 'endDate': end_date.isoformat(), 
                       'dimensions': ['query', 'page'], 'rowLimit': 25000}
                try:
                    resp = service.searchanalytics().query(siteUrl=site_url, body=req).execute()
                    rows = resp.get('rows', [])
                except Exception as e:
                    st.error(f"Error: {e}")
                    rows = []

            if rows:
                df_raw = pd.DataFrame([{
                    'query': r['keys'][0], 'page': r['keys'][1],
                    'clicks': r['clicks'], 'impressions': r['impressions'],
                    'ctr': r['ctr'], 'position': r['position']
                } for r in rows])
                
                with st.spinner('🧠 جاري تحليل التنافس بالذكاء الاصطناعي (V8)...'):
                    report = run_analysis(df_raw, [x.strip() for x in brands.split(',')])
                
                if not report.empty:
                    report = report.sort_values(by=['Priority_Score'], ascending=False)
                    
                    # Top Metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("إجمالي التضارب", len(report))
                    m2.metric("حالات حرجة 🔥", len(report[report['Severity'] == "🔥 Critical"]))
                    m3.metric("زيارات ضائعة", f"{report['Est_Traffic_Loss'].sum():,}")
                    
                    st.markdown("### 📋 النتائج التفصيلية")
                    st.dataframe(report.style.applymap(lambda x: 'color: red; font-weight: bold' if x == '🔥 Critical' else '', subset=['Severity']), use_container_width=True)
                    
                    csv = report.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 تحميل التقرير (Excel/CSV)", csv, "report.csv", "text/csv")
                else:
                    st.success("✅ موقعك نظيف! لا يوجد cannibalization.")
            else:
                st.warning("لا توجد بيانات.")
    elif run_btn and not uploaded_secret:
        st.warning("⚠️ يرجى رفع ملف client_secret.json أولاً")
    else:
        st.info("👈 ابدأ برفع ملف المصادقة واضغط على زر الفحص.")