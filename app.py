import streamlit as st
import pandas as pd
import numpy as np
import math
import re
import os
import datetime
import io
import urllib.parse
from urllib.parse import urlparse
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

# ==========================================
# 🎨 1. UI/UX CONFIGURATION
# ==========================================
st.set_page_config(page_title="Almaster Tech - SEO Suite", page_icon="🚀", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif !important; }
    
    .stApp {
        background: linear-gradient(to bottom, #0f172a, #1e293b, #0f172a);
        background-size: cover;
        background-attachment: fixed;
    }

    /* RTL Alignment */
    p, h1, h2, h3, h4, h5, h6, .stMarkdown, .stRadio, .stSelectbox label, .stTextInput label, .stTextArea label {
        direction: rtl; 
        text-align: right;
    }
    
    div[data-testid="stExpander"] details summary {
        flex-direction: row-reverse;
        text-align: right;
    }

    /* Metrics Cards */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        direction: rtl;
    }
    div[data-testid="stMetricValue"] { color: #3b82f6; font-size: 28px; font-weight: 800; }
    div[data-testid="stMetricLabel"] { color: #cbd5e1; }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #2563eb, #3b82f6);
        color: white; border: none; border-radius: 8px; font-weight: bold;
        transition: 0.3s; height: 50px;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(37, 99, 235, 0.4); }

    .stDataFrame { direction: ltr; } 
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="text-align: center; margin-bottom: 30px; background: rgba(0,0,0,0.2); padding: 20px; border-radius: 15px;">
    <h1 style="color:white; margin:0;">ALMASTER <span style="color:#3b82f6;">TECH</span></h1>
    <p style="color:#94a3b8; font-size:16px;">Enterprise SEO Cannibalization System v14.0</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# ⚙️ 2. LOGIC CONFIGURATION
# ==========================================
class Config:
    SEVERITY_CRITICAL = 0.8
    SEVERITY_HIGH = 0.5
    SEVERITY_MEDIUM = 0.3
    
    MIN_IMP_QUANTILE = 0.2
    MIN_IMP_ABSOLUTE = 10
    MAX_DEPTH_PENALTY = 6
    WEIGHTS = {'clicks': 0.4, 'pos': 0.3, 'imps': 0.2, 'depth': 0.1}
    EXPECTED_CTR = {1: 30.0, 2: 15.0, 3: 10.0, 4: 7.0, 5: 5.0, 6: 3.5, 7: 3.0, 8: 2.5, 9: 2.0, 10: 1.5}
    URL_PATTERNS = {
        'Commercial': {'terms': ['/product', '/service', '/shop', 'cart', 'checkout', '/pricing'], 'weight': 3},
        'Informational': {'terms': ['/blog', '/news', '/article', '/guide', '/wiki', 'learn', '/tag/', '/doctor/'], 'weight': 2}
    }
    COMM_TERMS = ['buy', 'price', 'cost', 'service', 'company', 'agency', 'hire', 'شراء', 'سعر', 'تكلفة', 'شركة', 'خدمة', 'عيادة', 'دكتور', 'حجز']
    INFO_TERMS = ['how', 'what', 'guide', 'tutorial', 'tips', 'why', 'review', 'vs', 'best', 'كيف', 'دليل', 'شرح', 'نصائح', 'علاج', 'أعراض', 'اسباب']

# ==========================================
# 🧠 3. SMART LOGIC FUNCTIONS
# ==========================================
def identify_market_segment(url, default_lang="EN"):
    try:
        decoded_url = urllib.parse.unquote(str(url))
        path = urlparse(decoded_url).path.strip('/').split('/')
        first = path[0].lower() if path else ""
        
        if re.match(r'^[a-z]{2}-[a-z]{2}$', first): return first.upper()
        if re.match(r'^[a-z]{2}$', first): return f"Global-{first.upper()}"
        if re.search(r'[\u0600-\u06FF]', decoded_url): return "Global-AR"
        return f"Global-{default_lang}"
    except: return "Unknown"

def detect_intents_vectorized(df):
    q_lower = df['query'].str.lower()
    url_decoded = df['page_clean'].apply(lambda x: urllib.parse.unquote(str(x)).lower())

    c_mask = q_lower.apply(lambda x: any(t in x for t in Config.COMM_TERMS))
    i_mask = q_lower.apply(lambda x: any(t in x for t in Config.INFO_TERMS))
    
    df['q_intent'] = np.select(
        [c_mask & ~i_mask, i_mask & ~c_mask], 
        ['Commercial', 'Informational'], 
        default='Navigational/General'
    )
    
    p_intent = pd.Series('General', index=df.index)
    for term in Config.URL_PATTERNS['Informational']['terms']:
        p_intent[url_decoded.str.contains(term, regex=False)] = 'Informational'
    for term in Config.URL_PATTERNS['Commercial']['terms']:
        p_intent[url_decoded.str.contains(term, regex=False)] = 'Commercial'
        
    df['p_intent'] = p_intent
    return df

def calculate_score_vectorized(df):
    max_imp = df.groupby('query_market')['impressions'].transform('max').replace(0, 1)
    max_click = df.groupby('query_market')['clicks'].transform('max').replace(0, 1)
    
    norm_imp = df['impressions'] / max_imp
    norm_click = df['clicks'] / max_click
    pos_score = 10 / (np.log(df['position'] + 1) + 1)
    
    depth = df['page_clean'].str.count('/').clip(upper=Config.MAX_DEPTH_PENALTY)
    depth_factor = 1 / (depth + 1)
    
    score = (norm_click * Config.WEIGHTS['clicks']) + \
            (pos_score * Config.WEIGHTS['pos']) + \
            (norm_imp * Config.WEIGHTS['imps']) + \
            (depth_factor * Config.WEIGHTS['depth'])
    
    return score

# ==========================================
# 🔌 4. DATA HANDLING
# ==========================================
@st.cache_resource
def authenticate_gsc(auth_code):
    SCOPES = ['https://www.googleapis.com/auth/webmasters.readonly']
    try:
        flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", SCOPES)
        flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
        flow.fetch_token(code=auth_code)
        return build('searchconsole', 'v1', credentials=flow.credentials)
    except Exception as e:
        return None

def fetch_all_data(service, site_url, days):
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days)
    all_rows = []
    start_row = 0
    batch_size = 25000
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while True:
        status_text.text(f"📡 جاري سحب البيانات... (تم سحب {len(all_rows)} صف)")
        req = {
            'startDate': start_date.isoformat(), 'endDate': end_date.isoformat(),
            'dimensions': ['query', 'page'], 'rowLimit': batch_size, 'startRow': start_row
        }
        try:
            resp = service.searchanalytics().query(siteUrl=site_url, body=req).execute()
            rows = resp.get('rows', [])
            if not rows: break
            
            all_rows.extend(rows)
            start_row += len(rows)
            progress = min(start_row / 100000, 0.95) 
            progress_bar.progress(progress)
            
            if len(rows) < batch_size: break
        except Exception as e:
            st.error(f"API Error: {e}")
            break
            
    progress_bar.empty()
    status_text.empty()
    return all_rows

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Analysis')
        workbook = writer.book
        worksheet = writer.sheets['Analysis']
        
        header_fmt = workbook.add_format({'bold': True, 'font_color': 'white', 'bg_color': '#0f172a', 'border': 1, 'align': 'center'})
        critical_fmt = workbook.add_format({'bg_color': '#fee2e2', 'font_color': '#991b1b'})
        
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_fmt)
        
        sev_col_idx = df.columns.get_loc('Severity')
        worksheet.conditional_format(1, sev_col_idx, len(df), sev_col_idx, {
            'type': 'text', 'criteria': 'containing', 'value': 'Critical', 'format': critical_fmt
        })
        worksheet.set_column(0, 0, 15) # Market
        worksheet.set_column(1, 1, 30) # Query
        worksheet.set_column(4, 4, 25) # Reason
        worksheet.set_column(8, 8, 50) # Winner
        worksheet.set_column(10, 10, 50) # Loser
    return output.getvalue()

# ==========================================
# 📊 5. ANALYSIS ENGINE
# ==========================================
@st.cache_data(show_spinner=False)
def run_full_analysis(df_raw, brands, default_lang):
    df = df_raw.copy()
    df.columns = [c.lower() for c in df.columns]
    
    df['page_clean'] = df['page'].astype(str).str.split('?').str[0].str.split('#').str[0].str.rstrip('/')
    df['market'] = df['page_clean'].apply(lambda x: identify_market_segment(x, default_lang))
    
    df = detect_intents_vectorized(df)
    
    df['query_market'] = df['query'] + "_" + df['market']
    df['weighted_pos'] = df['position'] * df['impressions']
    
    grp = df.groupby(['query', 'market', 'query_market', 'q_intent', 'page_clean', 'p_intent']).agg(
        clicks=('clicks', 'sum'),
        impressions=('impressions', 'sum'),
        weighted_pos=('weighted_pos', 'sum')
    ).reset_index()
    
    grp['position'] = grp['weighted_pos'] / grp['impressions']
    grp['ctr'] = (grp['clicks'] / grp['impressions'].replace(0, 1)) * 100
    
    threshold = max(grp['impressions'].quantile(Config.MIN_IMP_QUANTILE), Config.MIN_IMP_ABSOLUTE)
    grp = grp[grp['impressions'] >= threshold].copy()
    
    grp['score'] = calculate_score_vectorized(grp)
    
    report = []
    grp = grp.sort_values(['query_market', 'score'], ascending=[True, False])
    dupes = grp[grp.duplicated('query_market', keep=False)]
    
    for qm, group in dupes.groupby('query_market'):
        winner = group.iloc[0]
        losers = group.iloc[1:]
        top_loser = losers.iloc[0]
        
        overlap = top_loser['score'] / winner['score']
        is_brand = any(b in winner['query'] for b in brands)
        severity_score = overlap * (1.2 if is_brand else 1.0)
        
        # Severity
        severity = "Low"
        if severity_score > Config.SEVERITY_CRITICAL: severity = "Critical"
        elif severity_score > Config.SEVERITY_HIGH: severity = "High"
        elif severity_score > Config.SEVERITY_MEDIUM: severity = "Medium"
        
        # --- NEW: Reason Logic ---
        reason = "Unknown"
        if winner['p_intent'] != top_loser['p_intent']:
            reason = "Intent Mismatch (اختلاف نية)"
        elif winner['p_intent'] == top_loser['p_intent']:
            reason = "Duplicate Content (تكرار محتوى)"
        
        if is_brand:
            reason = "Brand Conflict (تضارب براند)"

        # Action & Difficulty
        action = "Monitor"
        difficulty = "Low"
        
        if severity == "Critical":
            if reason == "Intent Mismatch (اختلاف نية)":
                action = "Split Intent (فصل المحتوى)"
                difficulty = "High"
            else:
                action = "Merge / 301 (دمج)"
                difficulty = "Low"
        
        # Metrics
        wasted_imps = losers['impressions'].sum()
        traffic_loss = int(wasted_imps * (winner['ctr'] / 100))
        
        # Priority Score
        sev_points = {"Critical": 40, "High": 25, "Medium": 10, "Low": 5}.get(severity, 5)
        traffic_factor = min(math.log(wasted_imps + 1) * 10, 50)
        brand_points = 10 if is_brand else 0
        priority_score = int(min(traffic_factor + sev_points + brand_points, 100))
        
        report.append({
            'Market': winner['market'],
            'Query': winner['query'],
            'Severity': severity,
            'Priority_Score': priority_score,
            'Reason': reason, # New Column
            'Action_Plan': action,
            'Difficulty': difficulty,
            'Est_Traffic_Loss': traffic_loss,
            'Winner': winner['page_clean'],
            'Winner_CTR': round(winner['ctr'], 2),
            'Loser': top_loser['page_clean'],
            'Overlap': round(overlap * 100, 1)
        })
        
    return pd.DataFrame(report)

# ==========================================
# 🖥️ 6. MAIN APPLICATION
# ==========================================
with st.sidebar:
    st.header("⚙️ الإعدادات")
    uploaded_file = st.file_uploader("ملف المصادقة (JSON)", type="json")
    
    with st.expander("🔗 إعدادات الاتصال", expanded=True):
        site_url = st.text_input("GSC Property", "sc-domain:almaster.tech")
        days = st.slider("فترة التحليل", 7, 90, 30)
        default_lang_choice = st.radio("اللغة الافتراضية", ["AR", "EN"], index=0)

    with st.expander("🛡️ إعدادات البراند"):
        brands_input = st.text_area("كلمات البراند", "almaster, المستر, ماستر")
        brands_list = [x.strip() for x in brands_input.split(',')]

# Auth Flow
if uploaded_file:
    with open("client_secret.json", "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", ['https://www.googleapis.com/auth/webmasters.readonly'])
    flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
    auth_url, _ = flow.authorization_url()
    
    if 'creds' not in st.session_state:
        st.info("👋 مرحباً بك! يرجى المصادقة.")
        st.markdown(f"""
        <a href="{auth_url}" target="_blank" style="background:#3b82f6; color:white; padding:10px 20px; border-radius:10px; text-decoration:none; display:block; text-align:center;">
        👉 1. اضغط هنا للحصول على كود جديد
        </a>
        """, unsafe_allow_html=True)
        auth_code = st.text_input("2. الصق الكود هنا:", placeholder="4/1A...")
        
        if auth_code:
            service = authenticate_gsc(auth_code)
            if service:
                st.session_state.creds = service
                st.rerun()
            else:
                st.error("❌ كود خاطئ!")

if 'creds' in st.session_state:
    service = st.session_state.creds
    
    if st.button("🚀 بدء الفحص الشامل"):
        with st.spinner("جاري تهيئة المحرك..."):
            raw_data = fetch_all_data(service, site_url, days)
            
            if raw_data:
                df_raw = pd.DataFrame([{
                    'query': r['keys'][0], 'page': r['keys'][1],
                    'clicks': r['clicks'], 'impressions': r['impressions'],
                    'position': r['position']
                } for r in raw_data])
                
                with st.spinner("🤖 جاري تحليل التضارب..."):
                    report_df = run_full_analysis(df_raw, brands_list, default_lang_choice)
                    st.session_state.report = report_df
            else:
                st.warning("⚠️ لا توجد بيانات!")

    if 'report' in st.session_state:
        df = st.session_state.report
        
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            sev_filter = st.multiselect("تصفية حسب الخطورة", df['Severity'].unique(), default=['Critical', 'High'])
        with c2:
            market_filter = st.multiselect("تصفية حسب السوق", df['Market'].unique())
        with c3:
            reason_filter = st.multiselect("تصفية حسب السبب", df['Reason'].unique()) # New Filter
            
        if sev_filter: df = df[df['Severity'].isin(sev_filter)]
        if market_filter: df = df[df['Market'].isin(market_filter)]
        if reason_filter: df = df[df['Reason'].isin(reason_filter)]
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🔍 الكلمات المتضاربة", len(df))
        m2.metric("🔥 حالات حرجة", len(df[df['Severity']=='Critical']))
        m3.metric("📉 زيارات ضائعة", f"{df['Est_Traffic_Loss'].sum():,}")
        m4.metric("🧠 اختلاف نية البحث", len(df[df['Reason'].str.contains("Intent")]))
        
        st.subheader("📋 تقرير التضارب التفصيلي")
        
        def highlight_row(s):
            return ['background-color: rgba(255, 0, 0, 0.2)' if s.Severity == 'Critical' else '' for _ in s]

        st.dataframe(
            df.style.apply(highlight_row, axis=1)
              .format({'Overlap': "{:.1f}%", 'Winner_CTR': "{:.1f}%", 'Est_Traffic_Loss': "{:,}"}),
            use_container_width=True,
            height=600
        )
        
        excel_data = to_excel(df)
        st.download_button(
            label="📥 تحميل التقرير (Excel XLSX)",
            data=excel_data,
            file_name=f'SEO_Audit_{datetime.date.today()}.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
else:
    if not uploaded_file:
        st.info("⬅️ ابدأ برفع ملف المصادقة (client_secret.json) من القائمة الجانبية.")
