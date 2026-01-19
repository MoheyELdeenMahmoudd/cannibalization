import streamlit as st
import pandas as pd
import numpy as np
import math
import re
import os
import datetime
import io
import requests
import urllib.parse
from urllib.parse import urlparse
from fpdf import FPDF
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

# ==========================================
# 🎨 1. UI/UX CONFIGURATION
# ==========================================
st.set_page_config(page_title="Almaster Tech - SEO Command Center", page_icon="🛸", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif !important; }
    
    .stApp {
        background: linear-gradient(to bottom, #0f172a, #1e293b, #0f172a);
        color: white;
    }

    /* RTL */
    p, h1, h2, h3, h4, h5, h6, .stMarkdown, .stRadio, .stSelectbox label, .stTextInput label, .stTextArea label {
        direction: rtl; 
        text-align: right;
    }
    
    div[data-testid="stExpander"] details summary {
        flex-direction: row-reverse;
        text-align: right;
    }

    /* Cards */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        direction: rtl;
    }
    div[data-testid="stMetricValue"] { color: #38bdf8; font-size: 26px; }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #0ea5e9, #2563eb);
        color: white; border: none; height: 50px; font-weight: bold;
    }

    .stDataFrame { direction: ltr; } 
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="text-align: center; margin-bottom: 30px; background: rgba(0,0,0,0.3); padding: 20px; border-radius: 15px;">
    <h1 style="color:white; margin:0;">ALMASTER <span style="color:#38bdf8;">TECH</span></h1>
    <p style="color:#94a3b8; font-size:16px;">SEO Command Center v15.0</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# ⚙️ 2. CONFIG & HELPERS
# ==========================================
class Config:
    SEVERITY_CRITICAL = 0.8
    SEVERITY_HIGH = 0.5
    MIN_IMP_QUANTILE = 0.2
    MIN_IMP_ABSOLUTE = 10
    MAX_DEPTH_PENALTY = 6
    WEIGHTS = {'clicks': 0.4, 'pos': 0.3, 'imps': 0.2, 'depth': 0.1}
    
    # Enhanced NLP Patterns
    URL_PATTERNS = {
        'Commercial': {'terms': ['/product', '/service', '/shop', 'cart', 'checkout', '/pricing', 'booking', 'store'], 'weight': 3},
        'Informational': {'terms': ['/blog', '/news', '/article', '/guide', '/wiki', 'learn', '/tag/', '/doctor/', 'faq'], 'weight': 2}
    }
    COMM_TERMS = ['buy', 'price', 'cost', 'service', 'company', 'agency', 'hire', 'شراء', 'سعر', 'تكلفة', 'شركة', 'خدمة', 'عيادة', 'دكتور', 'حجز', 'متجر', 'طلب']
    INFO_TERMS = ['how', 'what', 'guide', 'tutorial', 'tips', 'why', 'review', 'vs', 'best', 'كيف', 'دليل', 'شرح', 'نصائح', 'علاج', 'أعراض', 'اسباب', 'معلومات', 'خطوات']

def send_slack_alert(webhook_url, critical_count, site_url):
    if not webhook_url: return
    payload = {
        "text": f"🚨 *Critical SEO Alert for {site_url}*\nFound *{critical_count}* critical cannibalization issues requiring immediate attention."
    }
    try:
        requests.post(webhook_url, json=payload)
        st.toast("تم إرسال تنبيه Slack بنجاح!", icon="🔔")
    except:
        st.toast("فشل إرسال تنبيه Slack", icon="❌")

def create_pdf_report(df, site_url):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"SEO Cannibalization Report: {site_url}", ln=1, align='C')
    pdf.cell(200, 10, txt=f"Date: {datetime.date.today()}", ln=2, align='C')
    pdf.ln(10)
    
    critical = df[df['Severity'] == 'Critical']
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 10, f"Critical Issues Found: {len(critical)}", ln=1)
    pdf.cell(0, 10, f"Total Traffic Loss: {df['Est_Traffic_Loss'].sum()}", ln=1)
    
    pdf.ln(10)
    pdf.set_font("Arial", size=8)
    
    # Simple Table
    col_width = 45
    row_height = 6
    headers = ['Query', 'Market', 'Winner', 'Loser']
    for h in headers:
        pdf.cell(col_width, row_height, h, border=1)
    pdf.ln(row_height)
    
    for _, row in critical.head(20).iterrows(): # Top 20 only for PDF
        pdf.cell(col_width, row_height, str(row['Query'])[:25], border=1)
        pdf.cell(col_width, row_height, str(row['Market']), border=1)
        pdf.cell(col_width, row_height, str(row['Winner'])[-20:], border=1)
        pdf.cell(col_width, row_height, str(row['Loser'])[-20:], border=1)
        pdf.ln(row_height)
        
    return pdf.output(dest='S').encode('latin-1', 'ignore')

# ==========================================
# 🧠 3. CORE LOGIC (Vectorized & Multi-Page)
# ==========================================
def identify_market_segment(url, default_lang="EN"):
    try:
        decoded = urllib.parse.unquote(str(url))
        path = urlparse(decoded).path.strip('/').split('/')
        first = path[0].lower() if path else ""
        if re.match(r'^[a-z]{2}-[a-z]{2}$', first): return first.upper()
        if re.match(r'^[a-z]{2}$', first): return f"Global-{first.upper()}"
        if re.search(r'[\u0600-\u06FF]', decoded): return "Global-AR"
        return f"Global-{default_lang}"
    except: return "Unknown"

def detect_intents(df):
    q_lower = df['query'].str.lower()
    url_decoded = df['page_clean'].apply(lambda x: urllib.parse.unquote(str(x)).lower())

    c_mask = q_lower.apply(lambda x: any(t in x for t in Config.COMM_TERMS))
    i_mask = q_lower.apply(lambda x: any(t in x for t in Config.INFO_TERMS))
    
    df['q_intent'] = np.select([c_mask & ~i_mask, i_mask & ~c_mask], ['Commercial', 'Informational'], default='General')
    
    p_intent = pd.Series('General', index=df.index)
    for term in Config.URL_PATTERNS['Informational']['terms']:
        p_intent[url_decoded.str.contains(term, regex=False)] = 'Informational'
    for term in Config.URL_PATTERNS['Commercial']['terms']:
        p_intent[url_decoded.str.contains(term, regex=False)] = 'Commercial'
    
    df['p_intent'] = p_intent
    return df

def calculate_score(df):
    max_imp = df.groupby('query_market')['impressions'].transform('max').replace(0, 1)
    max_click = df.groupby('query_market')['clicks'].transform('max').replace(0, 1)
    
    score = (df['clicks']/max_click * Config.WEIGHTS['clicks']) + \
            ((10 / (np.log(df['position'] + 1) + 1)) * Config.WEIGHTS['pos']) + \
            (df['impressions']/max_imp * Config.WEIGHTS['imps']) + \
            ((1 / (df['page_clean'].str.count('/').clip(upper=6) + 1)) * Config.WEIGHTS['depth'])
    return score

# ==========================================
# 🔌 4. DATA & API
# ==========================================
@st.cache_resource
def authenticate_gsc(auth_code):
    try:
        flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", ['https://www.googleapis.com/auth/webmasters.readonly'])
        flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
        flow.fetch_token(code=auth_code)
        return build('searchconsole', 'v1', credentials=flow.credentials)
    except Exception as e:
        return None

def get_site_list(service):
    """Fetch all verified properties from GSC"""
    try:
        site_list = service.sites().list().execute()
        return [s['siteUrl'] for s in site_list.get('siteEntry', [])]
    except Exception as e:
        st.error(f"Error fetching sites: {e}")
        return []

def fetch_gsc_data(service, site_url, days):
    end = datetime.date.today()
    start = end - datetime.timedelta(days=days)
    all_rows = []
    start_row = 0
    batch = 25000
    
    bar = st.progress(0)
    status = st.empty()
    
    while True:
        status.text(f"📡 جاري سحب البيانات... ({len(all_rows)} صف)")
        try:
            req = {'startDate': start.isoformat(), 'endDate': end.isoformat(), 'dimensions': ['query', 'page'], 'rowLimit': batch, 'startRow': start_row}
            resp = service.searchanalytics().query(siteUrl=site_url, body=req).execute()
            rows = resp.get('rows', [])
            if not rows: break
            all_rows.extend(rows)
            start_row += len(rows)
            bar.progress(min(start_row / 100000, 0.95))
            if len(rows) < batch: break
        except Exception as e:
            st.error(f"API Error: {e}")
            break
            
    bar.empty()
    status.empty()
    return all_rows

# ==========================================
# 📊 5. MULTI-PAGE ANALYSIS ENGINE
# ==========================================
@st.cache_data(show_spinner=False)
def run_analysis(df_raw, brands, default_lang):
    df = df_raw.copy()
    df.columns = [c.lower() for c in df.columns]
    
    df['page_clean'] = df['page'].astype(str).str.split('?').str[0].str.split('#').str[0].str.rstrip('/')
    df['market'] = df['page_clean'].apply(lambda x: identify_market_segment(x, default_lang))
    df = detect_intents(df)
    
    df['query_market'] = df['query'] + "_" + df['market']
    df['score'] = calculate_score(df)
    
    report = []
    
    # Group by Query+Market and find ALL conflicts (not just top loser)
    groups = df.groupby('query_market')
    
    for qm, group in groups:
        if len(group) < 2: continue
        
        # Sort pages by score
        group = group.sort_values('score', ascending=False)
        winner = group.iloc[0]
        losers = group.iloc[1:]
        
        # Filter insignificant losers (low impressions)
        min_imp_check = max(group['impressions'].max() * 0.05, 5)
        significant_losers = losers[losers['impressions'] >= min_imp_check]
        
        if significant_losers.empty: continue
        
        # Iterate over ALL significant losers
        for _, loser in significant_losers.iterrows():
            overlap = loser['score'] / winner['score']
            is_brand = any(b in winner['query'] for b in brands)
            severity_score = overlap * (1.2 if is_brand else 1.0)
            
            severity = "Low"
            if severity_score > Config.SEVERITY_CRITICAL: severity = "Critical"
            elif severity_score > Config.SEVERITY_HIGH: severity = "High"
            elif severity_score > Config.SEVERITY_MEDIUM: severity = "Medium"
            
            reason = "Duplicate Content" if winner['p_intent'] == loser['p_intent'] else "Intent Mismatch"
            if is_brand: reason = "Brand Conflict"
            
            action = "Merge / 301" if severity == "Critical" and reason != "Intent Mismatch" else "Split Intent"
            if severity == "Low": action = "Monitor"
            
            traffic_loss = int(loser['impressions'] * (winner['ctr']/100)) # Potential loss
            
            # Priority
            sev_pts = {"Critical": 40, "High": 25, "Medium": 10}.get(severity, 5)
            tf_pts = min(math.log(traffic_loss+1)*10, 50)
            priority = int(min(tf_pts + sev_pts + (10 if is_brand else 0), 100))
            
            report.append({
                'Market': winner['market'],
                'Query': winner['query'],
                'Severity': severity,
                'Priority': priority,
                'Reason': reason,
                'Action': action,
                'Traffic_Loss': traffic_loss,
                'Winner': winner['page_clean'],
                'Loser': loser['page_clean'],
                'Overlap': round(overlap * 100, 1),
                'Winner_Intent': winner['p_intent'],
                'Loser_Intent': loser['p_intent']
            })
            
    return pd.DataFrame(report)

# ==========================================
# 🖥️ 6. MAIN APP
# ==========================================
with st.sidebar:
    st.header("⚙️ مركز التحكم")
    uploaded_file = st.file_uploader("ملف المصادقة (JSON)", type="json")
    
    st.markdown("---")
    
    # 1. Auth & Site Selection
    if 'creds' in st.session_state:
        st.success("✅ متصل بـ GSC")
        
        # Fetch Sites Button
        if 'sites' not in st.session_state:
            with st.spinner("جلب قائمة المواقع..."):
                st.session_state.sites = get_site_list(st.session_state.creds)
        
        if st.session_state.sites:
            selected_site = st.selectbox("🌐 اختر الموقع (Property)", st.session_state.sites)
        else:
            selected_site = st.text_input("أو اكتب الرابط يدوياً", "sc-domain:example.com")
            
    else:
        st.warning("يرجى المصادقة أولاً")
        selected_site = None

    # Settings
    days = st.slider("فترة التحليل", 7, 90, 30)
    default_lang = st.radio("اللغة الافتراضية", ["AR", "EN"])
    
    with st.expander("🔔 إعدادات التنبيهات (Slack)"):
        slack_webhook = st.text_input("Slack Webhook URL", placeholder="https://hooks.slack.com/...")
        
    with st.expander("🛡️ إعدادات البراند"):
        brands_str = st.text_area("كلمات البراند", "almaster, المستر, ماستر")
        brands = [x.strip() for x in brands_str.split(',')]

# Auth Logic
if uploaded_file and 'creds' not in st.session_state:
    with open("client_secret.json", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", ['https://www.googleapis.com/auth/webmasters.readonly'])
    flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
    auth_url, _ = flow.authorization_url()
    
    st.info("👋 تسجيل الدخول مطلوب")
    st.markdown(f"[👉 **اضغط هنا للمصادقة**]({auth_url})")
    code = st.text_input("الصق الكود هنا:")
    if code:
        srv = authenticate_gsc(code)
        if srv:
            st.session_state.creds = srv
            st.rerun()
        else:
            st.error("كود خاطئ")

# Main Execution
if 'creds' in st.session_state and selected_site:
    if st.button("🚀 تشغيل الفحص الشامل"):
        with st.spinner(f"جاري تحليل {selected_site}..."):
            raw = fetch_gsc_data(st.session_state.creds, selected_site, days)
            if raw:
                df_raw = pd.DataFrame([{
                    'query': r['keys'][0], 'page': r['keys'][1],
                    'clicks': r['clicks'], 'impressions': r['impressions'],
                    'position': r['position']
                } for r in raw])
                
                with st.spinner("🤖 تحليل التضارب العميق (Multi-Page)..."):
                    report = run_analysis(df_raw, brands, default_lang)
                    st.session_state.report = report
                    
                    # Slack Alert
                    critical_count = len(report[report['Severity']=='Critical'])
                    if critical_count > 0 and slack_webhook:
                        send_slack_alert(slack_webhook, critical_count, selected_site)
            else:
                st.warning("لا توجد بيانات")

# Results Dashboard
if 'report' in st.session_state:
    df = st.session_state.report
    
    # Top Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 تضارب حرج", len(df[df['Severity']=='Critical']))
    c2.metric("⚠️ تضارب عالى", len(df[df['Severity']=='High']))
    c3.metric("📉 زيارات مهددة", f"{df['Traffic_Loss'].sum():,}")
    c4.metric("📄 صفحات متأثرة", df['Loser'].nunique())
    
    # Filters
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sev_fil = st.multiselect("تصفية الخطورة", df['Severity'].unique(), default=['Critical'])
    with col2:
        mkt_fil = st.multiselect("تصفية السوق", df['Market'].unique())
        
    filtered_df = df.copy()
    if sev_fil: filtered_df = filtered_df[filtered_df['Severity'].isin(sev_fil)]
    if mkt_fil: filtered_df = filtered_df[filtered_df['Market'].isin(mkt_fil)]
    
    st.dataframe(
        filtered_df.style.apply(lambda x: ['background-color: rgba(255,0,0,0.1)' if v == 'Critical' else '' for v in x], axis=1),
        use_container_width=True,
        height=500
    )
    
    # Exports
    col_dl1, col_dl2 = st.columns(2)
    
    # Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        filtered_df.to_excel(writer, index=False)
    col_dl1.download_button("📥 Excel Report", output.getvalue(), "seo_audit.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    # PDF
    try:
        pdf_bytes = create_pdf_report(filtered_df, selected_site)
        col_dl2.download_button("📄 PDF Summary", pdf_bytes, "seo_summary.pdf", "application/pdf")
    except Exception as e:
        col_dl2.warning("PDF غير متاح (Check Fonts)")

else:
    if 'creds' in st.session_state:
        st.info("👈 اختر الموقع واضغط زر التشغيل")
