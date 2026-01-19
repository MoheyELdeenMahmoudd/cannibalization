import streamlit as st
import pandas as pd
import numpy as np
import math
import re
import os
import datetime
import io
from urllib.parse import urlparse
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

# ==========================================
# 🎨 1. UI/UX CONFIGURATION
# ==========================================
st.set_page_config(page_title="Almaster Tech - SEO Suite", page_icon="🚀", layout="wide")

# Custom CSS for Professional Look & RTL
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');
    
    * { font-family: 'Cairo', sans-serif !important; }
    
    .stApp {
        background: linear-gradient(to bottom, #0f172a, #1e293b);
        color: white;
    }
    
    /* Metrics */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 15px;
        text-align: right;
    }
    div[data-testid="stMetricValue"] { color: #3b82f6; font-size: 28px; }

    /* Tables */
    .stDataFrame { direction: ltr; } /* URLs need LTR */
    
    /* Buttons */
    .stButton>button {
        background: #2563eb; color: white; border-radius: 8px; font-weight: bold;
        transition: 0.3s; border: none;
    }
    .stButton>button:hover { background: #1d4ed8; }

    /* Inputs */
    .stTextInput input, .stSelectbox, .stMultiSelect {
        color: #333 !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <h1 style="color:white; margin:0;">ALMASTER <span style="color:#3b82f6;">TECH</span></h1>
    <p style="color:#94a3b8; font-size:14px;">Enterprise SEO Cannibalization System v11.0</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# ⚙️ 2. LOGIC CONFIGURATION
# ==========================================
class Config:
    MIN_IMP_QUANTILE = 0.2
    MIN_IMP_ABSOLUTE = 10
    MAX_DEPTH_PENALTY = 6
    WEIGHTS = {'clicks': 0.4, 'pos': 0.3, 'imps': 0.2, 'depth': 0.1}
    EXPECTED_CTR = {1: 30.0, 2: 15.0, 3: 10.0, 4: 7.0, 5: 5.0, 6: 3.5, 7: 3.0, 8: 2.5, 9: 2.0, 10: 1.5}
    URL_PATTERNS = {
        'Commercial': {'terms': ['/product', '/service', '/shop', 'cart', 'checkout', '/pricing'], 'weight': 3},
        'Informational': {'terms': ['/blog', '/news', '/article', '/guide', '/wiki', 'learn'], 'weight': 2}
    }
    COMM_TERMS = ['buy', 'price', 'cost', 'service', 'company', 'agency', 'hire', 'شراء', 'سعر', 'تكلفة', 'شركة', 'خدمة']
    INFO_TERMS = ['how', 'what', 'guide', 'tutorial', 'tips', 'why', 'review', 'vs', 'best', 'كيف', 'دليل', 'شرح', 'نصائح']

# ==========================================
# 🧠 3. ADVANCED LOGIC FUNCTIONS
# ==========================================
def identify_market_segment(url):
    try:
        path = urlparse(str(url)).path.strip('/').split('/')
        first = path[0].lower() if path else ""
        if re.match(r'^[a-z]{2}-[a-z]{2}$', first): return first.upper()
        if re.match(r'^[a-z]{2}$', first): return f"Global-{first.upper()}"
        return "Global-EN"
    except: return "Unknown"

def detect_intents_vectorized(df):
    # Vectorized String Operations for Speed
    q_lower = df['query'].str.lower()
    url_lower = df['page_clean'].str.lower()
    
    # Query Intent
    c_mask = q_lower.apply(lambda x: any(t in x for t in Config.COMM_TERMS))
    i_mask = q_lower.apply(lambda x: any(t in x for t in Config.INFO_TERMS))
    
    df['q_intent'] = np.select(
        [c_mask & ~i_mask, i_mask & ~c_mask], 
        ['Commercial', 'Informational'], 
        default='Navigational/General'
    )
    
    # Page Intent
    conditions = [url_lower.str.contains(t) for t in Config.URL_PATTERNS['Commercial']['terms']]
    df['p_intent'] = np.where(np.any(conditions, axis=0), 'Commercial', 'General')
    
    conditions_info = [url_lower.str.contains(t) for t in Config.URL_PATTERNS['Informational']['terms']]
    df['p_intent'] = np.where(np.any(conditions_info, axis=0), 'Informational', df['p_intent'])
    
    return df

def calculate_score_vectorized(df):
    # Normalize
    max_imp = df.groupby('query_market')['impressions'].transform('max')
    max_click = df.groupby('query_market')['clicks'].transform('max')
    
    norm_imp = df['impressions'] / max_imp.replace(0, 1)
    norm_click = df['clicks'] / max_click.replace(0, 1)
    
    # Log Position Score
    pos_score = 10 / (np.log(df['position'] + 1) + 1)
    
    # Depth Penalty
    depth = df['page_clean'].str.count('/').clip(upper=Config.MAX_DEPTH_PENALTY)
    depth_factor = 1 / (depth + 1)
    
    score = (norm_click * Config.WEIGHTS['clicks']) + \
            (pos_score * Config.WEIGHTS['pos']) + \
            (norm_imp * Config.WEIGHTS['imps']) + \
            (depth_factor * Config.WEIGHTS['depth'])
    
    return score

# ==========================================
# 🔌 4. DATA HANDLING (Pagination & Excel)
# ==========================================
@st.cache_resource
def authenticate_gsc(auth_code, _uploaded_secret):
    SCOPES = ['https://www.googleapis.com/auth/webmasters.readonly']
    try:
        with open("client_secret.json", "wb") as f:
            f.write(_uploaded_secret.getbuffer())
        
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
    
    # Progress Bar UI
    progress_text = "جاري سحب البيانات من جوجل... (قد يستغرق وقتاً للمواقع الكبيرة)"
    my_bar = st.progress(0, text=progress_text)
    
    while True:
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
            
            # Fake progress update
            my_bar.progress(min(start_row / 100000, 0.9), text=f"تم سحب {len(all_rows)} صف...")
            
            if len(rows) < batch_size: break
        except Exception as e:
            st.error(f"API Error: {e}")
            break
            
    my_bar.empty()
    return all_rows

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Cannibalization')
        workbook = writer.book
        worksheet = writer.sheets['Cannibalization']
        
        # Formats
        header_fmt = workbook.add_format({'bold': True, 'font_color': 'white', 'bg_color': '#010172', 'border': 1})
        red_fmt = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
        
        # Apply Header Format
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_fmt)
            
        # Adjust Column Width
        worksheet.set_column(0, 10, 20) # General
        worksheet.set_column(1, 1, 30) # Query
        
    return output.getvalue()

# ==========================================
# 📊 5. ANALYSIS ENGINE
# ==========================================
@st.cache_data
def run_full_analysis(df_raw, brands):
    df = df_raw.copy()
    df.columns = [c.lower() for c in df.columns]
    
    # Preprocessing
    df['page_clean'] = df['page'].astype(str).str.split('?').str[0].str.split('#').str[0].str.rstrip('/')
    df['market'] = df['page_clean'].apply(identify_market_segment)
    
    # Intent Detection
    df = detect_intents_vectorized(df)
    
    # Grouping
    df['query_market'] = df['query'] + "_" + df['market']
    df['weighted_pos'] = df['position'] * df['impressions']
    
    grp = df.groupby(['query', 'market', 'query_market', 'q_intent', 'page_clean', 'p_intent']).agg(
        clicks=('clicks', 'sum'),
        impressions=('impressions', 'sum'),
        weighted_pos=('weighted_pos', 'sum')
    ).reset_index()
    
    grp['position'] = grp['weighted_pos'] / grp['impressions']
    grp['ctr'] = (grp['clicks'] / grp['impressions'].replace(0, 1)) * 100
    
    # Filtering Low Traffic
    threshold = max(grp['impressions'].quantile(Config.MIN_IMP_QUANTILE), Config.MIN_IMP_ABSOLUTE)
    grp = grp[grp['impressions'] >= threshold].copy()
    
    # Scoring
    grp['score'] = calculate_score_vectorized(grp)
    
    # Cannibalization Detection
    report = []
    
    # Sort by score desc within each query group
    grp = grp.sort_values(['query_market', 'score'], ascending=[True, False])
    
    # Filter only groups with > 1 page
    dupes = grp[grp.duplicated('query_market', keep=False)]
    
    for qm, group in dupes.groupby('query_market'):
        winner = group.iloc[0]
        losers = group.iloc[1:]
        top_loser = losers.iloc[0]
        
        overlap = top_loser['score'] / winner['score']
        
        # Severity
        is_brand = any(b in winner['query'] for b in brands)
        severity_score = overlap * (1.2 if is_brand else 1.0)
        
        severity = "Low"
        if severity_score > Config.SEVERITY_CRITICAL: severity = "Critical" # Removed emoji for clean sort
        elif severity_score > Config.SEVERITY_HIGH: severity = "High"
        elif severity_score > Config.SEVERITY_MEDIUM: severity = "Medium"
        
        # Metrics
        wasted_imps = losers['impressions'].sum()
        traffic_loss = int(wasted_imps * (winner['ctr'] / 100))
        
        # Action Plan
        action = "Monitor"
        if winner['p_intent'] == top_loser['p_intent'] and severity == "Critical": action = "Merge / 301"
        elif winner['p_intent'] != top_loser['p_intent']: action = "Split Intent"
        
        report.append({
            'Query': winner['query'],
            'Market': winner['market'],
            'Severity': severity,
            'Winner': winner['page_clean'],
            'Loser': top_loser['page_clean'],
            'Overlap': round(overlap * 100, 1),
            'Traffic_Loss': traffic_loss,
            'Action': action,
            'W_Intent': winner['p_intent'],
            'L_Intent': top_loser['p_intent']
        })
        
    return pd.DataFrame(report)

# ==========================================
# 🖥️ 6. MAIN APPLICATION
# ==========================================

# Sidebar
with st.sidebar:
    st.header("⚙️ الإعدادات")
    uploaded_file = st.file_uploader("ملف المصادقة (JSON)", type="json")
    
    with st.expander("🔗 إعدادات الاتصال", expanded=True):
        site_url = st.text_input("GSC Property", "sc-domain:almaster.tech")
        days = st.slider("فترة التحليل", 7, 90, 30)
    
    with st.expander("🛡️ إعدادات البراند"):
        brands_input = st.text_area("كلمات البراند", "almaster, المستر, ماستر")
        brands_list = [x.strip() for x in brands_input.split(',')]

# Auth Flow
if uploaded_file:
    flow = InstalledAppFlow.from_client_secrets_file(uploaded_file, ['https://www.googleapis.com/auth/webmasters.readonly'])
    flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
    auth_url, _ = flow.authorization_url()
    
    if 'creds' not in st.session_state:
        st.info("👋 يرجى المصادقة للبدء")
        st.markdown(f"[**👉 اضغط هنا للحصول على كود المصادقة**]({auth_url})")
        auth_code = st.text_input("أدخل الكود هنا:")
        
        if auth_code:
            service = authenticate_gsc(auth_code, uploaded_file)
            if service:
                st.session_state.creds = service
                st.rerun()
            else:
                st.error("كود خاطئ! حاول مجدداً برابط جديد.")

# Main Dashboard
if 'creds' in st.session_state:
    service = st.session_state.creds
    
    if st.button("🚀 بدء الفحص الشامل"):
        with st.spinner("جاري سحب وتحليل البيانات..."):
            raw_data = fetch_all_data(service, site_url, days)
            
            if raw_data:
                df_raw = pd.DataFrame([{
                    'query': r['keys'][0], 'page': r['keys'][1],
                    'clicks': r['clicks'], 'impressions': r['impressions'],
                    'position': r['position']
                } for r in raw_data])
                
                report_df = run_full_analysis(df_raw, brands_list)
                st.session_state.report = report_df
            else:
                st.warning("لا توجد بيانات!")

    # Results View
    if 'report' in st.session_state:
        df = st.session_state.report
        
        # --- Filters ---
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            sev_filter = st.multiselect("تصفية حسب الخطورة", df['Severity'].unique(), default=['Critical', 'High'])
        with c2:
            act_filter = st.multiselect("تصفية حسب الإجراء", df['Action'].unique())
        with c3:
            market_filter = st.multiselect("تصفية حسب السوق", df['Market'].unique())
            
        # Apply Filters
        if sev_filter: df = df[df['Severity'].isin(sev_filter)]
        if act_filter: df = df[df['Action'].isin(act_filter)]
        if market_filter: df = df[df['Market'].isin(market_filter)]
        
        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("الكلمات المتضاربة", len(df))
        m2.metric("حالات حرجة", len(df[df['Severity']=='Critical']))
        m3.metric("زيارات ضائعة", f"{df['Traffic_Loss'].sum():,}")
        m4.metric("فرص دمج المحتوى", len(df[df['Action']=='Merge / 301']))
        
        # Data Table with Formatting
        st.subheader("📋 تقرير التضارب التفصيلي")
        
        def highlight_severity(s):
            return ['background-color: #450a0a' if v == 'Critical' else '' for v in s]

        st.dataframe(
            df.style.apply(highlight_severity, subset=['Severity'])
              .format({'Overlap': "{:.1f}%", 'Traffic_Loss': "{:,}"}),
            use_container_width=True,
            height=600
        )
        
        # Excel Export
        excel_data = to_excel(df)
        st.download_button(
            label="📥 تحميل التقرير (Excel XLSX)",
            data=excel_data,
            file_name=f'Cannibalization_Report_{datetime.date.today()}.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

else:
    if not uploaded_file:
        st.warning("⬅️ ابدأ برفع ملف client_secret.json من القائمة الجانبية.")
