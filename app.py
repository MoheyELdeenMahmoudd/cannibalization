import streamlit as st
import pandas as pd
import numpy as np
import io
import datetime
import re
import requests
import urllib.parse
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# ==========================================
# 🎨 1. UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="Almaster Tech | SEO Command Center", page_icon="🦁", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif !important; }
    .stApp { background-color: #0e1117; color: #ffffff; }
    .header-box {
        background: linear-gradient(90deg, #1e293b, #0f172a);
        padding: 20px; border-radius: 12px; border-left: 5px solid #38bdf8;
        margin-bottom: 25px; text-align: center;
    }
    div[data-testid="stMetric"] {
        background-color: #1f2937; border: 1px solid #374151;
        border-radius: 10px; padding: 15px; direction: rtl;
    }
    .rtl { direction: rtl; text-align: right; }
    .stSelectbox, .stTextInput, .stSlider { direction: rtl; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
    <h1 style="color:white; margin:0;">ALMASTER <span style="color:#38bdf8;">TECH</span></h1>
    <p style="color:#94a3b8; font-size:16px;">SEO Command Center v5.0 (Geo-Targeting & Live Logic)</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# ⚙️ 2. CONFIG & LOGIC DEFINITIONS
# ==========================================
class Config:
    DOMINANCE_TOP = 3.5  
    DOMINANCE_SEC = 6.0 
    MIN_IMPS = 10     
    
    # Intent Terms
    COMM_TERMS = ['buy', 'price', 'service', 'agency', 'shop', 'store', 'book', 'شراء', 'سعر', 'اسعار', 'خدمة', 'شركة', 'وكالة', 'متجر', 'حجز', 'عيادة', 'دكتور']
    INFO_TERMS = ['how', 'what', 'guide', 'tips', 'best', 'review', 'vs', 'signs', 'causes', 'كيف', 'ما هو', 'دليل', 'شرح', 'نصائح', 'افضل', 'مقارنة', 'اعراض', 'علاج', 'اسباب']

# ==========================================
# 🌍 3. GEO & MARKET LOGIC (THE FIX)
# ==========================================
def detect_market(url, query):
    """
    Detects if the page is targeting Saudi, Egypt, Global, or General Arabic.
    """
    url_str = str(url).lower()
    path = urlparse(url_str).path
    query_str = str(query)
    
    # 1. Explicit Geo-Patterns in URL
    if '/sa/' in path or '.sa' in url_str or '-sa' in path:
        return "Saudi (KSA)"
    if '/eg/' in path or '.eg' in url_str or '-eg' in path:
        return "Egypt (EG)"
    if '/ae/' in path or '.ae' in url_str:
        return "UAE"
    
    # 2. Language Detection (Ar vs En)
    is_arabic_query = bool(re.search(r'[\u0600-\u06FF]', query_str))
    is_arabic_url = bool(re.search(r'[\u0600-\u06FF]', url_str)) or '/ar/' in path
    
    if is_arabic_query or is_arabic_url:
        return "Arabic (General)"
    
    return "Global (English)"

def get_page_intent(url, query):
    url_lower = str(url).lower()
    query_lower = str(query).lower()
    score_comm, score_info = 0, 0
    
    if any(t in query_lower for t in Config.COMM_TERMS): score_comm += 2
    if any(t in query_lower for t in Config.INFO_TERMS): score_info += 2
    if any(x in url_lower for x in ['/product', '/cart', '/checkout', '/services']): score_comm += 3
    if any(x in url_lower for x in ['/blog', '/article', '/news', '/wiki', '/guide']): score_info += 3
    
    if score_comm > score_info: return "Commercial"
    if score_info > score_comm: return "Informational"
    return "Ambiguous"

# ==========================================
# 🕵️ 4. LIVE VALIDATOR (REDIRECTION CHECK)
# ==========================================
def get_live_status(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AlmasterTechBot/1.0'}
        # allow_redirects=True follows the chain. history checks if it happened.
        response = requests.head(url, allow_redirects=True, timeout=5, headers=headers)
        was_redirected = bool(response.history) or response.status_code in [301, 302, 307, 308]
        return response.status_code, was_redirected, response.url
    except:
        return 0, False, url

# ==========================================
# 🧠 5. ANALYSIS ENGINE
# ==========================================
def classify_issue(row, brands):
    w_pos = row['Winner_Pos']
    l_pos = row['Loser_Pos']
    w_mkt = row['Winner_Market']
    l_mkt = row['Loser_Market']
    query = row['Query']
    
    # 1. Geo/Market Conflict (New Logic)
    # If markets are different (e.g., Saudi vs Global) and neither is "General Arabic" acting as fallback
    if w_mkt != l_mkt and "General" not in w_mkt and "General" not in l_mkt:
         return "Wrong Market Target", "🌍", "Hreflang / Geo-Targeting Fix"

    # 2. Brand Dominance
    if any(b.lower() in query.lower() for b in brands):
        return "Brand Dominance", "🟢", "Monitor"

    # 3. SERP Dominance
    if w_pos <= Config.DOMINANCE_TOP and l_pos <= Config.DOMINANCE_SEC:
        return "SERP Dominance", "🟢", "Monitor"

    # 4. Intent Conflict
    w_int = row['Winner_Intent']
    l_int = row['Loser_Intent']
    if w_int != "Ambiguous" and l_int != "Ambiguous" and w_int != l_int:
        return "Intent Conflict", "🟠", "Content Split"

    # 5. Cannibalization
    if row['Overlap_Score'] > 0.6: 
        return "Critical Cannibalization", "🔴", "Merge / 301"
    
    return "Moderate Cannibalization", "🟡", "Review"

def analyze_data(df_raw, brands):
    df = df_raw.copy()
    # Clean URLs
    df['page_clean'] = df['page'].apply(lambda x: str(x).split('?')[0].split('#')[0].rstrip('/'))
    
    # Aggregate
    df_agg = df.groupby(['query', 'page_clean']).agg({
        'clicks': 'sum', 'impressions': 'sum', 'ctr': 'mean', 'position': 'mean'
    }).reset_index()
    
    df_agg = df_agg[df_agg['impressions'] >= Config.MIN_IMPS]
    
    # Find Duplicates
    q_counts = df_agg['query'].value_counts()
    cannibal_qs = q_counts[q_counts > 1].index.tolist()
    
    if not cannibal_qs: return pd.DataFrame()
    
    df_c = df_agg[df_agg['query'].isin(cannibal_qs)]
    results = []
    check_queue = []
    
    for query, group in df_c.groupby('query'):
        group = group.sort_values(['clicks', 'impressions'], ascending=[False, False])
        winner = group.iloc[0]
        losers = group.iloc[1:]
        
        w_mkt = detect_market(winner['page_clean'], query)
        w_int = get_page_intent(winner['page_clean'], query)
        
        for _, loser in losers.iterrows():
            l_mkt = detect_market(loser['page_clean'], query)
            l_int = get_page_intent(loser['page_clean'], query)
            
            overlap = (loser['impressions']/winner['impressions']) if winner['impressions'] > 0 else 0
            loss = int(loser['impressions'] * winner['ctr'])
            
            row_dat = {
                'Query': query,
                'Winner_Page': winner['page_clean'], 'Winner_Pos': round(winner['position'],1),
                'Winner_Market': w_mkt, 'Winner_Intent': w_int,
                'Loser_Page': loser['page_clean'], 'Loser_Pos': round(loser['position'],1),
                'Loser_Market': l_mkt, 'Loser_Intent': l_int,
                'Overlap_Score': overlap, 'Traffic_Loss': loss
            }
            
            status, icon, action = classify_issue(row_dat, brands)
            
            # Queue for live check if issue is actionable
            needs_check = "Critical" in status or "Intent" in status or "Market" in status
            if needs_check: check_queue.append(loser['page_clean'])
            
            row_dat.update({'Status': status, 'Icon': icon, 'Action': action, 'Check_Live': needs_check})
            results.append(row_dat)

    # Parallel Live Check
    unique_urls = list(set(check_queue))
    status_map = {}
    if unique_urls:
        st.toast(f"🕵️ Checking {len(unique_urls)} URLs for Redirects...", icon="⚡")
        with ThreadPoolExecutor(max_workers=12) as ex:
            future_map = {ex.submit(get_live_status, u): u for u in unique_urls}
            for fut in future_map:
                try:
                    u = future_map[fut]
                    code, is_red, final = fut.result()
                    status_map[u] = {'code': code, 'redirected': is_red}
                except:
                    status_map[u] = {'code': 0, 'redirected': False}

    # Final Filter
    final_res = []
    for r in results:
        u = r['Loser_Page']
        if r['Check_Live'] and u in status_map:
            d = status_map[u]
            if d['redirected'] or str(d['code']).startswith('3'):
                r['Status'] = "Resolved (Redirected)"
                r['Icon'] = "✅"
                r['Action'] = "Fixed"
                r['Traffic_Loss'] = 0 # No real loss if redirected
            elif str(d['code']) in ['404', '410']:
                r['Status'] = "Resolved (Gone)"
                r['Icon'] = "🗑️"
                r['Action'] = "None"
                r['Traffic_Loss'] = 0
        final_res.append(r)
        
    return pd.DataFrame(final_res).sort_values('Traffic_Loss', ascending=False)

# ==========================================
# 🔌 6. GSC AUTH & FETCH (FIXED DATES)
# ==========================================
@st.cache_resource
def auth_gsc(code):
    try:
        flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", ['https://www.googleapis.com/auth/webmasters.readonly'])
        flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
        flow.fetch_token(code=code)
        return build('searchconsole', 'v1', credentials=flow.credentials)
    except Exception as e: return None

def fetch_data(srv, site, days):
    # FIXED DATE LOGIC: Display EXACT dates to user
    lag = 3
    end = datetime.date.today() - datetime.timedelta(days=lag)
    start = end - datetime.timedelta(days=days)
    
    st.info(f"📅 جاري تحليل البيانات للفترة من: **{start}** إلى **{end}** (مدة {days} يوم فعلي)", icon="📆")
    
    req = {'startDate': start.isoformat(), 'endDate': end.isoformat(), 'dimensions': ['query', 'page'], 'rowLimit': 25000}
    try:
        resp = srv.searchanalytics().query(siteUrl=site, body=req).execute()
        rows = resp.get('rows', [])
        if not rows: return pd.DataFrame()
        return pd.DataFrame([{
            'query': r['keys'][0], 'page': r['keys'][1], 
            'clicks': r['clicks'], 'impressions': r['impressions'], 
            'ctr': r['ctr'], 'position': r['position']
        } for r in rows])
    except Exception as e:
        st.error(f"API Error: {e}"); return pd.DataFrame()

# ==========================================
# 🖥️ 7. APP UI
# ==========================================
with st.sidebar:
    st.header("⚙️ الإعدادات")
    u_file = st.file_uploader("ملف JSON", type="json")
    
    if 'creds' in st.session_state:
        st.success("✅ متصل")
        if 'sites' not in st.session_state:
            try: st.session_state.sites = [s['siteUrl'] for s in st.session_state.creds.sites().list().execute().get('siteEntry', [])]
            except: st.session_state.sites = []
        sel_site = st.selectbox("الموقع", st.session_state.sites) if st.session_state.sites else st.text_input("رابط الموقع")
    else:
        sel_site = st.text_input("رابط الموقع")

    # Fixed Slider
    days_val = st.slider("فترة التحليل (أيام)", 7, 90, 28)
    
    st.markdown("---")
    br_txt = st.text_area("براندات (للاستبعاد)", "almaster, المستر, ماستر")
    brands = [b.strip() for b in br_txt.split(',')]

if u_file and 'creds' not in st.session_state:
    with open("client_secret.json", "wb") as f: f.write(u_file.getbuffer())
    flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", ['https://www.googleapis.com/auth/webmasters.readonly'])
    flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
    st.markdown(f"[🔗 رابط المصادقة]({flow.authorization_url()[0]})")
    code = st.text_input("كود المصادقة:")
    if code:
        srv = auth_gsc(code)
        if srv: st.session_state.creds = srv; st.rerun()

if st.button("🚀 تشغيل التحليل (Geo-Aware Scan)", type="primary"):
    if 'creds' in st.session_state:
        with st.spinner("جاري سحب البيانات وتحليل الأسواق والروابط..."):
            raw = fetch_data(st.session_state.creds, sel_site, days_val)
            if not raw.empty:
                st.session_state.rep = analyze_data(raw, brands)
            else:
                st.error("لا توجد بيانات متاحة لهذه الفترة.")
    else: st.warning("يجب تسجيل الدخول.")

if 'rep' in st.session_state and not st.session_state.rep.empty:
    df = st.session_state.rep
    act = df[~df['Status'].str.contains('Resolved|Dominance')]
    geo_err = act[act['Status'].str.contains('Market')]
    crit = act[act['Status'].str.contains('Critical')]
    res = df[df['Status'].str.contains('Resolved')]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🌍 مشاكل استهداف (Geo)", len(geo_err))
    c2.metric("🔴 تضارب محتوى", len(crit))
    c3.metric("🧹 تم حله (Redirects)", len(res))
    c4.metric("📉 زيارات مهددة", f"{act['Traffic_Loss'].sum():,}")
    
    st.markdown("---")
    
    t1, t2, t3 = st.tabs(["⚠️ مشاكل الاستهداف & التضارب", "✅ تم حله / هيمنة", "📊 الداتا الكاملة"])
    
    with t1:
        st.caption("يعرض هذا الجدول الصفحات التي تنافس بعضها (محتوى مكرر) أو صفحات تستهدف أسواق خاطئة (مثلاً صفحة عامة تنافس صفحة سعودية).")
        st.dataframe(
            act[['Icon', 'Query', 'Status', 'Winner_Market', 'Winner_Page', 'Loser_Market', 'Loser_Page', 'Traffic_Loss']],
            column_config={
                "Winner_Page": st.column_config.LinkColumn("الرابح"),
                "Loser_Page": st.column_config.LinkColumn("الخاسر"),
                "Traffic_Loss": st.column_config.ProgressColumn("خسارة متوقعة", format="%d", max_value=int(df['Traffic_Loss'].max()))
            }, use_container_width=True)
            
    with t2:
        good = df[df['Status'].str.contains('Resolved|Dominance')]
        st.dataframe(good[['Icon', 'Query', 'Status', 'Winner_Page', 'Loser_Page']], use_container_width=True)
        
    with t3:
        st.dataframe(df, use_container_width=True)

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine='xlsxwriter') as w:
        df.to_excel(w, sheet_name='All Data', index=False)
        act.to_excel(w, sheet_name='Action Plan', index=False)
        geo_err.to_excel(w, sheet_name='Geo Conflicts', index=False)
    
    st.download_button("📥 تحميل التقرير (Excel)", out.getvalue(), f"SEO_Audit_v5.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

elif 'rep' in st.session_state:
    st.success("الموقع سليم تماماً! 🦁")
