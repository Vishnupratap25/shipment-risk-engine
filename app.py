import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
import plotly.express as px
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import shap

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Intelligent Shipment Prioritization Engine",
    layout="wide"
)

# ==============================
# PREMIUM DARK CORPORATE STYLING (FEDEX THEME)
# ==============================
st.markdown("""
<style>
/* 1. GLOBAL LAYOUT */
[data-testid="stAppViewContainer"] { 
    background: radial-gradient(circle at top right, rgba(77, 20, 140, 0.15) 0%, #1F2937 40%),
                radial-gradient(circle at bottom left, rgba(255, 98, 0, 0.08) 0%, #1F2937 40%);
    background-color: #1F2937; 
}
[data-testid="stHeader"] { 
    background-color: transparent !important; 
}
[data-testid="stSidebar"] { 
    background-color: rgba(22, 27, 34, 0.5) !important; 
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255, 255, 255, 0.05); 
}

/* 2. CUSTOM SCROLLBARS */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-thumb { 
    background: rgba(123, 97, 255, 0.2); 
    border-radius: 10px; 
}
::-webkit-scrollbar-thumb:hover { background: rgba(255, 98, 0, 0.5); }

/* 3. SKELETON LOADERS */
@keyframes skeleton-pulse {
    0% { background-color: rgba(22, 27, 34, 0.4); }
    50% { background-color: rgba(48, 54, 61, 0.6); }
    100% { background-color: rgba(22, 27, 34, 0.4); }
}
.skeleton-box {
    height: 100px; margin-bottom: 12px; border-radius: 12px;
    animation: skeleton-pulse 1.5s infinite ease-in-out;
}

/* 4. TYPOGRAPHY */
[data-testid="stAppViewContainer"] :is(p, span, label, li, h1, h2, h3, h4, h5, h6, [data-testid="stMarkdownContainer"] p) {
    color: #E5E7EB !important;
}
[data-testid="stAppViewContainer"] :is(h1, h2, h3, h4, h5, h6) {
    color: #FFFFFF !important;
    font-weight: 700;
}
[data-testid="stMetricValue"] div, [data-testid="stMetricLabel"] div {
    color: #FFFFFF !important;
}

/* 5. METRIC CARDS */
[data-testid="metric-container"], [data-testid="stMetric"] { 
    background-color: rgba(22, 27, 34, 0.4) !important; 
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 12px; 
    padding: 15px 20px; 
    border: 1px solid rgba(255, 255, 255, 0.08); 
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3); 
    border-left: 4px solid #4D148C; 
    transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
}
[data-testid="metric-container"]:hover, [data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px 0 rgba(77, 20, 140, 0.4);
}

/* 6. TABS NAVIGATION */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab"] { 
    font-size: 16px; color: #8B949E !important; 
    background-color: #161B22 !important; 
    border: 1px solid #30363D !important;
    border-bottom: none !important;
    border-radius: 8px 8px 0px 0px !important;
    padding: 10px 24px !important;
}
.stTabs [aria-selected="true"] { 
    color: #FFFFFF !important; 
    background-color: #4D148C !important; 
    border-bottom: 3px solid #FF6200 !important;
}

/* 7. FILE UPLOADER */
[data-testid="stFileUploader"] { 
    background-color: rgba(22, 27, 34, 0.4) !important;
    backdrop-filter: blur(16px); 
    border-radius: 12px; 
    border: 1px solid rgba(255, 255, 255, 0.08); 
    padding: 10px;
    transition: all 0.3s ease;
}
[data-testid="stFileUploader"]:hover, [data-testid="stFileUploader"]:has([data-testid="stFileUploaderFileData"]) {
    border-color: #FF6200 !important;
    box-shadow: 0 0 20px rgba(255, 98, 0, 0.2);
}
[data-testid="stFileUploader"] section {
    background-color: transparent !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
    background-color: rgba(255, 255, 255, 0.03) !important;
    border: 2px dashed rgba(77, 20, 140, 0.4) !important;
    transition: all 0.3s ease;
}
[data-testid="stFileUploader"]:hover [data-testid="stFileUploaderDropzone"], [data-testid="stFileUploader"]:has([data-testid="stFileUploaderFileData"]) [data-testid="stFileUploaderDropzone"] {
    border-color: #FF6200 !important;
    background-color: rgba(255, 98, 0, 0.05) !important;
}
[data-testid="stFileUploader"] label {
    color: #E5E7EB !important;
}
[data-testid="stFileUploader"] * { 
    color: #E5E7EB !important; 
}
[data-testid="stFileUploader"] button {
    background-color: #4D148C !important;
    color: white !important;
    border-radius: 8px !important;
    transition: background-color 0.3s ease;
}
[data-testid="stFileUploader"] button:hover {
    background-color: #FF6200 !important;
}
[data-testid="stFileUploaderDropzone"] div div {
    color: #9CA3AF !important;
}

/* 8. BUTTONS & BRANDING */
button[kind="primary"], button[kind="secondary"] { 
    background-color: #4D148C !important; color: #FFFFFF !important; 
    border-radius: 8px !important; border: none !important;
}
button[kind="primary"]:hover { background-color: #FF6200 !important; }

.logo-container { 
    background-color: #FFFFFF; padding: 15px 20px; border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.2); display: inline-block;
}

/* 9. INPUTS, POPOVERS & MENUS */
div[data-baseweb="select"] > div { 
    background-color: #161B22 !important; 
    color: #FFFFFF !important; 
    border: 1px solid #30363D !important;
}
div[data-baseweb="select"] input { color: #FFFFFF !important; }

[data-baseweb="popover"], [role="listbox"], [role="option"], [data-baseweb="tooltip"], [role="menu"], [role="dialog"], [data-baseweb="menu"], [data-baseweb="popover"] > div {
    background-color: #FFFFFF !important;
    color: #1F2937 !important;
    border: 1px solid #E5E7EB !important;
    box-shadow: 0 10px 25px rgba(0,0,0,0.1) !important;
}
[data-baseweb="popover"] *, [data-baseweb="tooltip"] *, [role="menu"] *, [role="dialog"] *, [data-baseweb="menu"] * {
    color: #1F2937 !important;
}
[role="menuitem"]:hover, [role="option"]:hover, [data-baseweb="option"]:hover {
    background-color: #F3F4F6 !important;
    color: #1F2937 !important;
}
div.stSelect > div {
    background-color: #FFFFFF !important;
    color: #1F2937 !important;
}
/* 10. EXPANDER (DROPDOWN) STYLING */
div[data-testid="stExpander"] {
    background-color: rgba(22, 27, 34, 0.2) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
}
div[data-testid="stExpander"] details summary {
    background-color: rgba(255, 255, 255, 0.03) !important;
    color: #FFFFFF !important;
    padding: 10px 15px !important;
    border-radius: 12px !important;
}
div[data-testid="stExpander"] details summary:hover {
    background-color: rgba(255, 255, 255, 0.08) !important;
    color: #FF6200 !important; /* Highlights FedEx Orange on hover */
}
div[data-testid="stExpander"] details summary svg {
    fill: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)

col_logo, col_title = st.columns([1.5, 6])
with col_logo:
    st.markdown('<div class="logo-container"><img src="https://upload.wikimedia.org/wikipedia/commons/3/33/FedEx_Wordmark.svg" width="150"></div>', unsafe_allow_html=True)
with col_title:
    st.title("Intelligent Shipment Prioritization Engine")
    st.caption("AI-Powered SLA Breach Risk Dashboard")

# ==============================
# HUB CODE → CITY MAPPING
# ==============================
HUB_TO_CITY = {
    "BLRA": "BANGALORE", "NDCA": "DELHI", "HYDBG": "HYDERABAD", "MAATS": "CHENNAI"
}
CITY_COORDS = {
"DELHI": (28.7041,77.1025), "MUMBAI": (19.0760,72.8777), "CHENNAI": (13.0827,80.2707),
"BANGALORE": (12.9716, 77.5946), "HYDERABAD": (17.3850,78.4867), "KOLKATA": (22.5726,88.3639),
"JAIPUR": (26.9124,75.7873), "LUCKNOW": (26.8467,80.9462), "PATNA": (25.5941,85.1376),
"BHOPAL": (23.2599,77.4126), "RAIPUR": (21.2514,81.6296), "RANCHI": (23.3441,85.3096),
"DEHRADUN": (30.3165,78.0322), "SHIMLA": (31.1048,77.1734), "GANDHINAGAR": (23.2156,72.6369),
"CHANDIGARH": (30.7333,76.7794), "PUNE": (18.5204,73.8567), "AHMEDABAD": (23.0225,72.5714),
"SURAT": (21.1702,72.8311), "NAGPUR": (21.1458,79.0882), "INDORE": (22.7196,75.8577),
"VADODARA": (22.3072,73.1812), "VISAKHAPATNAM": (17.6868,83.2185), "COIMBATORE": (11.0168,76.9558),
"MADURAI": (9.9252,78.1198), "TIRUCHIRAPPALLI": (10.7905,78.7047), "KOCHI": (9.9312,76.2673),
"NOIDA": (28.5355,77.3910), "GURGAON": (28.4595,77.0266), "GHAZIABAD": (28.6692,77.4538),
"KANPUR": (26.4499,80.3319), "VARANASI": (25.3176,82.9739), "AGRA": (27.1767,78.0081),
"MEERUT": (28.9845,77.7064), "MYSORE": (12.2958,76.6394), "MANGALORE": (12.9141,74.8560),
"HUBLI": (15.3647,75.1240), "VIJAYAWADA": (16.5062,80.6480), "GUNTUR": (16.3067,80.4365),
"GUWAHATI": (26.1445,91.7362), "SILIGURI": (26.7271,88.3953), "THANE": (19.2183,72.9781),
"NASHIK": (19.9975,73.7898), "AURANGABAD": (19.8762,75.3433), "KOLHAPUR": (16.7050,74.2433)
}

# ==============================
# LOAD OUR ENSEMBLE MODEL
# ==============================
@st.cache_resource
def load_artifacts():
    MODEL_PATH = "shipment_model.pkl"
    if not os.path.exists(MODEL_PATH):
        st.error(f"Engine Alert: Expected native ensemble model '{MODEL_PATH}' was not found. Please retrain.")
        st.stop()
    with open(MODEL_PATH, 'rb') as f:
        bundle = pickle.load(f)
    if not isinstance(bundle, dict):
        return {'model': bundle, 'threshold': 0.50, 'label_encoders': {}, 'expected_cols': [], 'metrics': {}}
    return bundle

# ==============================
# ORIGINAL INFERENCE PIPELINE
# ==============================
def process_data(df_raw, bundle):
    df = df_raw.copy()

    # Align columns safely
    header_map = {c.lower().strip(): c for c in df.columns}
    ist_col = header_map.get("ist_svc_commit_tmstp")

    if ist_col and ist_col in df.columns:
        df["IST_svc_commit_tmstp"] = pd.to_datetime(df[ist_col], errors="coerce")
        df["commit_hour"] = df["IST_svc_commit_tmstp"].dt.hour
        df["commit_day"] = df["IST_svc_commit_tmstp"].dt.hour
        df["commit_month"] = df["IST_svc_commit_tmstp"].dt.month
        df["commit_weekday"] = df["IST_svc_commit_tmstp"].dt.weekday
        df["is_weekend"] = df["commit_weekday"].apply(lambda x: 1 if x in [5, 6] else 0)

        def hour_bucket(h):
            if pd.isna(h): return "Unknown"
            elif 6 <= h < 12: return "Morning"
            elif 12 <= h < 18: return "Afternoon"
            elif 18 <= h < 24: return "Evening"
            else: return "Night"
            
        df["commit_hour_bucket"] = df["commit_hour"].apply(hour_bucket)

    bso_col = header_map.get('bso_cd', 'bso_cd')
    if bso_col in df.columns and "commit_weekday" in df.columns:
        df["Station_Day_Cross"] = df[bso_col].astype(str) + "_" + df["commit_weekday"].astype(str)
        
    pstl_col = header_map.get('recp_pstl_cd', 'recp_pstl_cd')
    if pstl_col in df.columns:
        df["postal_zone"] = df[pstl_col].astype(str).str[:3]
        
    # SHP_PCE_QTY is now excluded from the model brain to prevent overfitting on data errors.
    # We keep the raw data in df_raw for UI display purposes as requested.

    scan_col = header_map.get('last_scan', 'last_scan')
    if scan_col in df.columns:
        def group_scan(s):
            s = str(s).lower()
            if 'clearance' in s or 'customs' in s: return 'Customs/Clearance'
            if 'delay' in s or 'exception' in s: return 'Delay/Exception'
            if 'departed' in s or 'arrived' in s or 'transit' in s: return 'In Transit'
            if 'delivery' in s or 'delivered' in s: return 'Delivery Phase'
            if 'facility' in s or 'hub' in s: return 'Facility Operations'
            return 'Other/Unknown'
        df['last_scan_category'] = df[scan_col].apply(group_scan)

    df = df.fillna("Unknown")

    # Target Mapping Instead of Brittle LabelEncoding
    target_encoders = bundle.get('target_encoders', {})
    for col, enc_data in target_encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(enc_data['map']).fillna(enc_data['default'])

    return df

def categorize_risk(p):
    if p>=60: return "High Risk"
    elif p>=30: return "Medium Risk"
    else: return "Low Risk"

@st.cache_data
def run_model_prediction(df):
    bundle = load_artifacts()
    X = process_data(df, bundle)
    model = bundle.get('model')
    
    # Strip & lower headers to guarantee perfect model inference injection
    X.columns = [str(c).lower().strip() for c in X.columns]

    # Filter only expected columns for inference
    expected_cols = bundle.get('expected_cols', [])
    for col in expected_cols:
        if col not in X.columns:
            X[col] = 0
            
    X_pred = X[expected_cols].copy()
    X_pred = X_pred.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Class 1 is Target 1 (Breached per user's logic)
    probs_failure = model.predict_proba(X_pred)[:, 1]
    return probs_failure

# ==============================
# ROBUST DATA LOADER
# ==============================
def load_file(file, name):
    try:
        if name.endswith('.xlsx'): return pd.read_excel(file)
        delimiter = '\t' if name.endswith('.tsv') else ','
        try:
            df = pd.read_csv(file, sep=delimiter, encoding='utf-8', low_memory=False, on_bad_lines='skip')
            if delimiter == ',' and df.shape[1] <= 2:
                file.seek(0)
                df = pd.read_csv(file, sep='\t', encoding='utf-8', low_memory=False, on_bad_lines='skip')
            return df
        except (UnicodeDecodeError, pd.errors.ParserError):
            file.seek(0)
            try:
                df = pd.read_csv(file, sep=delimiter, encoding='ISO-8859-1', low_memory=False, on_bad_lines='skip')
                if delimiter == ',' and df.shape[1] <= 2:
                    file.seek(0)
                    df = pd.read_csv(file, sep='\t', encoding='ISO-8859-1', low_memory=False, on_bad_lines='skip')
                return df
            except pd.errors.ParserError:
                file.seek(0)
                other_delim = '\t' if delimiter == ',' else ','
                return pd.read_csv(file, sep=other_delim, encoding='ISO-8859-1', low_memory=False, on_bad_lines='skip')
    except Exception as e:
        raise e

# ==============================
# FRONTEND EXECUTION START
# ==============================
st.markdown("### 📥 Secure Data Upload Portal")
uploaded_file = st.file_uploader("Upload Operational Shipment File (CSV / Excel / TSV)", type=["csv", "xlsx", "tsv"])

if not uploaded_file:
    st.markdown("""
    <style>
    .hero-empty-state {
        text-align: center;
        padding: 60px 20px;
        background: rgba(22, 27, 34, 0.4);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        margin-top: 20px;
        position: relative;
        overflow: hidden;
    }
    .hero-empty-state::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%; width: 200%; height: 200%;
        background: radial-gradient(circle at center, rgba(77,20,140,0.15) 0%, transparent 60%);
        pointer-events: none;
    }
    .hero-empty-state svg {
        width: 100px;
        height: 100px;
        fill: #FF6200;
        margin-bottom: 20px;
        filter: drop-shadow(0 0 15px rgba(255,98,0,0.5));
        animation: float 4s ease-in-out infinite;
    }
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-15px); }
        100% { transform: translateY(0px); }
    }
    .hero-title {
        font-size: 32px;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF6200, #9A5BFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 12px;
        letter-spacing: 0.5px;
    }
    .hero-subtitle {
        font-size: 16px;
        color: #A0AEC0;
        font-weight: 400;
        max-width: 550px;
        margin: 0 auto;
        line-height: 1.6;
    }
    </style>
    
    <div class="hero-empty-state">
        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path d="M21 16v-2l-8-5V3.5c0-.83-.67-1.5-1.5-1.5S10 2.67 10 3.5V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5l8 2.5z"/>
        </svg>
        <div class="hero-title">Last Location Scan Intelligence</div>
        <div class="hero-subtitle">Upload your <b>Last Location Scan Report-IB (INDIA)</b> payload. The intelligent risk engine will instantly evaluate regional SLA breach risks across the Indian logistics network and prioritize critical shipments for immediate tactical action.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if uploaded_file:
    if "uploaded_filename" not in st.session_state or st.session_state.uploaded_filename != uploaded_file.name:
        with st.spinner("AI securely indexing logistics pipeline and evaluating mathematical risks..."):
            try:
                df = load_file(uploaded_file, uploaded_file.name)
            except Exception as e:
                st.error(f"Error loading file format securely: {e}")
                st.stop()

            if "Last Scan Loc" in df.columns: scan_col = "Last Scan Loc"
            elif "last_scan" in df.columns: scan_col = "last_scan"
            else: scan_col = None

            if scan_col:
                df[scan_col] = df[scan_col].astype(str).str.upper().str.strip()
                ramp = ["BOMRT","DELRT","BLRRT"]
                clearance = ["BOMHV","BOMIP","PNQIP","AMDIP","DELHV","DELIP","JAIIP","CCUIP","BLRIP","BLRHV","HYDIP","COKIP","MAAIP","CJBIP"]
                hub = ["BOMA","DELGW","BLRGW"]

                def classify_location(loc):
                    if loc in ramp: return "RAMP"
                    elif loc in clearance: return "CLEARANCE"
                    elif loc in hub: return "HUB"
                    else: return "STATION"

                df["Location_Type"] = df[scan_col].apply(classify_location)

                def classify_region(loc):
                    if not isinstance(loc, str): return "UNKNOWN"
                    loc = loc.upper().strip()
                    code = loc[:3]
                    west = ["BOM","PNQ","AMD","BDQ","NAG","SUR","RAJ","JAI"]
                    south = ["BLR","HYD","MAA","COK","CJB","TIR","PNY","MYS","IXM","TRV","CCJ"]
                    north = ["DEL","LKO","JDH","PAT","IXA","IXB","GAU","CCU","RPR","RNC","VNS","AGR","GWL","KNU"]
                    if code in west: return "WEST"
                    elif code in south: return "SOUTH"
                    elif code in north: return "NORTH"
                    else: return "INTERNATIONAL"

                df["Region"] = None
                try:
                    station_df = pd.read_excel("loc region.xlsx")
                    station_df["dest_loc_cd"] = station_df["dest_loc_cd"].astype(str).str.upper().str.strip()
                    station_region_map = station_df.set_index("dest_loc_cd")["Region"].str.upper().to_dict()
                    df["Region"] = df[scan_col].map(station_region_map)
                except Exception:
                    pass

                mask_na = df["Region"].isna()
                df.loc[mask_na, "Region"] = df.loc[mask_na, scan_col].apply(classify_region)
                df["Region"] = df["Region"].fillna("UNKNOWN").str.upper()

            if "Trk Nos" in df.columns:
                df["Trk Nos"] = df["Trk Nos"].astype(str).str.strip()
                nan_equivalents = ["nan", "none", "", "na", "<na>"]
                df = df[~df["Trk Nos"].str.lower().isin(nan_equivalents)]
                df = df[~df["Trk Nos"].str.lower().str.startswith("applied filters")]

            for col in df.columns:
                if "city" in col.lower() or "loc" in col.lower():
                    df[col] = df[col].replace(HUB_TO_CITY)

            probs = run_model_prediction(df)
            df["Failure_Risk_%"] = (probs * 100).round(2)
            df["Risk_Category"] = df["Failure_Risk_%"].apply(categorize_risk)
            
            st.session_state.master_df = df.copy()
            st.session_state.uploaded_filename = uploaded_file.name

    filtered_df = st.session_state.master_df.copy()
    bundle = load_artifacts()
    threshold = bundle.get('threshold', 0.50)
    model = bundle.get('model')
    st.success("Model Loaded: Hybrid Voting Ensembled AI (Shipment Structure)")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Risk Analysis", "⏳ NSL Monitor", "🔮 Predict Single Shipment"])

    # ==============================
    # TAB 1: OVERVIEW
    # ==============================
    with tab1:
        total = len(filtered_df)
        high_count = (filtered_df["Risk_Category"] == "High Risk").sum()
        medium_count = (filtered_df["Risk_Category"] == "Medium Risk").sum()
        low_count = (filtered_df["Risk_Category"] == "Low Risk").sum()
        high_pct = round((high_count / max(total, 1)) * 100, 2)
        medium_pct = round((medium_count / max(total, 1)) * 100, 2)
        low_pct = round((low_count / max(total, 1)) * 100, 2)
        expected_breaches = (filtered_df["Failure_Risk_%"] >= threshold * 100).sum()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Shipments", f"{total:,}")
        col2.metric("High Risk %", f"{high_pct}%")
        col3.metric("Medium Risk %", f"{medium_pct}%")
        col4.metric("Low Risk %", f"{low_pct}%")

        st.subheader("📦 Risk Category Breakdown")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔴 High Risk", f"{high_count:,}")
        c2.metric("🟠 Medium Risk", f"{medium_count:,}")
        c3.metric("🟢 Low Risk", f"{low_count:,}")
        c4.metric("🔮 Predicted Breaches", f"{expected_breaches:,}")

        st.divider()
        st.subheader("📍 Complete City-Level Risk Dashboard")

        city_cols = [c for c in filtered_df.columns if "city" in c.lower()]
        if city_cols:
            city_col = city_cols[0]
            city_summary = (
                filtered_df.groupby(city_col).agg(
                    Total_Shipments=("Risk_Category", "count"),
                    High_Risk_Shipments=("Risk_Category", lambda x: (x == "High Risk").sum()),
                    Medium_Risk_Shipments=("Risk_Category", lambda x: (x == "Medium Risk").sum()),
                    Low_Risk_Shipments=("Risk_Category", lambda x: (x == "Low Risk").sum())
                ).reset_index()
            )

            colA, colB = st.columns(2)
            with colA: sort_col = st.selectbox("Sort By", [city_col, "Total_Shipments", "High_Risk_Shipments", "Medium_Risk_Shipments", "Low_Risk_Shipments"])
            with colB: sort_order = st.selectbox("Order", ["Descending", "Ascending"])
            ascending = True if sort_order == "Ascending" else False
            city_summary = city_summary.sort_values(by=sort_col, ascending=ascending)
            st.dataframe(city_summary, use_container_width=True)

            st.divider()
            st.subheader("🔎 Search City Shipment Risk")
            city_list = ["Select City"] + sorted(city_summary[city_col].dropna().unique())
            selected_city = st.selectbox("Select or Search City", city_list)

            if selected_city != "Select City":
                city_data = city_summary[city_summary[city_col] == selected_city]
                if not city_data.empty:
                    colX, colY, colZ, colW = st.columns(4)
                    colX.metric("📦 Total Shipments", int(city_data["Total_Shipments"].values[0]))
                    colY.metric("🔴 High Risk Shipments", int(city_data["High_Risk_Shipments"].values[0]))
                    colZ.metric("🟠 Medium Risk Shipments", int(city_data["Medium_Risk_Shipments"].values[0]))
                    colW.metric("🟢 Low Risk Shipments", int(city_data["Low_Risk_Shipments"].values[0]))

                    city_shipments = filtered_df[filtered_df[city_col] == selected_city]
                    st.divider()
                    st.subheader(f"📦 Shipment Details for {selected_city}")
                    if not city_shipments.empty:
                        city_shipments = city_shipments.sort_values(by="Failure_Risk_%", ascending=False)
                        st.dataframe(city_shipments, use_container_width=True)
                        st.download_button("⬇ Download Shipment Data", data=city_shipments.to_csv(index=False), file_name=f"{selected_city}_shipments.csv", mime="text/csv")

            st.divider()
            st.subheader("🔥 Top 10 Cities Risk Composition")
            city_summary_filtered = city_summary[city_summary["Total_Shipments"] >= 10]
            top_risk_cities = city_summary_filtered.sort_values("High_Risk_Shipments", ascending=False).head(10)
            
            # Prepare melted data for better label control (adding percentages)
            melt_cols = ["High_Risk_Shipments", "Medium_Risk_Shipments", "Low_Risk_Shipments"]
            df_melt = top_risk_cities.melt(id_vars=[city_col, "Total_Shipments"], value_vars=melt_cols, var_name="Risk_Status", value_name="Count")
            df_melt["Percentage"] = (df_melt["Count"] / df_melt["Total_Shipments"] * 100).round(1)
            df_melt["Label"] = df_melt.apply(lambda x: f"{int(x['Count']):,} ({x['Percentage']}%)" if x["Count"] > 0 else "", axis=1)

            fig_top = px.bar(df_melt, x=city_col, y="Count", color="Risk_Status", text="Label",
                             title="Top 10 Cities Shipment Risk Distribution",
                             color_discrete_map={"High_Risk_Shipments": "#EF4444", "Medium_Risk_Shipments": "#F59E0B", "Low_Risk_Shipments": "#22C55E"})
            
            fig_top.update_traces(textposition='inside')
            fig_top.update_layout(barmode="stack", xaxis_title="City", yaxis_title="Number of Shipments", template="plotly_dark")
            st.plotly_chart(fig_top, use_container_width=True)

            st.divider()
            st.subheader("🧠 AI Operational Insights")
            if not city_summary.empty:
                worst_city = city_summary.sort_values("High_Risk_Shipments", ascending=False).iloc[0]
                worst_city_name = worst_city[city_col]
                colA, colB = st.columns(2)
                with colA: st.info(f"🚨 **Highest Risk Region**\n\n**{worst_city_name}** has the highest number of high-risk shipments.\n\nSuggested action:\n• Monitor shipments closely\n• Prioritize hub processing")
                with colB: st.warning(f"⚠ **Operational Alert**\n\n**{worst_city_name}** currently has the highest number of high-risk shipments.\n\n📦 Total Shipments: **{int(worst_city['Total_Shipments'])}**\n🔴 High Risk: **{int(worst_city['High_Risk_Shipments'])}**")

            st.subheader("🗺 India Shipment Risk Heatmap")
            CITY_ALIAS = {"BENGALURU": "BANGALORE", "BANGALORE URBAN": "BANGALORE", "BANGALORE RURAL": "BANGALORE", "BLR": "BANGALORE", "DELHI NCR": "DELHI"}
            city_summary["city_upper"] = city_summary[city_col].astype(str).str.upper().str.strip().replace(CITY_ALIAS)
            city_summary_map = city_summary[city_summary["city_upper"].isin(CITY_COORDS.keys())]

            map_type = st.radio("Select Map View", ["City Risk Map", "Risk Hotspot Map"], horizontal=True)
            map_data = []
            for _, row in city_summary_map.iterrows():
                coords = CITY_COORDS.get(row["city_upper"])
                if coords:
                    map_data.append({"city": row["city_upper"], "lat": coords[0], "lon": coords[1], "risk": max(int(row["High_Risk_Shipments"]), 1), "bubble_size": 18, "Total_Shipments": int(row["Total_Shipments"]), "High_Risk_Shipments": int(row["High_Risk_Shipments"]), "Medium_Risk_Shipments": int(row["Medium_Risk_Shipments"]), "Low_Risk_Shipments": int(row["Low_Risk_Shipments"])})
            
            if map_data:
                map_df = pd.DataFrame(map_data)
                map_df["risk_log"] = np.log1p(map_df["risk"])
                
                fig_map = px.scatter_mapbox(map_df, lat="lat", lon="lon", size="bubble_size", size_max=18, color="risk_log", hover_name="city",
                                            hover_data={"lat": False, "lon": False, "risk": False, "risk_log": False, "bubble_size": False, "Total_Shipments": True, "High_Risk_Shipments": True, "Medium_Risk_Shipments": True, "Low_Risk_Shipments": True},
                                            color_continuous_scale=["#FFE5E5", "#FF9999", "#FF4D4D", "#E60000", "#990000"], zoom=4, height=700)
                fig_map.update_traces(marker=dict(opacity=0.95))
                fig_map.update_layout(mapbox_style="carto-positron", mapbox=dict(center=dict(lat=22.5, lon=78.9), zoom=4), margin={"r":0,"t":0,"l":0,"b":0})

                import plotly.graph_objects as go
                fig_hotspot = go.Figure(go.Densitymapbox(lat=map_df["lat"], lon=map_df["lon"], z=map_df["High_Risk_Shipments"], radius=40, colorscale="Reds", showscale=True))
                fig_hotspot.update_layout(mapbox_style="carto-positron", mapbox_zoom=4, mapbox_center={"lat": 22.5, "lon": 78.9}, height=700, margin={"r":0,"t":0,"l":0,"b":0})

                if map_type == "City Risk Map": st.plotly_chart(fig_map, use_container_width=True)
                else: st.plotly_chart(fig_hotspot, use_container_width=True)
            else:
                st.warning("No city coordinates matched.")

    # ==============================
    # TAB 2: RISK ANALYSIS
    # ==============================
    with tab2:
        st.subheader("🔥 Top 10 High Risk Shipments")
        st.dataframe(filtered_df.sort_values("Failure_Risk_%", ascending=False).head(10), use_container_width=True)
        st.subheader("📋 Shipment Preview")
        st.dataframe(filtered_df.head(100), use_container_width=True)
        st.download_button("⬇ Download Filtered Data", data=filtered_df.to_csv(index=False).encode("utf-8"), file_name=f"filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
        
        st.divider()
        st.subheader("📊 Model Performance")
        bundle = load_artifacts()
        _metrics = bundle.get('metrics', {})
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy",  f"{_metrics.get('accuracy', 0.0):.2f}", help="Total percentage of shipments the model categorized correctly.")
        c2.metric("Precision", f"{_metrics.get('precision', 0.0):.2f}", help="When the model flags a shipment as 'High Risk', this is how often it is actually correct.")
        c3.metric("Recall",    f"{_metrics.get('recall', 0.0):.2f}", help="Out of all the shipments that actually breached their SLA, this is how many the model successfully caught.")
        c4.metric("F1 Score",  f"{_metrics.get('f1', 0.0):.2f}", help="The structural balance rating grading the trade-off between Precision and Recall.")
        c5.metric("ROC AUC",   f"{_metrics.get('roc_auc', 0.0):.2f}", help="The absolute core capability of the AI engine to cleanly separate failures from safe deliveries.")
        
        d_size = _metrics.get('dataset_size', '186K')
        st.caption(f"Model performance based on training test dataset ({d_size} rows).")

    # ==============================
    # TAB 3: NSL MONITOR
    # ==============================
    with tab3:
        st.subheader("⏳ Live Commitment Breach Control Tower")

        st.markdown("""
        <style>
        div.stButton > button {
            width: 100%;
            border-radius: 10px;
            height: 48px;
            font-weight: 600;
            font-size: 15px;
        }
        </style>
        """, unsafe_allow_html=True)

        colA, colB = st.columns([1,5])

        with colA:
            if st.button("🔄 Refresh Now"):
                st.rerun()

        try:
            filtered_df["IST_svc_commit_tmstp"] = pd.to_datetime(
                filtered_df["IST_svc_commit_tmstp"], errors="coerce"
            )

            current_time = pd.Timestamp.now() + pd.Timedelta(hours=5, minutes=30)

            temp_td = filtered_df["IST_svc_commit_tmstp"] - current_time
            filtered_df["Remaining_Seconds"] = temp_td.dt.total_seconds()

            def global_format_time(x):
                if pd.isna(x):
                    return "N/A"
                sign = "-" if x < 0 else ""
                x = abs(int(x))
                hrs = x // 3600
                mins = (x % 3600) // 60
                secs = x % 60
                return f"{sign}{hrs:02}:{mins:02}:{secs:02}"

            filtered_df["Time_Remaining"] = filtered_df["Remaining_Seconds"].apply(global_format_time)

            sla_df = filtered_df.copy()
            filtered_df.drop(columns=["Remaining_Seconds"], inplace=True, errors="ignore")

            status_list = []

            for _, row in sla_df.iterrows():
                sec = row["Remaining_Seconds"]
                risk = row["Failure_Risk_%"]

                status_value = str(row.get("Status", "")).lower().strip()
                location_type = str(row.get("Location_Type", "")).lower()
                region = str(row.get("Region", "")).upper()

                if status_value == "ontime":
                    status_list.append("Delivered")
                elif status_value in ["commit_fail", "pod_commit_fail"]:
                    status_list.append("Breached")
                elif sec < 0:
                    status_list.append("Breached")
                elif pd.isna(sec):
                    status_list.append("Unknown")
                elif location_type == "ramp" and sec < 48 * 3600:
                    status_list.append("Critical")
                elif region == "INTERNATIONAL" and sec < 24 * 3600:
                    status_list.append("Critical")
                elif sec < 2 * 3600:
                    status_list.append("Critical")
                elif sec < 6 * 3600 and risk >= 60:
                    status_list.append("Critical")
                elif sec < 6 * 3600:
                    status_list.append("Warning")
                elif risk >= 70:
                    status_list.append("Warning")
                else:
                    status_list.append("Safe")

            sla_df["SLA_Status"] = status_list

            sla_df["SLA_Urgency"] = sla_df["Remaining_Seconds"].apply(
                lambda x: 1 if x < 0 else 1 / (x/3600 + 1)
            )

            sla_df["Priority_Score"] = sla_df["Failure_Risk_%"] * sla_df["SLA_Urgency"]
            sla_df = sla_df.sort_values("Priority_Score", ascending=False)

            breach_count = (sla_df["SLA_Status"]=="Breached").sum()
            critical_count = (sla_df["SLA_Status"]=="Critical").sum()
            warning_count = (sla_df["SLA_Status"]=="Warning").sum()
            safe_count = (sla_df["SLA_Status"]=="Safe").sum()
            delivered_count = (sla_df["SLA_Status"]=="Delivered").sum()

            if "status_filter_btn_key" not in st.session_state:
                st.session_state.status_filter_btn_key = ["Breached", "Critical", "Warning", "Safe", "Delivered"]
            if "active_qf" not in st.session_state:
                st.session_state.active_qf = "All"

            active_key = st.session_state.active_qf

            # on_click callback runs BEFORE page re-renders, so active state is always correct
            def set_status(val):
                st.session_state.active_qf = val
                if val == "All": st.session_state.status_filter_btn_key = ["Breached", "Critical", "Warning", "Safe", "Delivered"]
                else: st.session_state.status_filter_btn_key = [val]

            # Active and Pulsing button CSS
            pulse_css = f"""
            <style>
            button[kind="primary"] {{
                background-color: #4D148C !important;
                border: 2px solid #FF6200 !important;
                box-shadow: 0 0 14px rgba(255,98,0,0.7) !important;
                color: #FFFFFF !important;
                font-weight: 700 !important;
            }}
            
            @keyframes neon-pulse-red {{
                0% {{ box-shadow: 0 0 4px rgba(239, 68, 68, 0.4); border-color: rgba(239, 68, 68, 0.5); }}
                50% {{ box-shadow: 0 0 18px rgba(239, 68, 68, 0.9), 0 0 25px rgba(239, 68, 68, 0.6); border-color: #EF4444; }}
                100% {{ box-shadow: 0 0 4px rgba(239, 68, 68, 0.4); border-color: rgba(239, 68, 68, 0.5); }}
            }}

            @keyframes neon-pulse-orange {{
                0% {{ box-shadow: 0 0 4px rgba(245, 158, 11, 0.4); border-color: rgba(245, 158, 11, 0.5); }}
                50% {{ box-shadow: 0 0 18px rgba(245, 158, 11, 0.9), 0 0 25px rgba(245, 158, 11, 0.6); border-color: #F59E0B; }}
                100% {{ box-shadow: 0 0 4px rgba(245, 158, 11, 0.4); border-color: rgba(245, 158, 11, 0.5); }}
            }}
            """
            
            # Apply to the 1st button (Breached) only if there are breaches
            if breach_count > 0:
                pulse_css += """
                div[data-testid="stHorizontalBlock"]:has(> div[data-testid="column"]:nth-child(6):last-child) > div[data-testid="column"]:nth-child(1) button {
                    animation: neon-pulse-red 1.5s infinite;
                }
                """
                
            # Apply to the 2nd button (Critical) only if there are criticals
            if critical_count > 0:
                pulse_css += """
                div[data-testid="stHorizontalBlock"]:has(> div[data-testid="column"]:nth-child(6):last-child) > div[data-testid="column"]:nth-child(2) button {
                    animation: neon-pulse-orange 1.6s infinite;
                }
                """
                
            pulse_css += "</style>"
            st.markdown(pulse_css, unsafe_allow_html=True)

            st.markdown("##### 🖱 Quick-Filter by SLA Status")
            f_cols = st.columns(6)

            f_cols[0].button(f"🚨 {breach_count} Breached", use_container_width=True, type="primary" if active_key=="Breached" else "secondary", on_click=set_status, args=("Breached",))
            f_cols[1].button(f"🔴 {critical_count} Critical", use_container_width=True, type="primary" if active_key=="Critical" else "secondary", on_click=set_status, args=("Critical",))
            f_cols[2].button(f"🟠 {warning_count} Warning", use_container_width=True, type="primary" if active_key=="Warning" else "secondary", on_click=set_status, args=("Warning",))
            f_cols[3].button(f"🟢 {safe_count} Safe", use_container_width=True, type="primary" if active_key=="Safe" else "secondary", on_click=set_status, args=("Safe",))
            f_cols[4].button(f"📦 {delivered_count} Delivered", use_container_width=True, type="primary" if active_key=="Delivered" else "secondary", on_click=set_status, args=("Delivered",))
            f_cols[5].button("🌐 Show All", use_container_width=True, type="primary" if active_key=="All" else "secondary", on_click=set_status, args=("All",))

            st.divider()
            st.subheader("🎛 Shipment Filters")

            statuses = ["Breached","Critical","Warning","Safe","Delivered"]

            status_filter = st.multiselect(
                "Filter by Shipment Status",
                statuses,
                key="status_filter_btn_key"
            )

            if len(status_filter) > 0:
                sla_df = sla_df[sla_df["SLA_Status"].isin(status_filter)]

            col1, col2 = st.columns(2)

            with col1:
                if "Region" in sla_df.columns:
                    region_filter = st.multiselect(
                        "Filter by Region",
                        sorted(filtered_df["Region"].dropna().unique()),
                        default=sorted(sla_df["Region"].dropna().unique())
                    )
                else:
                    region_filter = []

            with col2:
                if "Location_Type" in sla_df.columns:
                    location_filter = st.multiselect(
                        "Filter by Location Type",
                        sorted(filtered_df["Location_Type"].dropna().unique()),
                        default=sorted(filtered_df["Location_Type"].dropna().unique())
                    )
                else:
                    location_filter = []

            if region_filter:
                sla_df = sla_df[sla_df["Region"].isin(region_filter)]

            if location_filter:
                sla_df = sla_df[sla_df["Location_Type"].isin(location_filter)]

            st.divider()
            st.subheader("📊 Shipment Operational Trend")

            if "Location_Type" in sla_df.columns:

                # Rename regions for better UI display and order
                region_map = {
                    "INTERNATIONAL": "Not Arrived", 
                    "OTHER": "Not Arrived", 
                    "NORTH": "North", 
                    "SOUTH": "South", 
                    "WEST": "West"
                }
                sla_df["Region"] = sla_df["Region"].replace(region_map)

                # Consolidate Not Arrived locations since they are uncertain
                sla_df.loc[sla_df["Region"] == "Not Arrived", "Location_Type"] = "Pending Arrival"

                trend_df = (
                    sla_df
                    .groupby(["Region","Location_Type","SLA_Status"])
                    .size()
                    .reset_index(name="Shipment_Count")
                )
                
                # Add percentages per (Region, Location_Type) stack
                group_totals = trend_df.groupby(["Region", "Location_Type"])["Shipment_Count"].transform("sum")
                trend_df["Percentage"] = (trend_df["Shipment_Count"] / group_totals * 100).round(1)
                trend_df["Label"] = trend_df.apply(lambda x: f"{int(x['Shipment_Count']):,} ({x['Percentage']}%)" if x["Shipment_Count"] > 0 else "", axis=1)

                fig = px.bar(
                    trend_df,
                    x="Location_Type",
                    y="Shipment_Count",
                    color="SLA_Status",
                    facet_col="Region",
                    text="Label",
                    category_orders={"Region": ["Not Arrived", "North", "West", "South"]},
                    barmode="stack",
                    title="Shipment Trend by Location Type and Region",
                    color_discrete_map={
                        "Breached": "#7F1D1D",
                        "Critical": "#EF4444",
                        "Warning": "#F59E0B",
                        "Safe": "#22C55E",
                        "Delivered": "#3B82F6"
                    }
                )
                
                fig.update_traces(textposition='inside', textfont_size=10)

                fig.update_layout(
                    xaxis_title="Location Type",
                    yaxis_title="Number of Shipments",
                    legend_title="Shipment Status",
                    template="plotly_dark"
                 )
                
                # Decouple X-axes to hide empty categories in each facet (especially Not Arrived)
                fig.update_xaxes(matches=None, showticklabels=True)

                # Remove "Region=" prefix from facet titles
                fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

                st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("📊 Full Shipment NSL Monitor")

            table_df = sla_df.copy()

            table_df["Remaining Time"] = table_df["Time_Remaining"]

            if "Last Scan Date Time" in table_df.columns:
                table_df["Last Scan Date Time"] = pd.to_datetime(
                    table_df["Last Scan Date Time"], errors="coerce"
                ).dt.strftime("%Y-%m-%d %H:%M:%S")

            cols = [
                "Trk Nos",
                "IST_svc_commit_tmstp",
                "Remaining Time",
                "Failure_Risk_%",
                "SLA_Status",
                "Dest Loc",
                "recp_pstl_cd",
                "last_scan",
                "Last Scan Date Time",
                "Last Scan Loc",
                "City name"
            ]

            cols = [c for c in cols if c in table_df.columns]

            display_table = table_df[cols].rename(
                columns={
                    "Trk Nos":"Shipment",
                    "Failure_Risk_%":"Risk %",
                    "SLA_Status":"Status"
                }
            )

            st.dataframe(display_table,use_container_width=True)

            st.write(f"Showing **{len(display_table)} shipments**")
            
            # --- New Logic Legend Section ---
            st.divider()
            with st.expander("ℹ️ **How are these Statuses Calculated? (System Logic Key)**", expanded=False):
                st.markdown("""
                <h3 style='margin-bottom: 20px;'>🧠 NSL Model Classification Logic</h3>
                <p style='color: #CBD5E1; margin-bottom: 25px;'>This dashboard uses an intelligent hybrid engine that combines <b>Real-Time Logistics Commitment (SLA)</b> with <b>AI-Powered Failure Risk %</b>.</p>
                
                <div style='display: flex; flex-direction: column; gap: 15px;'>
                    <div style='background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border-left: 5px solid #3B82F6;'>
                        <span style='font-size: 1.1rem; color: #60A5FA;'>🔵 Delivered</span>
                        <p style='margin: 5px 0 0 0; color: #E2E8F0; font-size: 0.95rem;'>Shipment has already reached its final destination (On-Time).</p>
                    </div>
                    <div style='background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border-left: 5px solid #F87171;'>
                        <span style='font-size: 1.1rem; color: #F87171;'>🔴 Breached</span>
                        <p style='margin: 5px 0 0 0; color: #E2E8F0; font-size: 0.95rem;'>Shipment has missed its commitment or the status is confirmed as 'commit_fail'.</p>
                    </div>
                    <div style='background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border-left: 5px solid #EF4444;'>
                        <span style='font-size: 1.1rem; color: #EF4444;'>💥 Critical</span>
                        <p style='margin: 5px 0 0 0; color: #E2E8F0; font-size: 0.95rem; line-height: 1.6;'>
                            <b>Immediate Tactical Action Required.</b><br>
                            • <b>Ramp Scan:</b> < 48 hours remaining<br>
                            • <b>International:</b> < 24 hours remaining<br>
                            • <b>Any:</b> < 2 hours remaining<br>
                            • <b>AI Risk:</b> < 6 hours remaining AND <b>Failure Risk ≥ 60%</b>
                        </p>
                    </div>
                    <div style='background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border-left: 5px solid #F59E0B;'>
                        <span style='font-size: 1.1rem; color: #F59E0B;'>🟠 Warning</span>
                        <p style='margin: 5px 0 0 0; color: #E2E8F0; font-size: 0.95rem;'>
                            <b>Pre-emptive Monitoring Required.</b><br>
                            • <b>Deadline:</b> < 6 hours remaining<br>
                            • <b>AI Risk:</b> <b>Failure Risk ≥ 70%</b> regardless of time remaining.
                        </p>
                    </div>
                    <div style='background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border-left: 5px solid #22C55E;'>
                        <span style='font-size: 1.1rem; color: #22C55E;'>🟢 Safe</span>
                        <p style='margin: 5px 0 0 0; color: #E2E8F0; font-size: 0.95rem;'>Shipment is currently on-track based on both time and AI risk evaluation.</p>
                    </div>
                </div>

                <h3 style='margin-top: 30px; margin-bottom: 20px;'>🤖 AI Model Risk Categories</h3>
                <p style='color: #CBD5E1; margin-bottom: 25px;'>The AI engine calculates a raw probability of failure (0-100%). These probabilities are grouped into the categories seen across the dashboard:</p>
                
                <div style='display: flex; flex-direction: column; gap: 15px;'>
                    <div style='background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border-left: 5px solid #EF4444;'>
                        <span style='font-size: 1.1rem; color: #EF4444;'>🔴 High Risk</span>
                        <p style='margin: 5px 0 0 0; color: #E2E8F0; font-size: 0.95rem;'><b>60% to 100%</b> probability of missing the service commitment.</p>
                    </div>
                    <div style='background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border-left: 5px solid #F59E0B;'>
                        <span style='font-size: 1.1rem; color: #F59E0B;'>🟠 Medium Risk</span>
                        <p style='margin: 5px 0 0 0; color: #E2E8F0; font-size: 0.95rem;'><b>30% to 60%</b> probability of missing the service commitment.</p>
                    </div>
                    <div style='background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border-left: 5px solid #22C55E;'>
                        <span style='font-size: 1.1rem; color: #22C55E;'>🟢 Low Risk</span>
                        <p style='margin: 5px 0 0 0; color: #E2E8F0; font-size: 0.95rem;'><b>0% to 30%</b> probability of missing the service commitment.</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br><p style='font-size: 0.8rem; color: #64748B;'>*Note: The final 'Status' (Critical vs Warning) takes precedence over these AI categories as it also considers time remaining.</p>", unsafe_allow_html=True)

        except Exception:
            st.info("SLA monitoring unavailable for this dataset.")

    # ==============================
    # TAB 4
    # ==============================

    with tab4:
        st.subheader("🔎 Predict Shipment Risk Using Tracking Number")

        tracking_no = st.text_input("Enter Tracking Number")

        if st.button("Fetch Shipment & Predict Risk"):
            shipment_row = filtered_df[filtered_df["Trk Nos"] == tracking_no]

            if shipment_row.empty:
                st.error("Tracking number not found")

            else:
                st.success("Shipment Found")

                st.subheader("📦 Shipment Details")

                st.dataframe(shipment_row, use_container_width=True)

                risk_pct = float(shipment_row["Failure_Risk_%"].values[0])
                category = categorize_risk(risk_pct)

                st.subheader("🚨 Prediction Result")

                col1, col2 = st.columns(2)

                if "Region" in sla_df.columns:
                    sla_df["Region"] = sla_df["Region"].replace({"INTERNATIONAL": "OTHER"})

                col1.metric("Failure Risk %", f"{risk_pct:.2f}%")
                col2.metric("Risk Category", category)

                confidence = int(abs(risk_pct - 50) * 2)

                st.subheader("📊 Model Confidence Level")

                st.progress(confidence if confidence <= 100 else 100)

                st.write(f"Confidence Score: {confidence}/100")

                try:
                    st.divider()
                    st.subheader("⏱ SLA Status")

                    if "IST_svc_commit_tmstp" in shipment_row.columns:
                        commit_time = pd.to_datetime(
                            shipment_row["IST_svc_commit_tmstp"].values[0],
                            errors="coerce"
                        )

                        current_time = pd.Timestamp.now() + pd.Timedelta(hours=5, minutes=30)

                        if pd.notna(commit_time):
                            remaining_seconds = (commit_time - current_time).total_seconds()
                        else:
                            remaining_seconds = None
                    else:
                        remaining_seconds = None

                    status_value = str(shipment_row.get("Status", pd.Series([""])).values[0]).lower().strip()
                    location_type = str(shipment_row.get("Location_Type", pd.Series([""])).values[0]).lower()
                    region = str(shipment_row.get("Region", pd.Series([""])).values[0]).upper()

                    if status_value == "ontime":
                        sla_status = "Delivered"
                    elif status_value in ["commit_fail", "pod_commit_fail"]:
                        sla_status = "Breached"
                    elif remaining_seconds is None:
                        sla_status = "Unknown"
                    elif remaining_seconds < 0:
                        sla_status = "Breached"
                    elif location_type == "ramp" and remaining_seconds < 48 * 3600:
                        sla_status = "Critical"
                    elif region == "INTERNATIONAL" and remaining_seconds < 24 * 3600:
                        sla_status = "Critical"
                    elif remaining_seconds < 2 * 3600:
                        sla_status = "Critical"
                    elif remaining_seconds < 6 * 3600 and risk_pct >= 60:
                        sla_status = "Critical"
                    elif remaining_seconds < 6 * 3600:
                        sla_status = "Warning"
                    elif risk_pct >= 70:
                        sla_status = "Warning"
                    else:
                        sla_status = "Safe"

                    st.metric("Current SLA Status", sla_status)

                except Exception as e:
                    st.warning("SLA status unavailable.")
                    st.text(str(e))


                try:
                    st.divider()
                    st.subheader("🧠 AI Root Cause Prediction")
                    st.caption("These operational factors contributed most to the predicted SLA breach risk.")

                    feature_names = {
                        "shp_pce_qty": "Shipment Volume",
                        "recp_pstl_cd": "Destination Postal Code",
                        "Dest Loc": "Destination Hub",
                        "Time_Remaining_Commit": "Remaining SLA Time",
                        "Trk Nos": "Tracking Number",
                        "last_scan": "Last Scan Status",
                        "last_scan_loc": "Last Scan Location",
                        "cntry_cd": "Country Code",
                        "Emp Nos": "Employee Count"
                    }

                    X_single_raw = process_data(shipment_row, bundle)
                    
                    expected_cols = bundle.get("expected_cols", [])
                    for col in expected_cols:
                        if col not in X_single_raw.columns:
                            X_single_raw[col] = 0
                            
                    X_single = X_single_raw[expected_cols].copy()
                    X_single = X_single.apply(pd.to_numeric, errors='coerce').fillna(0)

                    with st.spinner("Extracting AI Target-Encoded root-cause SHAP values..."):
                        try:
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer.shap_values(X_single)
                            
                            if isinstance(shap_values, list):
                                impact_values = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
                            else:
                                impact_values = shap_values[0]
                                
                        except Exception as e:
                            # Fallback to absolute feature importance if SHAP structure fails
                            impact_values = model.feature_importances_

                    feature_impact = pd.DataFrame({
                        "Feature": X_single.columns,
                        "Impact": impact_values
                    })

                    exclude_features = ["cntry_cd", "Trk Nos"]

                    feature_impact = feature_impact[
                        ~feature_impact["Feature"].isin(exclude_features)
                    ]

                    feature_impact["AbsImpact"] = feature_impact["Impact"].abs()

                    top_drivers = feature_impact.sort_values(
                        "AbsImpact",
                        ascending=False
                    ).head(5)

                    st.write("### 🔍 Top Risk Drivers")

                    for _, row in top_drivers.iterrows():
                        feature = feature_names.get(row["Feature"], row["Feature"])
                        impact = row["Impact"]

                        direction = "↑ increases risk" if impact > 0 else "↓ reduces risk"

                        st.write(
                            f"• **{feature}** → Impact Score {impact:.4f} ({direction})"
                        )

                    st.subheader("📊 Risk Driver Visualization")

                    top_drivers["Feature"] = top_drivers["Feature"].apply(
                        lambda x: feature_names.get(x, x)
                    )

                    top_drivers["Impact"] = top_drivers["Impact"] * 100

                    fig_root = px.bar(
                        top_drivers,
                        x="Impact",
                        y="Feature",
                        orientation="h",
                        color="Impact",
                        color_continuous_scale="Reds",
                        title="Top Factors Driving Shipment Risk",
                        height=400
                    )

                    fig_root.update_layout(
                        yaxis=dict(categoryorder="total ascending"),
                        xaxis_title="Impact Score (Scaled)",
                        yaxis_title="Operational Factor"
                    )

                    st.plotly_chart(fig_root, use_container_width=True)

                except Exception as e:
                    st.warning("Root cause explanation unavailable.")
                    st.text(str(e))
