"""
app.py  –  Sea Lion Population Trend Classification
MSIS 522 HW1 | University of Washington
Run: streamlit run app.py
"""
import os, json, warnings, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st
from scipy import stats
from scipy.interpolate import PchipInterpolator
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import plot_tree

warnings.filterwarnings("ignore")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
sys.path.insert(0, BASE_DIR)

# -- Ocean palette --------------------------------------------------------------
OC = {
    "navy":    "#0d2137", "dark":    "#0f3460", "mid":     "#1a7abf",
    "light":   "#5bb0e0", "pale":    "#c8e6f5", "decline": "#c0392b",
    "stable":  "#1a7a4a", "accent":  "#e8a000", "neutral": "#4a7a9b",
}
# Blue sequential for multi-region charts
BLUES  = ["#083d6e","#0f5a9c","#1a7abf","#3498db","#5bb0e0","#87ceeb"]
WDPS_C = ["#083d6e","#1a5c8a","#1a7abf","#3498db","#5bb0e0","#87ceeb"]
EDPS_C = ["#0a4a6e","#0d6b8c","#1a8fa8","#26b5c5","#40cfd5","#7de0e6"]

plt.rcParams.update({
    "font.family":"DejaVu Sans","axes.facecolor":"#f8fbff",
    "figure.facecolor":"#f8fbff","axes.edgecolor":"#d0dde8",
    "axes.labelcolor":"#0f3460","xtick.color":"#4a7a9b",
    "ytick.color":"#4a7a9b","text.color":"#0d2137",
    "grid.color":"#dde8f0","grid.linewidth":0.7,
})

# -- Page config ----------------------------------------------------------------
st.set_page_config(page_title="SeaLion Analytics", page_icon="🦭",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown('<style>\n@import url(\'https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&display=swap\');\nhtml,body,[class*="css"]{font-family:\'DM Sans\',sans-serif;}\n.stApp{background:linear-gradient(160deg,#f0f4f8 0%,#e8eef5 50%,#dde8f0 100%);}\nsection[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d2137 0%,#0f3460 40%,#164e7a 100%) !important;border-right:1px solid rgba(255,255,255,0.08);}\nsection[data-testid="stSidebar"] *{color:#d4e8f7 !important;}\nsection[data-testid="stSidebar"] strong{color:#7ecfff !important;}\nsection[data-testid="stSidebar"] hr{border-color:rgba(126,207,255,0.25) !important;}\ndiv[data-testid="stTabs"]>div>div>button{font-family:\'DM Sans\',sans-serif !important;font-size:0.9rem !important;font-weight:500 !important;color:#3d6080 !important;padding:0.5rem 0.9rem !important;}\ndiv[data-testid="stTabs"]>div>div>button[aria-selected="true"]{color:#0f3460 !important;font-weight:700 !important;border-bottom:3px solid #1a7abf !important;background:rgba(255,255,255,0.7) !important;}\ndiv[data-testid="metric-container"]{background:rgba(255,255,255,0.75);border:1px solid rgba(26,122,191,0.18);border-radius:12px;padding:0.8rem 1rem;box-shadow:0 2px 12px rgba(15,52,96,0.07);}\ndiv[data-testid="metric-container"] label{font-size:0.78rem !important;letter-spacing:0.06em !important;text-transform:uppercase !important;color:#4a7a9b !important;}\ndiv[data-testid="metric-container"] [data-testid="stMetricValue"]{font-family:\'Cormorant Garamond\',serif !important;font-size:2.1rem !important;font-weight:700 !important;color:#0d2137 !important;}\ndiv[data-testid="stDataFrame"]{border-radius:10px;overflow:hidden;box-shadow:0 2px 12px rgba(15,52,96,0.08);}\n.dash-title{font-family:\'Cormorant Garamond\',serif;font-size:3.1rem;font-weight:700;color:#0d2137;line-height:1.1;}\n.dash-subtitle{font-family:\'DM Sans\',sans-serif;font-size:1.0rem;font-weight:300;color:#4a7a9b;letter-spacing:0.04em;text-transform:uppercase;}\n.section-header{font-family:\'Cormorant Garamond\',serif;font-size:1.55rem;font-weight:600;color:#0f3460;border-bottom:2px solid #1a7abf;padding-bottom:0.3rem;margin:1.4rem 0 0.9rem 0;}\n.metric-card{background:linear-gradient(135deg,rgba(255,255,255,0.9) 0%,rgba(232,242,250,0.9) 100%);border-left:4px solid #1a7abf;border-radius:10px;padding:1rem 1.3rem;margin:0.3rem 0;box-shadow:0 2px 10px rgba(15,52,96,0.06);}\n.metric-val{font-family:\'Cormorant Garamond\',serif;font-size:2.2rem;font-weight:700;color:#0d2137;line-height:1.1;}\n.metric-label{font-family:\'DM Sans\',sans-serif;font-size:0.78rem;color:#4a7a9b;letter-spacing:0.05em;text-transform:uppercase;}\n.insight-box{background:linear-gradient(135deg,rgba(224,240,255,0.95) 0%,rgba(205,230,250,0.95) 100%);border-left:4px solid #1a7abf;border-radius:10px;padding:1rem 1.2rem;margin:0.6rem 0;font-size:0.93rem;line-height:1.65;}\n.forecast-box{background:linear-gradient(135deg,rgba(224,240,255,0.95) 0%,rgba(200,230,250,0.95) 100%);border-left:4px solid #1a7abf;border-radius:10px;padding:1rem 1.2rem;margin:0.6rem 0;font-size:0.93rem;line-height:1.65;}\n</style>', unsafe_allow_html=True)

# -- Sidebar ---------------------------------------------------------------------
with st.sidebar:
    # Accept any common logo filename
    logo_path = None
    for fname in ["sealion_img1.png", "logo.png", "sealion_img.png", "sea_lion_logo.png"]:
        p = os.path.join(BASE_DIR, fname)
        if os.path.exists(p):
            logo_path = p
            break
    if logo_path:
        # Reduce dead space above/below the image
        st.markdown('<div style="margin-top:-3rem;margin-bottom:-2rem;line-height:0;">', unsafe_allow_html=True)
        st.image(logo_path, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("""<hr style="border-color:rgba(126,207,255,0.2);margin:0.4rem 0 0.8rem 0;">""", unsafe_allow_html=True)
    st.markdown('<div style="font-family:\'DM Sans\',sans-serif;font-size:0.85rem;line-height:1.7;padding:0 0.3rem;">\n        <div style="color:#7ecfff;font-weight:600;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.3rem;">Dataset</div>\n        Steller Sea Lion aerial surveys across Alaska &amp; Pacific Coast rookeries (1970–2024).\n        <br><br>\n        <div style="color:#7ecfff;font-weight:600;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.3rem;">Task</div>\n        Binary classification -- predict whether a site shows a <strong>declining</strong> population trend.\n        <br><br>\n        <div style="color:#7ecfff;font-weight:600;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.3rem;">Target</div>\n        <code style="background:rgba(126,207,255,0.12);padding:0.1rem 0.4rem;border-radius:4px;font-size:0.78rem;color:#a8d8f0;">is_declining = 1</code>\n        if slope &lt; 0 AND recent &lt; 85% of early counts.\n    </div>', unsafe_allow_html=True)
    st.markdown("""<hr style="border-color:rgba(126,207,255,0.2);margin:0.8rem 0 0.5rem 0;">""", unsafe_allow_html=True)
    st.markdown("""<div style="font-family:'DM Sans',sans-serif;font-size:0.72rem;color:#6a9ab8;text-align:center;">Data: NOAA / NMFS<br>MSIS 522 · HW1 · UW</div>""", unsafe_allow_html=True)

# -- Cached loaders ------------------------------------------------------------
@st.cache_data
def load_data():
    return joblib.load(os.path.join(MODEL_DIR, "processed_df.pkl"))

@st.cache_data
def load_splits():
    return joblib.load(os.path.join(MODEL_DIR, "splits.pkl"))

@st.cache_resource
def load_model(name):
    tag = name.replace(" ", "_").lower()
    p = os.path.join(MODEL_DIR, f"{tag}.pkl")
    return joblib.load(p) if os.path.exists(p) else None

@st.cache_resource
def load_scaler():
    return joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

@st.cache_data
def load_results():
    with open(os.path.join(MODEL_DIR, "results.json")) as f:
        return json.load(f)

@st.cache_data
def load_shap_data():
    return joblib.load(os.path.join(MODEL_DIR, "shap_data.pkl"))

@st.cache_data
def load_clustering():
    return joblib.load(os.path.join(MODEL_DIR, "clustering.pkl"))

@st.cache_data
def load_mlp_history():
    return joblib.load(os.path.join(MODEL_DIR, "mlp_history.pkl"))

@st.cache_data
def load_long_format():
    try:
        from data_utils import get_long_format
        df = joblib.load(os.path.join(MODEL_DIR, "processed_df.pkl"))
        long_df = get_long_format(df)
        # Ensure is_observed column always exists
        if "is_observed" not in long_df.columns:
            long_df["is_observed"] = True
        return long_df
    except Exception as e:
        # Fallback: build minimal long format from processed_df directly
        import warnings
        warnings.warn(f"get_long_format failed ({e}), using fallback")
        df = joblib.load(os.path.join(MODEL_DIR, "processed_df.pkl"))
        rows = []
        for _, r in df.iterrows():
            for yr, cnt in zip(r["_years"], r["_counts"]):
                rows.append({"site": r["site"], "region": r["region"],
                             "dps": r["dps"], "count_type": r["count_type"],
                             "year": int(yr), "count": cnt,
                             "is_observed": True, "is_declining": r["is_declining"]})
        return pd.DataFrame(rows).dropna(subset=["count"])

# -- Helpers -------------------------------------------------------------------
def ocean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    return ax

def regional_palette(regions):
    n = len(regions)
    cmap = plt.cm.Blues
    return [cmap(0.35 + 0.55 * i / max(n - 1, 1)) for i in range(n)]

# -- Load data -----------------------------------------------------------------
df       = load_data()
X_train, X_test, y_train, y_test, feature_names = load_splits()
scaler   = load_scaler()
results  = load_results()
shap_data = load_shap_data()
cluster  = load_clustering()
mlp_hist = load_mlp_history()
df_long  = load_long_format()

MODEL_NAMES = ["Logistic Regression","LASSO","Ridge","CART","Random Forest","LightGBM","MLP","GAM"]
ROC_COLORS  = {
    "Logistic Regression":"#1a7abf","LASSO":"#7a4a9b","Ridge":"#2ab09a",
    "CART":"#d4700a","Random Forest":"#083d6e","LightGBM":"#c0392b",
    "MLP":"#e8a000","GAM":"#0f3460",
}

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
tabs = st.tabs([
    "🦭  Steller Sea Lion",
    "📋  Executive Summary",
    "📊  Data Visualization",
    "🏆  Model Performance",
    "🔍  SHAP & Prediction",
    "📈  agTrend & Forecast",
    "🗂️  Clustering",
])

# -----------------------------------------------------------------------------
# TAB 0 -- STELLER SEA LION
# -----------------------------------------------------------------------------
with tabs[0]:
    st.markdown(
        '<div style="padding:0.5rem 0 1rem 0;">'
        '<div class="dash-subtitle">Species Profile & Conservation Status</div>'
        '<div class="dash-title">Steller Sea Lion</div>'
        '<div style="font-family:DM Sans,sans-serif;font-size:1rem;color:#4a7a9b;margin-top:0.4rem;font-weight:300;">'
        'Eumetopias jubatus -- North Pacific keystone predator'
        '</div></div>',
        unsafe_allow_html=True)
    st.markdown('---')

    img_col, facts_col = st.columns([1.5, 1])
    with img_col:
        ssl_img = None
        for fname in ['steller_sea_lion.jpg', 'steller_sea_lion.png', 'ssl.jpg']:
            p = os.path.join(BASE_DIR, fname)
            if os.path.exists(p):
                ssl_img = p
                break
        if ssl_img:
            st.image(ssl_img, caption='Steller sea lion (Eumetopias jubatus) -- Photo: NOAA Fisheries', use_container_width=True)
    with facts_col:
        st.markdown(
            '<div style="background:rgba(15,52,96,0.08);border-radius:10px;padding:1rem 1.2rem;font-size:0.87rem;line-height:1.85;">'
            '<b style="font-size:0.95rem;">Quick Facts</b><br><br>'
            '<b>Family:</b> Otariidae (eared seals)<br>'
            '<b>Size:</b> Males up to 2,500 lbs, 11 ft<br>'
            '<b>Diet:</b> Fish, squid, octopus<br>'
            '<b>Range:</b> Japan to California<br>'
            '<b>WDPS:</b> Endangered (ESA 1997)<br>'
            '<b>EDPS:</b> Delisted, recovered 2013<br>'
            '<b>Named for:</b> Georg Steller, 1742<br><br>'
            '<b>2024 NOAA Survey:</b><br>'
            'WDPS overall: +0.96%/yr<br>'
            'W ALEU non-pups: -5.67%/yr<br>'
            'W ALEU pups: -4.07%/yr'
            '</div>',
            unsafe_allow_html=True)
    st.markdown('---')
    st.markdown('### About the Species')
    st.markdown(
        'The **Steller sea lion** (*Eumetopias jubatus*) is the largest member of the eared seal family '
        '(Otariidae), which includes all sea lions and fur seals. Named for German naturalist Georg Wilhelm '
        'Steller who first described the species during the 1741-42 Bering expedition, they are the sole '
        'living member of their genus. Adult males can reach 11 feet and 2,500 lbs -- nearly three times '
        'the size of females. They inhabit rocky shores and offshore haul-out sites across the coastal '
        'North Pacific from Japan to California, feeding primarily on walleye pollock, Pacific cod, '
        'Atka mackerel, and squid. Unlike the familiar barks of California sea lions, Steller sea lions '
        'produce deep, guttural roars. They are culturally and nutritionally important to Alaska Native '
        'communities as a traditional subsistence resource.'
    )
    st.markdown('### Conservation History')
    st.markdown(
        'Steller sea lions were first **listed as Threatened under the Endangered Species Act in 1990** '
        'following a dramatic, unexplained population collapse in Alaska beginning in the late 1970s. '
        'In 1997, NOAA Fisheries formally recognized two **Distinct Population Segments (DPS)**:'
    )
    st.markdown(
        '- **Western DPS (WDPS):** Ranges west of Cape Suckling, Alaska (144°W) through the western '
        'Aleutian Islands. Listed as *Endangered* in 1997. Remains endangered today. Population declined '
        'over 80% from its 1970s peak, with the steepest losses in the western Aleutian Islands.\n'
        '- **Eastern DPS (EDPS):** Ranges east of Cape Suckling through California. Listed as *Threatened* '
        'in 1997, then **fully recovered and delisted in 2013** -- one of the ESA\'s landmark successes.'
    )
    st.markdown(
        'The cause of the western decline remains actively debated. Leading hypotheses include: '
        'nutritional stress from commercial fishing competition for pollock and cod; oceanographic '
        'regime shifts reducing prey quality and availability; increased killer whale predation; '
        'and contaminant effects. The 2014-2016 North Pacific marine heatwave further stressed '
        'the population. The most recent NOAA survey (Sweeney et al. 2025) found the overall '
        'western DPS slightly increasing (+0.96%/yr non-pups since 2009), but the western Aleutian '
        'Islands region continues severe decline (-5.67%/yr non-pups, -4.07%/yr pups) while '
        'the magnitude of increase has been slowing since ~2016.'
    )
    st.markdown('### Survey Methods')
    st.markdown(
        'NOAA\'s Marine Mammal Laboratory (MML) conducts annual crewed aircraft and vessel-based surveys '
        'of known rookery and haulout sites. Two count types are recorded:\n\n'
        '**Non-pup counts** (adults and juveniles): Conducted throughout the year using ground surveys, '
        'cliff-side overlooks, and aerial imagery (oblique and vertical). Breeding-season counts '
        '(June-July) are used for population trend analysis; other counts are used for distribution '
        'and habitat analyses. Non-pup counts capture only a fraction of total site use -- sea lions '
        'haul out less in winter, so winter counts underrepresent the population.\n\n'
        '**Pup counts**: Conducted in June-July when pups are born. Pup counts serve as a leading '
        'indicator of population health because pups represent new recruitment. Declining pup counts '
        'signal reproductive failure, which precedes adult population decline by years.'
    )
    st.markdown(
        '<div class="insight-box">'
        '<b>NOAA Species Page:</b> '
        '<a href="https://www.fisheries.noaa.gov/species/steller-sea-lion" target="_blank">fisheries.noaa.gov/species/steller-sea-lion</a><br><br>'
        '<b>Official Survey Reports (agTrend analysis):</b> '
        '<a href="https://www.fisheries.noaa.gov/alaska/marine-mammal-protection/steller-sea-lion-survey-reports" target="_blank">NOAA Steller Sea Lion Survey Reports</a><br><br>'
        '<b>Pup count database:</b> '
        '<a href="https://www.fisheries.noaa.gov/resource/data/counts-alaska-steller-sea-lion-pups-conducted-rookeries-alaska-1961-06-22" target="_blank">NOAA Pup Counts (1961-present)</a><br><br>'
        '<b>Non-pup count database:</b> '
        '<a href="https://www.fisheries.noaa.gov/resource/data/counts-alaska-steller-sea-lion-adult-and-juvenile-non-pup-conducted-rookeries" target="_blank">NOAA Non-Pup Counts Database</a>'
        '</div>',
        unsafe_allow_html=True)

# TAB 1 -- EXECUTIVE SUMMARY
# -----------------------------------------------------------------------------
with tabs[1]:
    st.markdown(
        '<div style="padding:0.5rem 0 1rem 0;">'
        '<div class="dash-subtitle">MSIS 522 · Machine Learning · University of Washington</div>'
        '<div class="dash-title">Executive Summary</div>'
        '<div style="font-family:DM Sans,sans-serif;font-size:1rem;color:#4a7a9b;margin-top:0.4rem;font-weight:300;">'
        'Predicting Steller sea lion population decline using machine learning'
        '</div></div>',
        unsafe_allow_html=True)
    st.markdown('---')

    # KPI strip
    n_dec  = int(df['is_declining'].sum())
    n_stab = int((df['is_declining']==0).sum())
    n_sit  = len(df)
    best_m   = max(results, key=lambda m: results[m]['auc_roc'])
    best_a   = results[best_m]['auc_roc']
    best_f1m = max(results, key=lambda m: results[m]['f1'])
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.markdown(f'<div class="metric-card"><div class="metric-val">{n_sit}</div><div class="metric-label">Survey Sites</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#c0392b">{n_dec}</div><div class="metric-label">Declining ({n_dec/n_sit*100:.0f}%)</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#1a7a4a">{n_stab}</div><div class="metric-label">Stable / Recovering</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-val">{best_a:.3f}</div><div class="metric-label">Best AUC ({best_m})</div></div>', unsafe_allow_html=True)
    c5.markdown(f'<div class="metric-card"><div class="metric-val">{results[best_f1m]["f1"]:.3f}</div><div class="metric-label">Best F1 ({best_f1m})</div></div>', unsafe_allow_html=True)
    st.markdown('---')

    # Required section 1: Dataset description
    st.markdown('<div class="section-header">The Dataset & Prediction Task</div>', unsafe_allow_html=True)
    st.markdown(
        f'This project uses aerial survey count data collected annually by NOAA\'s National Marine Fisheries '
        f'Service (NMFS) at Steller sea lion rookeries and haulout sites across Alaska and the Pacific Coast. '
        f'The dataset spans approximately 1970 to 2024 and covers **{n_sit} survey sites**. '
        f'At each site, observers record the number of sea lions present -- but surveys are not conducted '
        f'every year at every location, leaving a sparse time-series with many missing years. '
        f'Two count types exist: **non-pup counts** (adults and juveniles, used for trend analysis when '
        f'collected during the June-July breeding season) and **pup counts** (newborns, a leading indicator '
        f'of population recruitment). These are the same raw data used in NOAA\'s official annual '
        f'agTrend survey reports.\n\n'
        f'Each site\'s multi-decade count time-series is transformed into **{len(feature_names)} features**: '
        f'16 numerical features capturing trend direction, magnitude, variability, and temporal structure '
        f'(e.g., % decline from peak, recent vs. early mean counts, trend slope R-squared), plus categorical '
        f'features for population segment (WDPS/EDPS), geographic region, and count type. '
        f'Missing survey years are imputed using PCHIP spline interpolation in log-count space -- '
        f'the same philosophy as the agTrend.ssl R package developed by NOAA scientists for this dataset.\n\n'
        f'The **prediction target** is `is_declining` (binary): a site is labeled Declining (1) if it shows '
        f'both a negative linear trend slope AND recent counts below 85% of early counts. '
        f'Otherwise it is labeled Stable/Recovering (0). Note that grouping Stable and Recovering together '
        f'is a deliberate simplification -- ecologically, stable (flat trend) and recovering (positive trend) '
        f'are distinct states, and a 3-class model is a natural extension for future work. '
        f'For this analysis, the key conservation question is binary: **is this site in decline?** '
        f'Of the {n_sit} sites, {n_dec} ({n_dec/n_sit*100:.1f}%) are Declining and {n_stab} ({n_stab/n_sit*100:.1f}%) are Stable/Recovering.'
    )
    st.markdown('---')

    # Required section 2: Why it matters
    st.markdown('<div class="section-header">Why This Problem Matters</div>', unsafe_allow_html=True)
    st.markdown(
        'The western Steller sea lion population has declined by over 80% since the 1970s and remains listed '
        'as **Endangered under the U.S. Endangered Species Act**. This is not just an ecological issue -- it '
        'has significant regulatory and economic consequences. Under the ESA and the Marine Mammal Protection '
        'Act, NOAA must ensure that commercial fishing operations in the North Pacific do not jeopardize the '
        'western population\'s recovery. This means fishery closures, quota restrictions, and exclusion zones '
        'around critical Steller sea lion habitat -- particularly in the western Aleutian Islands where pollock '
        'and Pacific cod fishing overlaps with the most severely declining rookeries. These restrictions cost '
        'the commercial fishing industry hundreds of millions of dollars annually and are a source of ongoing '
        'scientific and legal conflict.\n\n'
        'In this context, a model that can identify which survey sites are most at risk of population collapse -- '
        'and explain *why* using interpretable features -- is directly actionable. NOAA managers need to '
        'prioritize monitoring resources, justify critical habitat designations, and communicate population '
        'status to policymakers and courts. The SHAP analysis in this project identifies the specific '
        'site-level features driving decline predictions, giving conservation managers a data-driven basis '
        'for these decisions. The population forecast (Tab 6) further extends this by projecting which regions '
        'may fall below viable population thresholds under current trends -- providing a forward-looking tool '
        'for conservation planning.'
    )
    st.markdown('---')

    # Required section 3: Approach and key findings
    st.markdown('<div class="section-header">Approach & Key Findings</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns([1.1, 1])
    with col_a:
        st.markdown(
            f'**Analytical approach:** Seven machine learning models were trained and compared on the '
            f'70/30 stratified train/test split: Logistic Regression (baseline), LASSO, Ridge, '
            f'CART decision tree, Random Forest, LightGBM gradient boosting, MLP neural network, '
            f'and a Generalized Additive Model (GAM) inspired by the agTrend.ssl methodology. '
            f'All models used the same engineered feature set. Tree-based models received raw features '
            f'while linear models and the MLP received standardized features. Hyperparameters were tuned '
            f'using 5-fold stratified cross-validation with GridSearchCV, and model explanations were '
            f'generated using SHAP (SHapley Additive exPlanations) applied to the Random Forest.\n\n'
            f'**Key findings:** The best-performing model was **{best_m}** with an AUC-ROC of {best_a:.3f} '
            f'and F1 of {results[best_f1m]["f1"]:.3f}. All models substantially outperformed the naive '
            f'52.7% accuracy baseline. Tree ensembles outperformed linear models, confirming that '
            f'non-linear interactions between population segment, region, and trajectory cannot be '
            f'captured by a linear boundary alone. The most predictive features -- identified by both '
            f'Random Forest importance and SHAP analysis -- are **% decline from historical peak** and '
            f'**recent mean count**. Sites that have lost most relative to their peak and show low recent '
            f'counts are almost certainly classified as Declining. DPS membership (WDPS vs EDPS) is '
            f'the third most important feature, reflecting the fundamental difference between the '
            f'endangered western population and the recovering eastern population.'
        )
    with col_b:
        st.markdown('**Results at a glance:**')
        st.markdown(
            f'<div class="insight-box">'
            f'Best model: <b>{best_m}</b><br>'
            f'AUC-ROC: <b>{best_a:.3f}</b> | F1: <b>{results[best_f1m]["f1"]:.3f}</b><br><br>'
            f'W ALEU region: -5.67%/yr non-pups (2009-2024)<br>'
            f'EDPS: fully recovered, delisted 2013<br>'
            f'Top predictor: % decline from peak<br>'
            f'All 7 models beat naive baseline<br><br>'
            f'<b>Note on classification:</b> Stable and Recovering sites '
            f'are grouped as one class (0). Ecologically these are distinct '
            f'states -- a 3-class model (Declining / Stable / Increasing) '
            f'is a natural extension for future work but binary classification '
            f'addresses the core conservation question: is this site in trouble?'
            f'</div>',
            unsafe_allow_html=True)
        st.markdown('**Explore more:**')
        st.markdown(
            '- **Tab 3:** Data visualizations -- regional trends, feature distributions\n'
            '- **Tab 4:** Full model comparison, ROC curves, hyperparameters\n'
            '- **Tab 5:** SHAP explainability + interactive prediction\n'
            '- **Tab 6:** agTrend analysis + population forecast to 2040\n'
            '- **Tab 7:** Clustering -- unsupervised validation of DPS boundaries'
        )

# TAB 2 -- DATA VISUALIZATION
# -----------------------------------------------------------------------------
with tabs[2]:
    st.markdown("## 📊 Descriptive Analytics")

    # -- 1.1 Dataset Introduction ----------------------------------------------
    st.markdown('<div class="section-header">1.1 · Dataset Introduction</div>', unsafe_allow_html=True)
    n_feat_num = len(["mean_count","max_count","min_count","std_count","cv","n_years",
                       "r_squared","trend_pct_per_year","early_mean","recent_mean",
                       "pct_decline_from_peak","years_since_peak","avg_1980s","avg_2000s","avg_2010s","peak_year"])
    n_feat_cat = 3
    st.markdown(f"""
**What the dataset contains:** This dataset consists of aerial survey count records for Steller sea lions
(*Eumetopias jubatus*) collected by NOAA's National Marine Fisheries Service (NMFS) at rookeries and
haul-out sites across Alaska and the Pacific Coast from approximately 1970 to 2024. Each row represents
one survey site. Raw counts are recorded per year -- but surveys are not conducted every year at every site,
leaving a sparse, irregularly sampled time-series. For modeling, each site's time-series is collapsed into
{n_feat_num} numerical features (trend statistics, recent vs. early counts, variability metrics, decade averages)
and {n_feat_cat} categorical features (DPS membership, geographic region, count type: pup or non-pup).
The final dataset contains **{len(df)} sites** and **{len(feature_names)} features** after one-hot encoding.

**Prediction target:** The binary target variable `is_declining` equals 1 if a site has both (a) a negative
linear trend slope and (b) a recent mean count below 85% of its early mean count. It equals 0 otherwise
(Stable or Recovering). This two-condition rule ensures that only sites with both a downward trajectory
*and* meaningful population loss are labeled Declining -- a site with a slightly negative slope but still
high counts would not be flagged. Of the {len(df)} sites, **{int(df["is_declining"].sum())} are Declining
({df["is_declining"].mean()*100:.1f}%)** and **{int((df["is_declining"]==0).sum())} are Stable/Recovering
({(1-df["is_declining"].mean())*100:.1f}%)**.

**Why this task is interesting and impactful:** The Steller sea lion has been the subject of one of the most
contentious wildlife management debates in U.S. history. The Western DPS has lost over 80% of its 1970s
population and remains endangered, with causes still debated among scientists. A model that can identify
which site-level features most strongly predict decline -- and which sites are at highest risk -- has direct
applications for NOAA resource allocation, critical habitat designation, and fishery management decisions
(commercial pollock and cod fishing in the western Aleutians is directly constrained by sea lion recovery
requirements). This is also a methodologically interesting dataset because of its sparse, irregularly
sampled time-series structure and the contrast between a recovering Eastern DPS and a declining Western DPS.
""")
    st.markdown("---")

    # -- 1.2 Target Distribution -----------------------------------------------
    st.markdown('<div class="section-header">1.2 · Target Distribution</div>', unsafe_allow_html=True)
    st.markdown(f"""
**Class balance:** The target is **slightly imbalanced** -- {int(df["is_declining"].sum())} Declining
({df["is_declining"].mean()*100:.1f}%) vs {int((df["is_declining"]==0).sum())} Stable
({(1-df["is_declining"].mean())*100:.1f}%). This modest imbalance reflects the real-world situation:
the WDPS (endangered) accounts for the majority of declining sites, while EDPS sites are mostly recovering.
The imbalance is not severe enough to require resampling -- F1 score is used as the primary metric
(rather than accuracy) since it penalizes false negatives, which matter most in a conservation context
where missing a truly declining population is the costlier error. All models use `class_weight='balanced'`
where supported to further mitigate imbalance effects.
""")
    c1, c2 = st.columns([1, 1.5])
    with c1:
        fig, ax = plt.subplots(figsize=(5,4))
        vals = [(df["is_declining"]==0).sum(), df["is_declining"].sum()]
        bars = ax.barh(["Stable","Declining"], vals,
                       color=[OC["mid"], OC["dark"]], edgecolor="white", linewidth=1.5)
        for bar,v in zip(bars, vals):
            ax.text(bar.get_width()+3, bar.get_y()+bar.get_height()/2,
                    f"{v} ({v/sum(vals)*100:.1f}%)", va="center", fontweight="bold", fontsize=11)
        ax.set_xlim(0, max(vals)*1.3); ax.set_xlabel("Number of Sites")
        ax.set_title("Target Class Distribution", fontweight="bold"); ocean_ax(ax)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('<div class="insight-box">Slightly imbalanced (~53% Declining), reflecting the real-world WDPS endangered status vs EDPS recovery.</div>', unsafe_allow_html=True)
    with c2:
        fig, ax = plt.subplots(figsize=(7,4))
        rc = df.groupby(["region","is_declining"]).size().unstack(fill_value=0)
        rc.columns = ["Stable","Declining"]
        rc.sort_values("Declining",ascending=True).plot(
            kind="barh", ax=ax, color=[OC["mid"], OC["dark"]], edgecolor="white")
        ax.set_xlabel("Number of Sites"); ax.set_title("Decline by Region", fontweight="bold"); ocean_ax(ax)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('<div class="insight-box"><b>W ALEU</b> is most severely declining. EDPS regions (BC, OR, WA, CA) show mostly stable/recovering trends.</div>', unsafe_allow_html=True)

    st.markdown("---")

    # -- 2. WDPS all regions over time -----------------------------------------
    st.markdown('<div class="section-header">2 · WDPS Regional Trends Over Time</div>', unsafe_allow_html=True)
    st.caption("ℹ️ Missing survey years filled with PCHIP spline interpolation (agTrend.ssl methodology). Dots = actual observations.")

    wdps_regions = sorted(df_long[df_long["dps"]=="WDPS"]["region"].unique())
    wpal = WDPS_C[:len(wdps_regions)]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=False)
    axes = axes.flat
    for ax, region, color in zip(axes, wdps_regions, wpal):
        sub = df_long[df_long["region"]==region]
        yr_all = sub.groupby("year")["count"].sum()
        yr_obs = sub[sub["is_observed"]].groupby("year")["count"].sum()
        ax.fill_between(yr_all.index, yr_all.values/1000, alpha=0.18, color=color)
        ax.plot(yr_all.index, yr_all.values/1000, color=color, lw=2.2, zorder=3)
        ax.scatter(yr_obs.index, yr_obs.values/1000, color=color, s=24,
                   edgecolors="white", linewidths=0.7, zorder=4)
        ax.set_title(f"WDPS -- {region}", fontweight="bold", color=OC["dark"])
        ax.set_xlabel("Year"); ax.set_ylabel("Count (thousands)")
        ocean_ax(ax)
    # hide unused panels
    for ax in list(axes)[len(wdps_regions):]:
        ax.set_visible(False)
    plt.suptitle("WDPS (Western DPS -- Endangered) Regional Population Trends",
                 fontsize=14, fontweight="bold", color=OC["dark"], y=1.01)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Combined WDPS overlay
    fig, ax = plt.subplots(figsize=(12, 5))
    for region, color in zip(wdps_regions, wpal):
        sub = df_long[df_long["region"]==region]
        yr_all = sub.groupby("year")["count"].sum()
        ax.plot(yr_all.index, yr_all.values/1000, color=color, lw=2.2, label=region, alpha=0.9)
    ax.set_xlabel("Year"); ax.set_ylabel("Total Count (thousands)")
    ax.set_title("All WDPS Regions -- Combined View", fontweight="bold")
    ax.legend(fontsize=9); ocean_ax(ax)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown('<div class="insight-box">\n<b>W ALEU</b> shows the most severe decline -- from ~40k in the 1980s to near-critical levels today.\n<b>C ALEU</b> shows moderate decline while Gulf regions (W GULF, C GULF, E GULF) have partially stabilized.\nThe collapse tracks the loss of Steller sea lion prey (pollock, cod) linked to commercial fishing pressure and oceanographic regime shifts.\n</div>', unsafe_allow_html=True)

    st.markdown("---")

    # -- 3. EDPS all regions over time -----------------------------------------
    st.markdown('<div class="section-header">3 · EDPS Regional Trends Over Time</div>', unsafe_allow_html=True)
    edps_regions = sorted(df_long[df_long["dps"]=="EDPS"]["region"].unique())
    epal = EDPS_C[:len(edps_regions)]

    fig, axes = plt.subplots(1, len(edps_regions), figsize=(16, 4.5), sharey=False)
    if len(edps_regions)==1: axes=[axes]
    for ax, region, color in zip(axes, edps_regions, epal):
        sub = df_long[df_long["region"]==region]
        yr_all = sub.groupby("year")["count"].sum()
        yr_obs = sub[sub["is_observed"]].groupby("year")["count"].sum()
        ax.fill_between(yr_all.index, yr_all.values/1000, alpha=0.2, color=color)
        ax.plot(yr_all.index, yr_all.values/1000, color=color, lw=2.2, zorder=3)
        ax.scatter(yr_obs.index, yr_obs.values/1000, color=color, s=24,
                   edgecolors="white", linewidths=0.7, zorder=4)
        ax.set_title(f"EDPS -- {region}", fontweight="bold", color=OC["dark"])
        ax.set_xlabel("Year"); ax.set_ylabel("Count (thousands)"); ocean_ax(ax)
    plt.suptitle("EDPS (Eastern DPS -- Recovering) Regional Population Trends",
                 fontsize=14, fontweight="bold", color=OC["dark"], y=1.01)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Combined EDPS overlay
    fig, ax = plt.subplots(figsize=(12,5))
    for region, color in zip(edps_regions, epal):
        sub = df_long[df_long["region"]==region]
        yr_all = sub.groupby("year")["count"].sum()
        ax.plot(yr_all.index, yr_all.values/1000, color=color, lw=2.2, label=region, alpha=0.9)
    ax.set_xlabel("Year"); ax.set_ylabel("Total Count (thousands)")
    ax.set_title("All EDPS Regions -- Combined View", fontweight="bold")
    ax.legend(fontsize=9); ocean_ax(ax)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown('<div class="insight-box">\nAll EDPS regions show <b>recovery trends</b> since the late 1990s following federal protections under the ESA.\n<b>SE AK</b> and <b>BC</b> show the strongest recoveries. California populations remain smaller but are growing steadily.\n</div>', unsafe_allow_html=True)

    st.markdown("---")

    # -- 4. WDPS vs EDPS combined ----------------------------------------------
    st.markdown('<div class="section-header">4 · WDPS vs EDPS -- Big Picture</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12,5))
    for dps, color, label in [("WDPS", OC["dark"], "WDPS (Endangered)"),
                               ("EDPS", OC["mid"],  "EDPS (Delisted)")]:
        sub = df_long[df_long["dps"]==dps]
        yr_all = sub.groupby("year")["count"].sum()
        yr_obs = sub[sub["is_observed"]].groupby("year")["count"].sum()
        ax.fill_between(yr_all.index, yr_all.values/1000, alpha=0.13, color=color)
        ax.plot(yr_all.index, yr_all.values/1000, color=color, lw=2.8, label=label, zorder=3)
        ax.scatter(yr_obs.index, yr_obs.values/1000, color=color, s=25,
                   edgecolors="white", linewidths=0.7, zorder=4)
    ax.set_xlabel("Year"); ax.set_ylabel("Total Count (thousands)")
    ax.set_title("Total Population by DPS (Interpolated)", fontweight="bold")
    ax.legend(fontsize=11); ocean_ax(ax)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")

    # -- 5. Feature distributions ----------------------------------------------
    # -- 5a. Histograms --------------------------------------------------------
    st.markdown('<div class="section-header">5 · Feature Distributions by Class</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    feats = [("recent_mean","Recent Mean Count"),("pct_decline_from_peak","% Decline from Peak"),
             ("cv","Coefficient of Variation"),("years_since_peak","Years Since Peak"),
             ("n_years","N Survey Years"),("r_squared","Trend R²")]
    for ax,(feat,title) in zip(axes.flat, feats):
        for cls, color, lbl in [(0, OC["mid"], "Stable"), (1, OC["decline"], "Declining")]:
            vals = df[df["is_declining"]==cls][feat].dropna()
            ax.hist(vals, bins=25, alpha=0.6, color=color, label=lbl,
                    edgecolor="white", linewidth=0.5)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xlabel("Value"); ax.set_ylabel("Count"); ocean_ax(ax); ax.legend(fontsize=8)
    plt.suptitle("Feature Distributions: Declining vs Stable", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown('<div class="insight-box">\n<b>Recent Mean Count</b> shows the clearest separation -- declining sites cluster near zero while stable sites span a wide range,\nconfirming that current abundance is the most direct signal of decline status.\n<b>% Decline from Peak</b> shows a bimodal distribution with declining sites concentrated above 60%,\nwhile <b>Coefficient of Variation</b> is higher for declining sites, reflecting the irregular survey patterns\nas populations drop toward extinction. <b>Trend R²</b> tends to be higher for declining sites -- a consistently\nfalling trend is more statistically reliable than a noisy stable one.\n</div>', unsafe_allow_html=True)

    st.markdown("---")

    # -- 5b. Violin plots ------------------------------------------------------
    st.markdown('<div class="section-header">6 · Violin Plots -- Distribution Shape by Class</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    violin_feats = [("recent_mean","Recent Mean Count"),
                    ("pct_decline_from_peak","% Decline from Peak"),
                    ("cv","CV (Count Variability)")]
    for ax, (feat, title) in zip(axes, violin_feats):
        data_stable   = df[df["is_declining"]==0][feat].dropna()
        data_declining= df[df["is_declining"]==1][feat].dropna()
        parts = ax.violinplot([data_stable, data_declining], positions=[0,1],
                               showmedians=True, showextrema=True)
        for i,(pc,c) in enumerate(zip(parts["bodies"], [OC["mid"], OC["decline"]])):
            pc.set_facecolor(c); pc.set_alpha(0.7)
        parts["cmedians"].set_color("white"); parts["cmedians"].set_linewidth(2)
        ax.set_xticks([0,1]); ax.set_xticklabels(["Stable","Declining"], fontsize=11)
        ax.set_title(title, fontweight="bold"); ax.set_ylabel("Value"); ocean_ax(ax)
    plt.suptitle("Violin Plots: Distribution Shape by Decline Status", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown('<div class="insight-box">\nViolin plots reveal the full distribution shape, not just counts. <b>Recent Mean Count</b> shows\ndeclining sites are tightly clustered near zero -- there is very little variance, meaning almost all\ndeclining sites have similarly low recent populations regardless of region.\n<b>% Decline from Peak</b> for stable sites is bimodal (some never declined; others partially recovered),\nwhile declining sites have a narrow high-decline distribution. This tight separation explains why\ntree models achieve near-perfect classification.\n</div>', unsafe_allow_html=True)

    st.markdown("---")

    # -- 5c. Pup vs Non-Pup ---------------------------------------------------
    st.markdown('<div class="section-header">7 · Pup vs Non-Pup Counts by Region</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, dps, pal in [(axes[0],"WDPS",WDPS_C),(axes[1],"EDPS",EDPS_C)]:
        sub = df_long[(df_long["dps"]==dps) & (df_long["is_observed"])]
        pnp = sub.groupby(["region","count_type"])["count"].mean().unstack(fill_value=0)
        regions = pnp.index.tolist()
        colors_r = pal[:len(regions)]
        x = np.arange(len(regions)); w = 0.35
        for i,(ct,hatch) in enumerate([("Non-Pup",""),("Pup","///")]):
            if ct in pnp.columns:
                ax.bar(x + i*w, pnp[ct].values/1000, w, label=ct,
                       color=[c for c in colors_r], alpha=0.85, hatch=hatch, edgecolor="white")
        ax.set_xticks(x+w/2); ax.set_xticklabels(regions, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Mean Count (thousands)"); ax.set_title(f"{dps} -- Pup vs Non-Pup by Region", fontweight="bold")
        ax.legend(fontsize=9); ocean_ax(ax)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown('<div class="insight-box">\nPup counts are a leading indicator of population health -- pups represent recruitment (new births),\nso declining pup counts signal that a population is failing to replace itself, which precedes\nadult population decline by several years. <b>W ALEU</b> shows severely depressed pup counts\nrelative to non-pups, suggesting recruitment failure -- one of the key mechanisms behind the\nWDPS endangered listing. EDPS regions show healthier pup-to-adult ratios, consistent with recovery.\n</div>', unsafe_allow_html=True)

    st.markdown("---")

    # -- 8. Correlation heatmap ------------------------------------------------
    st.markdown('<div class="section-header">8 · Correlation Heatmap</div>', unsafe_allow_html=True)
    num_f = ["mean_count","max_count","min_count","std_count","cv","n_years",
             "trend_pct_per_year","r_squared","early_mean","recent_mean",
             "pct_decline_from_peak","years_since_peak","is_declining"]
    corr = df[num_f].corr()
    fig, ax = plt.subplots(figsize=(11,9))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={"shrink":0.8}, annot_kws={"size":8})
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=12)
    plt.xticks(rotation=40, ha="right", fontsize=9); plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown('<div class="insight-box">\n<b>pct_decline_from_peak</b> (+0.63) and <b>recent_mean</b> (−0.55) are most correlated with <i>is_declining</i>.\nMean/early/max counts are highly collinear -- tree models handle this naturally; linear models benefit from the L1/L2 regularization in LASSO/Ridge.\n</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 3 -- MODEL PERFORMANCE
# -----------------------------------------------------------------------------
with tabs[3]:
    st.markdown("## 🏆 Model Performance Report")

    # -- 2.7 Discussion at top -------------------------------------------------
    st.markdown('<div class="section-header">2.7 · Model Comparison -- Summary & Discussion</div>', unsafe_allow_html=True)
    _bfm  = max(results, key=lambda m: results[m]["f1"])
    _bfma = max(results, key=lambda m: results[m]["auc_roc"])
    _bfcv = max(results, key=lambda m: results[m]["cv_f1_mean"])
    _f1v  = results[_bfm]["f1"]
    _aucv = results[_bfma]["auc_roc"]
    _cvm  = results[_bfcv]["cv_f1_mean"]
    _cvs  = results[_bfcv]["cv_f1_std"]
    st.markdown(
        f"Seven models were trained and evaluated: Logistic Regression (baseline) plus LASSO and Ridge "
        f"regularized variants, a CART decision tree, Random Forest, LightGBM gradient boosting, an MLP "
        f"neural network, and a GAM (agTrend-inspired Generalized Additive Model). "
        f"All use the same 70/30 stratified train/test split with random_state=42.\n\n"
        f"**Best overall model: {_bfm}** -- F1 = {_f1v:.3f}, AUC-ROC = {_aucv:.3f}. "
        f"**Most cross-validation stable: {_bfcv}** (CV F1 = {_cvm:.3f} \u00b1 {_cvs:.3f}).\n\n"
        f"Tree-based ensembles consistently outperform linear models, confirming non-linear interactions "
        f"between DPS membership, region, and population trajectory that a linear boundary cannot capture. "
        f"All models substantially beat the naive 52.7% accuracy baseline.\n\n"
        f"**Was this surprising?** Partially. Logistic Regression achieved AUC = 0.986 -- unexpectedly strong. "
        f"This suggests the engineered features (`pct_decline_from_peak`, `recent_mean`) create a nearly "
        f"linearly separable problem. Domain-specific feature engineering mattered more than model complexity.\n\n"
        f"**Key trade-offs:** Interpretability vs. performance is the main axis. Logistic Regression and LASSO "
        f"offer direct coefficient interpretation; tree models require SHAP. The GAM provides smooth "
        f"interpretable functions most aligned with NOAA's agTrend methodology. For training time: "
        f"Logistic Regression < 1s; LightGBM GridSearch ~30s; MLP ~2 min; GAM ~5 min. "
        f"LightGBM offers the best balance: near-perfect performance, fast retraining, and SHAP explainability. "
        f"The MLP underperforms because 512 sites is too small for deep learning -- gradient boosting's "
        f"inductive bias suits this dataset better.\n\n"
        f"*Full metrics, ROC curves, and per-model details are shown below.*"
    )
    st.markdown("---")
    # -- 2.7 Discussion at top -------------------------------------------------
    st.markdown('<div class="section-header">2.7 · Model Comparison -- Summary & Discussion</div>', unsafe_allow_html=True)
    _bfm2  = max(results, key=lambda m: results[m]['f1'])
    _bfma2 = max(results, key=lambda m: results[m]['auc_roc'])
    _bfcv2 = max(results, key=lambda m: results[m]['cv_f1_mean'])
    st.markdown(
        f'Seven models were trained: Logistic Regression (+ LASSO, Ridge), CART, Random Forest, '
        f'LightGBM, MLP, and GAM. All use 70/30 stratified split, random_state=42.\n\n'
        f'**Best F1: {_bfm2}** ({results[_bfm2]["f1"]:.3f}) | '
        f'**Best AUC: {_bfma2}** ({results[_bfma2]["auc_roc"]:.3f}) | '
        f'**Most stable CV: {_bfcv2}** ({results[_bfcv2]["cv_f1_mean"]:.3f} +/- {results[_bfcv2]["cv_f1_std"]:.3f})\n\n'
        f'Tree ensembles outperform linear models, confirming non-linear interactions between DPS, '
        f'region, and trajectory. All models beat the 52.7% naive baseline substantially.\n\n'
        f'**Surprising:** Logistic Regression achieved AUC=0.986 -- nearly as strong as tree models. '
        f'This means engineered features like pct_decline_from_peak and recent_mean create a nearly '
        f'linearly separable problem. Feature engineering mattered more than model choice.\n\n'
        f'**Trade-offs:** Interpretability vs. performance is the main axis. Logistic/LASSO offer direct '
        f'coefficient interpretation; trees need SHAP. LightGBM offers the best balance: near-perfect '
        f'performance + fast retraining + SHAP explainability. MLP underperforms because 512 sites is too '
        f'small for deep learning. GAM is most ecologically aligned (matches agTrend methodology).\n\n'
        f'*Full metrics, ROC curves, and per-model details are shown below.*'
    )
    st.markdown('---')

    st.markdown(
        'All models were trained on a **70/30 stratified train/test split** (random_state=42) '
        'using 31 engineered features. Numerical features scaled with StandardScaler (training data only). '
        'Categorical features one-hot encoded. Missing values imputed with feature medians. '
        'Hyperparameter tuning via **5-fold stratified CV with GridSearchCV** (scoring=F1). '
        'Final metrics evaluated on held-out test set only.'
    )
    st.caption("All models evaluated on the same 30% held-out test set. CV F1 is the mean ± std from 5-fold cross-validation on the training set.")
    rows = []
    for m in MODEL_NAMES:
        if m not in results: continue
        r = results[m]
        rows.append({"Model":m,"Accuracy":r["accuracy"],"Precision":r["precision"],
                     "Recall":r["recall"],"F1":r["f1"],"AUC-ROC":r["auc_roc"],
                     "CV F1 (mean)":r["cv_f1_mean"],"CV F1 (±std)":r["cv_f1_std"]})
    mdf = pd.DataFrame(rows)
    st.dataframe(mdf.set_index("Model").style
                 .background_gradient(cmap="Blues", subset=["AUC-ROC","F1"])
                 .format("{:.4f}"), use_container_width=True)

    st.markdown("---")

    # -- F1 / AUC bar charts ---------------------------------------------------
    st.markdown('<div class="section-header">Figure 1 -- Key Metric Comparison Bar Charts (Section 2.7)</div>', unsafe_allow_html=True)
    st.caption("F1 score (left) and AUC-ROC (right) for all models on the held-out test set. Higher is better. Dashed line at 1.0 = perfect classifier.")
    c1, c2 = st.columns(2)
    for col, metric, title in [(c1,"F1","F1 Score by Model"),(c2,"AUC-ROC","AUC-ROC by Model")]:
        with col:
            fig, ax = plt.subplots(figsize=(7,4.5))
            ms = mdf.sort_values(metric, ascending=True)
            colors = [ROC_COLORS.get(m, OC["mid"]) for m in ms["Model"]]
            bars = ax.barh(ms["Model"], ms[metric], color=colors, edgecolor="white", linewidth=1.5)
            for bar, v in zip(bars, ms[metric]):
                ax.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2,
                        f"{v:.3f}", va="center", fontsize=9, fontweight="bold")
            ax.set_xlabel(metric); ax.set_title(title, fontweight="bold")
            ax.set_xlim(0.8, 1.03); ax.axvline(1.0, color="gray", ls="--", alpha=0.4)
            ocean_ax(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

    # -- ROC curves ------------------------------------------------------------
    st.markdown("---")
    st.markdown('<div class="section-header">Figure 2 -- ROC Curves: All Models (Sections 2.4 & 2.5)</div>', unsafe_allow_html=True)
    st.caption("Receiver Operating Characteristic curves. Each curve shows the trade-off between true positive rate (sensitivity) and false positive rate at every decision threshold. Higher AUC = better overall discrimination. The dashed diagonal represents a random classifier (AUC=0.5).")
    fig, ax = plt.subplots(figsize=(8,6))
    for m in MODEL_NAMES:
        if m not in results: continue
        r = results[m]
        ax.plot(r["roc_fpr"], r["roc_tpr"], label=f"{m} (AUC={r['auc_roc']:.3f})",
                color=ROC_COLORS.get(m, OC["mid"]), lw=2, alpha=0.85)
    ax.plot([0,1],[0,1],"k--",alpha=0.35,label="Random (0.5)")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves -- All Models", fontweight="bold", fontsize=13)
    ax.legend(loc="lower right", fontsize=8.5); ocean_ax(ax)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown('<div class="insight-box">\nRandom Forest achieves the highest AUC-ROC (0.995), meaning it nearly perfectly ranks declining sites above stable ones.\nLightGBM achieves the highest F1 (0.988), indicating the best balance between precision and recall on the actual binary predictions.\nLinear models (Logistic Regression, LASSO, Ridge) all achieve AUC ~0.986 -- strong but clearly below the tree ensembles,\nconfirming that non-linear feature interactions exist that linear decision boundaries cannot capture.\nThe MLP underperforms tree models despite its theoretical flexibility, likely due to the small dataset size (~512 sites)\nwhere gradient boosting\'s inductive bias toward tree structure outperforms deep learning.\n</div>', unsafe_allow_html=True)

    # -- Per-model detail -------------------------------------------------------
    st.markdown("---")
    st.markdown('<div class="section-header">Per-Model Detail -- Hyperparameters & Confusion Matrix</div>', unsafe_allow_html=True)
    st.caption("Select a model to see its confusion matrix, cross-validation results, best hyperparameters from GridSearchCV, and architecture details.")
    sel = st.selectbox("Select model to inspect:", [m for m in MODEL_NAMES if m in results])
    if sel in results:
        r = results[sel]
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Accuracy",  f"{r['accuracy']:.3f}")
        c2.metric("Precision", f"{r['precision']:.3f}")
        c3.metric("Recall",    f"{r['recall']:.3f}")
        c4.metric("F1",        f"{r['f1']:.3f}")
        c5.metric("AUC-ROC",   f"{r['auc_roc']:.3f}")
        ca, cb = st.columns(2)
        with ca:
            cm_arr = np.array(r["confusion_matrix"])
            fig, ax = plt.subplots(figsize=(4.5,3.5))
            ConfusionMatrixDisplay(cm_arr, display_labels=["Stable","Declining"]).plot(
                ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(f"Confusion Matrix -- {sel}", fontweight="bold")
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with cb:
            st.markdown(f"""#### 5-Fold CV (F1)
| Metric | Value |
|---|---|
| Mean F1 | **{r['cv_f1_mean']:.4f}** |
| Std Dev | {r['cv_f1_std']:.4f} |""")
            if "best_params" in r:
                st.markdown("#### Best Hyperparameters")
                for k,v in r["best_params"].items():
                    st.write(f"- `{k}`: **{v}**")
        if sel == "CART":
            cart_model = load_model("CART")
            if cart_model:
                st.markdown("#### Decision Tree (depth ≤ 3)")
                fig, ax = plt.subplots(figsize=(16,6))
                plot_tree(cart_model, feature_names=feature_names,
                          class_names=["Stable","Declining"], filled=True,
                          rounded=True, max_depth=3, ax=ax, fontsize=7,
                          impurity=False, proportion=True)
                plt.tight_layout(); st.pyplot(fig); plt.close()
        if sel == "MLP":
            st.markdown("#### Training History")
            fig, axes = plt.subplots(1, 2 if mlp_hist["val"] else 1,
                                     figsize=(10 if mlp_hist["val"] else 5, 4))
            if not isinstance(axes, np.ndarray): axes = [axes]
            axes[0].plot(mlp_hist["loss"], color=OC["dark"], lw=2)
            axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch"); ocean_ax(axes[0])
            if mlp_hist["val"] and len(axes)>1:
                axes[1].plot(mlp_hist["val"], color=OC["mid"], lw=2)
                axes[1].set_title("Validation Accuracy"); axes[1].set_xlabel("Epoch"); ocean_ax(axes[1])
            plt.tight_layout(); st.pyplot(fig); plt.close()



# -----------------------------------------------------------------------------
# TAB 4 -- SHAP & INTERACTIVE PREDICTION
# -----------------------------------------------------------------------------
with tabs[4]:
    st.markdown("## 🔍 Explainability & Interactive Prediction")
    rf_model = load_model("Random Forest")

    # -- Feature importance (RF built-in) --------------------------------------
    st.markdown('<div class="section-header">Feature Importance -- Random Forest</div>', unsafe_allow_html=True)
    if rf_model is not None:
        importances = rf_model.feature_importances_
        fi_df = pd.DataFrame({"Feature":feature_names,"Importance":importances})                  .sort_values("Importance", ascending=False).head(15)
        c1, c2 = st.columns([1.3,1])
        with c1:
            fig, ax = plt.subplots(figsize=(7,5))
            n = len(fi_df)
            colors = [plt.cm.Blues(0.35 + 0.6*i/max(n-1,1)) for i in range(n)][::-1]
            ax.barh(fi_df["Feature"][::-1], fi_df["Importance"][::-1],
                    color=colors, edgecolor="white", linewidth=0.8)
            ax.set_xlabel("Mean Decrease in Impurity")
            ax.set_title("Top 15 Feature Importances (Random Forest)", fontweight="bold")
            ocean_ax(ax); plt.tight_layout(); st.pyplot(fig); plt.close()
        with c2:
            st.dataframe(fi_df.set_index("Feature").style.format("{:.4f}")
                         .background_gradient(cmap="Blues"), use_container_width=True)
        st.markdown('<div class="insight-box">\n<b>pct_decline_from_peak</b> and <b>recent_mean</b> dominate -- sites with large historical losses\nand low recent counts are almost certain to be classified as Declining. <b>DPS membership</b>\nreflects the fundamental WDPS/EDPS biological split: WDPS sites are endangered by definition.\n<b>r_squared</b> captures trend reliability -- a consistent downward slope strengthens the Declining signal.\nThese results align with ecological expectation: population trajectory and current abundance\nare the most direct indicators of conservation concern.\n</div>', unsafe_allow_html=True)

    # -- SHAP Analysis ---------------------------------------------------------
    st.markdown("---")
    st.markdown('<div class="section-header">SHAP Analysis (TreeExplainer -- Random Forest)</div>', unsafe_allow_html=True)
    st.markdown('SHAP (SHapley Additive exPlanations) assigns each feature a contribution to each individual prediction,\ngrounded in cooperative game theory. Unlike feature importance, SHAP shows both **magnitude** and\n**direction** of impact for every feature on every prediction.')

    sd = shap_data
    if sd.get("shap_values") is not None:
        try:
            import shap as shap_lib

            shap_vals   = sd["shap_values"]         # shape (n_test, n_features)
            X_test_arr  = sd["X_test"]               # shape (n_test, n_features)
            feat_names  = sd["feature_names"]
            exp_val     = sd.get("expected_value", 0)

            # Plot 1 & 2: Beeswarm + Bar side by side
            import io
            col_sh1, col_sh2 = st.columns(2)
            with col_sh1:
                st.markdown("**1. Beeswarm -- Feature Impact Direction & Magnitude**")
                st.caption("Each dot = one survey site. Color = feature value (red=high, blue=low). X-axis = SHAP impact on P(Declining).")
                # Use savefig to buffer to avoid SHAP messing with global rcParams
                plt.close("all")
                shap_lib.summary_plot(shap_vals, X_test_arr, feature_names=feat_names,
                                      max_display=12, show=False, plot_size=(10, 7))
                buf = io.BytesIO()
                plt.savefig(buf, format="png", dpi=90, bbox_inches="tight", facecolor="#f8fbff")
                buf.seek(0); plt.close("all")
                st.image(buf, use_container_width=True)
                st.markdown('''<div class="insight-box" style="font-size:0.85rem;">
<b>How to read:</b> Dots far right push prediction toward Declining; dots far left push toward Stable.
High <b>pct_decline_from_peak</b> (red dots, far right) most strongly drives Declining predictions -- sites that lost the most relative to their historical peak.
Low <b>recent_mean</b> (blue dots, far right) similarly -- few animals counted recently almost always indicates decline.
WDPS membership (red = WDPS) consistently pushes toward Declining, reflecting the endangered status of the western population.
</div>''', unsafe_allow_html=True)

            with col_sh2:
                st.markdown("**2. Bar Plot -- Mean Absolute SHAP Values**")
                st.caption("Average magnitude of each feature's SHAP impact across all test predictions, regardless of direction.")
                plt.close("all")
                shap_lib.summary_plot(shap_vals, X_test_arr, feature_names=feat_names,
                                      plot_type="bar", max_display=12, show=False, plot_size=(10, 7))
                buf2 = io.BytesIO()
                plt.savefig(buf2, format="png", dpi=90, bbox_inches="tight", facecolor="#f8fbff")
                buf2.seek(0); plt.close("all")
                st.image(buf2, use_container_width=True)
                st.markdown('''<div class="insight-box" style="font-size:0.85rem;">
<b>How to read:</b> Longer bar = higher average absolute SHAP value = more impactful feature overall.
This ranks features by total predictive contribution regardless of direction, complementing the beeswarm.
<b>pct_decline_from_peak</b> and <b>recent_mean</b> dominate both plots, validating the RF built-in importance
rankings with a more rigorous game-theory-based measure.
</div>''', unsafe_allow_html=True)

            # Plot 3: Waterfall for a specific interesting case
            st.markdown("---")
            st.markdown("**3. Waterfall Plot -- Single Prediction Explanation**")
            st.caption("Shows exactly how each feature pushes a specific prediction above or below the baseline.")

            # Find most interesting cases: highest confidence Declining and a borderline case
            pred_probs_test = rf_model.predict_proba(X_test_arr)[:,1] if rf_model else None
            wf_options = {}
            if pred_probs_test is not None:
                wf_options["Highest-confidence Declining (most certain)"] = int(np.argmax(pred_probs_test))
                wf_options["Borderline case (closest to 50/50)"]          = int(np.argmin(np.abs(pred_probs_test - 0.5)))
                wf_options["Highest-confidence Stable (most certain)"]    = int(np.argmin(pred_probs_test))

            wf_sel = st.selectbox("Select prediction to explain:", list(wf_options.keys()))
            wf_idx = wf_options[wf_sel]

            # Ensure numpy arrays for safe integer indexing
            import io as _io_wf
            sv_arr = np.array(shap_vals)
            if sv_arr.ndim == 3: sv_arr = sv_arr[:, :, 1]  # 3D -> class 1
            xt_arr = X_test_arr.values if hasattr(X_test_arr, 'values') else np.array(X_test_arr)
            explanation = shap_lib.Explanation(
                values        = sv_arr[wf_idx],
                base_values   = float(exp_val),
                data          = xt_arr[wf_idx],
                feature_names = feat_names,
            )
            plt.close('all')
            shap_lib.plots.waterfall(explanation, max_display=15, show=False)
            buf_wf = _io_wf.BytesIO()
            plt.savefig(buf_wf, format='png', dpi=110, bbox_inches='tight', facecolor='#f8fbff')
            buf_wf.seek(0); plt.close('all')
            st.image(buf_wf, use_container_width=True)
            if pred_probs_test is not None:
                p = pred_probs_test[wf_idx]
                st.markdown(f"""<div class="insight-box">
<b>This site's prediction:</b> P(Declining) = <b>{p:.3f}</b> -- {"🔴 Declining" if p>0.5 else "🟢 Stable"}<br><br>
The waterfall starts at the <b>base rate</b> (E[f(x)] = average prediction across all sites).
Each bar shows how much one feature pushes the prediction up (red = toward Declining) or down (blue = toward Stable).
The final prediction is the sum of all contributions plus the base rate.
Features above their average push toward Declining; features below their average push toward Stable.
</div>""", unsafe_allow_html=True)


        except Exception as e:
            st.warning(f"SHAP plot error: {e}")
            import traceback; st.code(traceback.format_exc())
    else:
        st.info("ℹ️ Run `pip install shap` then `python train.py` to enable SHAP plots.")

    # -- Logistic coefficients -------------------------------------------------
    st.markdown("---")
    st.markdown('<div class="section-header">Logistic Regression Coefficients</div>', unsafe_allow_html=True)
    st.markdown('For the linear baseline, coefficients directly show feature impact direction and magnitude\n(after scaling). Positive coefficient → increases P(Declining); negative → decreases it.\nThis complements SHAP by giving a globally interpretable linear view of the same features.')
    lr_model = load_model("Logistic Regression")
    if lr_model is not None:
        coefs = lr_model.coef_[0]
        cdf = pd.DataFrame({"Feature":feature_names,"Coefficient":coefs})                .sort_values("Coefficient", key=abs, ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(8,5))
        bar_c = [OC["decline"] if c>0 else OC["mid"] for c in cdf["Coefficient"][::-1]]
        ax.barh(cdf["Feature"][::-1], cdf["Coefficient"][::-1], color=bar_c, edgecolor="white", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Coefficient Value (features scaled to unit variance)")
        ax.set_title("Logistic Regression -- Top 15 Feature Coefficients", fontweight="bold")
        ocean_ax(ax); plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('<div class="insight-box">\n<b>Red bars</b> (positive) → increase P(Declining). <b>Blue bars</b> (negative) → decrease it (associated with Stable).\n<b>pct_decline_from_peak</b> is the strongest positive predictor -- large historical losses are the clearest signal of a declining site.\n<b>WDPS membership</b> is a strong positive predictor, while <b>recent_mean</b> is strongly negative (more animals recently = more likely Stable).\nLogistic coefficients agree closely with SHAP rankings, validating that the non-linear models are capturing the same fundamental signal.\n</div>', unsafe_allow_html=True)

    # -- Interactive prediction ------------------------------------------------
    st.markdown("---")
    st.markdown('<div class="section-header">🎛️ Interactive Prediction + SHAP Waterfall</div>', unsafe_allow_html=True)
    st.markdown("Adjust feature values to get a real-time prediction. The SHAP waterfall shows exactly **why** the model makes that prediction.")

    X_full = joblib.load(os.path.join(MODEL_DIR, "X_full.pkl"))
    medians = X_full.median()

    s1, s2, s3 = st.columns(3)
    with s1:
        rec_mean_v = st.slider("Recent Mean Count",      0,   int(df["recent_mean"].quantile(0.99)), 500, step=10)
        pct_dec_v  = st.slider("% Decline from Peak",    0.0, 100.0, 50.0, step=1.0)
        cv_v       = st.slider("Coefficient of Variation",0.0, 3.0,  0.5,  step=0.05)
    with s2:
        n_yr_v     = st.slider("N Survey Years",         1,   50,    20)
        r2_v       = st.slider("Trend R²",               0.0, 1.0,   0.5,  step=0.01)
        peak_v     = st.slider("Years Since Peak",       0,   50,    20)
    with s3:
        dps_v      = st.selectbox("Population Segment", ["WDPS","EDPS"])
        reg_v      = st.selectbox("Region", sorted(df["region"].unique()))
        ctype_v    = st.selectbox("Count Type", ["Non-Pup","Pup"])
        model_ch   = st.selectbox("Model", MODEL_NAMES)

    Xc = medians.copy()
    Xc["recent_mean"]           = rec_mean_v
    Xc["pct_decline_from_peak"] = pct_dec_v
    Xc["cv"]                    = cv_v
    Xc["n_years"]               = n_yr_v
    Xc["r_squared"]             = r2_v
    Xc["years_since_peak"]      = peak_v
    Xc["early_mean"]            = rec_mean_v * (1 + pct_dec_v/100)
    for col in X_full.columns:
        if col.startswith(("dps_","count_type_","region_")): Xc[col] = 0
    for k,v in [(f"dps_{dps_v}",1),(f"count_type_{ctype_v}",1),(f"region_{reg_v}",1)]:
        if k in Xc.index: Xc[k] = v
    Xvec = Xc.values.reshape(1,-1)

    pm = load_model(model_ch)
    if pm is not None:
        needs_sc  = model_ch in ["Logistic Regression","LASSO","Ridge","MLP","GAM"]
        Xi        = scaler.transform(Xvec) if needs_sc else Xvec
        pred_cls  = pm.predict(Xi)[0]
        pred_prob = pm.predict_proba(Xi)[0]
        outcome   = "🔴  DECLINING" if pred_cls==1 else "🟢  STABLE / RECOVERING"

        r1,r2,r3 = st.columns(3)
        r1.metric("Prediction",   outcome)
        r2.metric("P(Declining)", f"{pred_prob[1]:.3f}")
        r3.metric("P(Stable)",    f"{pred_prob[0]:.3f}")

        fig, ax = plt.subplots(figsize=(7,2))
        ax.barh(["Declining"],[pred_prob[1]], color=OC["decline"], height=0.45)
        ax.barh(["Stable"],   [pred_prob[0]], color=OC["mid"],     height=0.45)
        ax.set_xlim(0,1); ax.axvline(0.5, color="gray", ls="--", alpha=0.5, label="Decision boundary")
        ax.set_xlabel("Probability"); ax.set_title(f"Prediction -- {model_ch}", fontweight="bold")
        ocean_ax(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

        # SHAP waterfall for custom input (RF only)
        st.markdown("#### SHAP Waterfall -- Why this prediction?")
        if rf_model is not None and sd.get("shap_values") is not None:
            try:
                import shap as shap_lib
                explainer_live = shap_lib.TreeExplainer(rf_model)
                shap_vals_live = explainer_live.shap_values(Xvec)
                # Handle both list-of-arrays and 3D array formats
                if isinstance(shap_vals_live, list):
                    sv_live = np.array(shap_vals_live[1]).flatten()
                    ev_live = explainer_live.expected_value[1] if hasattr(explainer_live.expected_value, "__len__") else explainer_live.expected_value
                elif hasattr(shap_vals_live, "ndim") and shap_vals_live.ndim == 3:
                    # shape (1, n_features, 2) -- take class 1
                    sv_live = shap_vals_live[0, :, 1]
                    ev_live = explainer_live.expected_value[1] if hasattr(explainer_live.expected_value, "__len__") else explainer_live.expected_value
                elif hasattr(shap_vals_live, "ndim") and shap_vals_live.ndim == 2:
                    # shape (1, n_features) -- already class 1
                    sv_live = shap_vals_live[0]
                    ev_live = explainer_live.expected_value if not hasattr(explainer_live.expected_value, "__len__") else explainer_live.expected_value[1]
                else:
                    sv_live = np.array(shap_vals_live).flatten()
                    ev_live = float(np.array(explainer_live.expected_value).flat[-1])

                exp_live = shap_lib.Explanation(
                    values        = sv_live.flatten(),
                    base_values   = float(ev_live),
                    data          = Xvec[0],
                    feature_names = list(X_full.columns),
                )
                plt.close("all")
                shap_lib.plots.waterfall(exp_live, max_display=15, show=False)
                import io as _io2
                buf_live = _io2.BytesIO()
                plt.savefig(buf_live, format="png", dpi=110, bbox_inches="tight", facecolor="#f8fbff")
                buf_live.seek(0); plt.close("all")
                st.image(buf_live, use_container_width=True)
                st.markdown('<div class="insight-box" style="font-size:0.85rem;">\nThe waterfall shows <b>your specific input\'s</b> SHAP explanation. Red bars push toward Declining,\nblue bars push toward Stable. Try sliding <b>% Decline from Peak</b> to 90% and watch which features\ndrive the prediction -- the waterfall updates in real time.\n</div>', unsafe_allow_html=True)
            except Exception as e:
                st.info(f"Live SHAP waterfall unavailable: {e}")
        else:
            st.info("Live SHAP waterfall requires Random Forest model and SHAP data.")

# TAB 5 -- agTrend & FORECAST
# -----------------------------------------------------------------------------
with tabs[5]:
    st.markdown("## 📈 agTrend Analysis & Population Forecast")
    st.markdown('The `agTrend.ssl` R package (Johnson et al., NOAA/NMFS) fits **hierarchical Generalized Additive Models (GAMs)**\nto impute missing survey years and estimate regional log-linear population trends.\nHere we replicate this methodology in Python using PCHIP interpolation + linear trend estimation,\nthen **extrapolate trends to 2040** to forecast future population trajectories.\n\n> ⚠️ *Forecasts are extrapolations of recent trends and carry significant uncertainty -- they assume\n> current pressures (fishing, climate, prey availability) continue unchanged. Treat as scenario\n> projections, not predictions.*')
    st.markdown("---")

    # -- Regional trend rates (agTrend-style table) ----------------------------
    st.markdown('<div class="section-header">Regional Trend Rates (agTrend-style)</div>', unsafe_allow_html=True)
    st.caption("Annual log-linear growth rate (% per year) estimated from 2000–2023 observed counts, matching agTrend.ssl output format.")

    trend_rows = []
    for dps in ["WDPS","EDPS"]:
        for region in sorted(df_long[df_long["dps"]==dps]["region"].unique()):
            sub = df_long[(df_long["region"]==region) & (df_long["is_observed"]) & (df_long["year"]>=2000)]
            if len(sub) < 4: continue
            yr_obs = sub.groupby("year")["count"].sum()
            yrs = yr_obs.index.values.astype(float)
            cts = yr_obs.values.astype(float)
            cts_pos = cts[cts > 0]
            yrs_pos = yrs[cts > 0]
            if len(yrs_pos) < 3: continue
            log_cts = np.log(cts_pos + 1)
            slope, intercept, r, p, se = stats.linregress(yrs_pos, log_cts)
            pct_yr = (np.exp(slope) - 1) * 100
            ci_lo  = (np.exp(slope - 1.96*se) - 1) * 100
            ci_hi  = (np.exp(slope + 1.96*se) - 1) * 100
            trend_rows.append({
                "DPS":dps,"Region":region,
                "Est. % / yr":round(pct_yr,2),
                "95% CI lower":round(ci_lo,2),
                "95% CI upper":round(ci_hi,2),
                "R²":round(r**2,3),"p-value":round(p,4),
                "N surveys":len(yrs_pos),
            })

    tdf = pd.DataFrame(trend_rows)

    def color_rate(v):
        if v < -3:  return "background-color:#ffe0e0;color:#8b0000"
        if v < 0:   return "background-color:#fff0f0;color:#c0392b"
        if v > 3:   return "background-color:#e0ffe0;color:#1a5c1a"
        return "background-color:#f0fff0;color:#1a7a4a"

    st.dataframe(
        tdf.set_index("Region").style
           .applymap(color_rate, subset=["Est. % / yr"])
           .format({"Est. % / yr":"{:+.2f}%","95% CI lower":"{:+.2f}%",
                    "95% CI upper":"{:+.2f}%","R²":"{:.3f}","p-value":"{:.4f}"}),
        use_container_width=True
    )
    st.markdown('<div class="insight-box">\n<b>Red rows</b> = strong decline (&lt;−3%/yr). <b>Light red</b> = moderate decline. <b>Green</b> = recovery.\nW ALEU typically shows −5 to −8%/yr -- among the steepest documented declines for any marine mammal population.\nCompare to agTrend.ssl output: W ALEU predicted −7.06%/yr (2000–2019) with 95% CI [−7.78, −6.26].\n</div>', unsafe_allow_html=True)

    st.markdown("---")

    # -- Population Forecast to 2040 -------------------------------------------
    st.markdown('<div class="section-header">Population Forecast to 2040</div>', unsafe_allow_html=True)
    FORECAST_YR = 2040
    THRESHOLD   = 0.5  # 50% of 2000 baseline = "critical" threshold

    forecast_tab = st.radio("Select DPS to forecast:", ["WDPS", "EDPS"], horizontal=True)
    regions_fc = sorted(df_long[df_long["dps"]==forecast_tab]["region"].unique())
    pal_fc = WDPS_C[:len(regions_fc)] if forecast_tab=="WDPS" else EDPS_C[:len(regions_fc)]

    fig, axes = plt.subplots(2, 3, figsize=(16,10))
    axes_flat = axes.flat
    crit_table = []

    for ax, region, color in zip(axes_flat, regions_fc, pal_fc):
        sub_all = df_long[(df_long["region"]==region)]
        sub_obs = sub_all[sub_all["is_observed"]]
        yr_all  = sub_all.groupby("year")["count"].sum()
        yr_obs  = sub_obs.groupby("year")["count"].sum()

        # Fit trend on recent 20 years of observed data
        fit_yrs  = yr_obs[yr_obs.index >= 2000]
        if len(fit_yrs) < 3:
            fit_yrs = yr_obs

        yrs_fit  = fit_yrs.index.values.astype(float)
        cts_fit  = fit_yrs.values.astype(float)
        mask     = cts_fit > 0
        if mask.sum() < 2:
            ax.set_visible(False); continue

        log_cts  = np.log(cts_fit[mask] + 1)
        sl, ic, rv, pv, se = stats.linregress(yrs_fit[mask], log_cts)

        # Extrapolate
        future_yrs = np.arange(yr_obs.index.max()+1, FORECAST_YR+1)
        all_yrs_fc = np.concatenate([yrs_fit[mask], future_yrs])

        pred_log   = ic + sl * all_yrs_fc
        pred_cts   = np.expm1(np.clip(pred_log, 0, None))
        # 95% CI bands using SE of slope
        pred_lo    = np.expm1(np.clip((ic - 1.96*se*(all_yrs_fc - yrs_fit[mask].mean())) + sl*all_yrs_fc, 0, None))
        pred_hi    = np.expm1((ic + 1.96*se*np.abs(all_yrs_fc - yrs_fit[mask].mean())) + sl*all_yrs_fc)

        # Critical threshold = 50% of 2000 baseline
        baseline_2000 = yr_obs[yr_obs.index <= 2005].mean() if len(yr_obs[yr_obs.index<=2005])>0 else yr_obs.iloc[0]
        critical_lvl  = baseline_2000 * THRESHOLD

        # Plot historical
        ax.fill_between(yr_all.index, yr_all.values/1000, alpha=0.12, color=color)
        ax.plot(yr_all.index, yr_all.values/1000, color=color, lw=2, alpha=0.7, label="Historical")
        ax.scatter(yr_obs.index, yr_obs.values/1000, color=color, s=20,
                   edgecolors="white", linewidths=0.6, zorder=4)

        # Plot forecast
        fc_start = yr_obs.index.max()
        fc_mask  = all_yrs_fc >= fc_start
        ax.plot(all_yrs_fc[fc_mask], pred_cts[fc_mask]/1000,
                color=color, lw=2, ls="--", label="Forecast", zorder=5)
        ax.fill_between(all_yrs_fc[fc_mask],
                        np.clip(pred_lo[fc_mask], 0, None)/1000,
                        pred_hi[fc_mask]/1000,
                        alpha=0.15, color=color)

        # Critical threshold line
        ax.axhline(critical_lvl/1000, color="#c0392b", ls=":", lw=1.4, alpha=0.7, label="50% baseline")

        # Shade forecast zone
        ax.axvspan(fc_start, FORECAST_YR, alpha=0.05, color=color)
        ax.axvline(fc_start, color="gray", ls=":", lw=0.8, alpha=0.6)

        # Annotate projected 2040 value
        final_pred = pred_cts[-1]
        ax.annotate(f"2040: {final_pred/1000:.1f}k",
                    xy=(FORECAST_YR, final_pred/1000),
                    xytext=(-40, 10), textcoords="offset points",
                    fontsize=7.5, color=color, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

        ax.set_title(f"{region}", fontweight="bold", color=OC["dark"], fontsize=10)
        ax.set_xlabel("Year"); ax.set_ylabel("Count (thousands)")
        ax.legend(fontsize=7); ocean_ax(ax)

        # Critical year estimate
        for i, (yr, ct) in enumerate(zip(all_yrs_fc, pred_cts)):
            if ct <= critical_lvl and yr > 2024:
                crit_table.append({"Region":region,"Est. Critical Year":int(yr),
                                   "2040 Projected":f"{final_pred/1000:.1f}k",
                                   "Trend (%/yr)":f"{(np.exp(sl)-1)*100:+.2f}%"})
                break

    for ax in list(axes_flat)[len(regions_fc):]:
        ax.set_visible(False)

    plt.suptitle(f"{forecast_tab} Population Forecast to {FORECAST_YR}",
                 fontsize=14, fontweight="bold", color=OC["dark"], y=1.01)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="forecast-box">\n<b>How to read this chart:</b> Solid lines = historical interpolated data, dashed = linear trend extrapolation,\nshaded band = 95% confidence interval, dotted red line = 50% of 2000–2005 baseline (critical threshold).\nThe gray shaded region marks the forecast period. Linear extrapolation is the simplest assumption --\nactual populations could recover faster if pressures ease or decline faster under climate stress.\n</div>', unsafe_allow_html=True)

    if crit_table:
        st.markdown("#### ⚠️ Projected Critical Threshold Dates")
        st.caption("Regions projected to fall below 50% of their 2000–2005 baseline count at current trend rates:")
        ct_df = pd.DataFrame(crit_table)
        st.dataframe(ct_df.set_index("Region").style
                     .applymap(lambda v: "background-color:#ffe0e0;color:#8b0000"
                               if isinstance(v,int) and v < 2035 else ""),
                     use_container_width=True)

    st.markdown("---")

    # GAM smooth fit visualization
    st.markdown('<div class="section-header">GAM Smooth Trend Fits (LogisticGAM / agTrend-inspired)</div>', unsafe_allow_html=True)
    gam_model = load_model("GAM")
    if gam_model is not None:
        st.markdown('The **GAM model** (LogisticGAM with penalized splines) trained on site-level features\nmirrors the agTrend.ssl approach: instead of assuming linear feature effects, it fits\nsmooth non-parametric functions -- capturing the curved relationship between, e.g.,\n*years since peak* and decline probability without overfitting.')
        st.markdown(f"""<div class="forecast-box">
GAM test-set metrics: F1 = {results.get('GAM',{}).get('f1','N/A')},
AUC-ROC = {results.get('GAM',{}).get('auc_roc','N/A')},
CV F1 = {results.get('GAM',{}).get('cv_f1_mean','N/A')} ±{results.get('GAM',{}).get('cv_f1_std','N/A')}
<br><br>
Compared to agTrend.ssl's mgcv GAM, this Python LogisticGAM uses the same penalized smoothing
philosophy but on classification features rather than raw counts. The native agTrend.ssl approach
(R-based) uses a Tweedie distribution for count imputation then log-linear trend estimation --
that raw-count functionality is captured in the Forecast section above.
</div>""", unsafe_allow_html=True)
    else:
        st.info("""GAM model not found. To enable:
1. In PowerShell with venv active: `pip install pygam`
2. Re-run: `python train.py`
3. Restart: `streamlit run app.py`
The LogisticGAM model will then appear here alongside the agTrend comparison.""")

# -----------------------------------------------------------------------------
# TAB 6 -- CLUSTERING
# -----------------------------------------------------------------------------
with tabs[6]:
    st.markdown("## 🗂️ Clustering Analysis")
    st.markdown('KMeans (k=4) clustering discovers **natural groupings** of survey sites without using the\ndecline label. Combined with PCA for visualization, it reveals whether the machine learning\ntarget aligns with organic ecological groupings -- and it does.')
    X_pca  = cluster["X_pca"]
    labels = cluster["cluster_labels"]
    ev     = cluster["explained_variance"]

    c1, c2 = st.columns([1.3, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(7,6))
        for k, color in enumerate(BLUES[:4]):
            mask = labels==k
            ax.scatter(X_pca[mask,0], X_pca[mask,1], color=color,
                       label=f"Cluster {k+1}", alpha=0.75, s=50,
                       edgecolors="white", linewidth=0.5)
        dec_mask = df["is_declining"].values.astype(bool)
        ax.scatter(X_pca[dec_mask,0], X_pca[dec_mask,1],
                   marker="x", color=OC["decline"], s=28, alpha=0.35, label="Declining (×)")
        ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% var)")
        ax.set_title("KMeans Clusters (k=4) in PCA Space", fontweight="bold")
        ax.legend(fontsize=9); ocean_ax(ax)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with c2:
        df_cl = df.copy(); df_cl["cluster"] = labels
        profile = df_cl.groupby("cluster").agg({
            "is_declining":"mean","mean_count":"median",
            "recent_mean":"median","pct_decline_from_peak":"median",
            "dps": lambda x: x.value_counts().index[0],
        }).round(3)
        profile.index = [f"Cluster {i+1}" for i in profile.index]
        profile.columns = ["% Declining","Median Count","Recent Mean","% Peak Decline","Top DPS"]
        st.dataframe(profile.style.background_gradient(cmap="Blues", subset=["% Declining"]),
                     use_container_width=True)
        st.markdown('<div class="insight-box">\nClustering aligns closely with the biological DPS split. WDPS sites naturally cluster into high-decline groups;\nEDPS sites cluster separately. This unsupervised finding validates the supervised classification: the DPS\ndivision reflects genuinely different ecological trajectories, not just a regulatory label.\n</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Cluster Composition</div>', unsafe_allow_html=True)
    df_cl["cluster_label"] = df_cl["cluster"].map({i:f"Cluster {i+1}" for i in range(4)})
    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    pd.crosstab(df_cl["cluster_label"], df_cl["dps"]).plot(
        kind="bar", ax=axes[0], color=[OC["mid"], OC["dark"]], edgecolor="white", rot=0)
    axes[0].set_title("DPS by Cluster", fontweight="bold"); axes[0].set_xlabel("")
    ocean_ax(axes[0])
    pd.crosstab(df_cl["cluster_label"], df_cl["region"]).plot(
        kind="bar", ax=axes[1], edgecolor="white", rot=30,
        color=plt.cm.Blues(np.linspace(0.3, 0.9, df["region"].nunique())))
    axes[1].set_title("Region by Cluster", fontweight="bold"); axes[1].set_xlabel("")
    axes[1].legend(fontsize=7, bbox_to_anchor=(1,1)); ocean_ax(axes[1])
    plt.tight_layout(); st.pyplot(fig); plt.close()

