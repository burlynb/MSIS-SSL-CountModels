"""
app.py
Sea Lion Population Trend Classification — Streamlit App
MSIS 522 HW1 | University of Washington

Run: streamlit run app.py
"""

import os, json, warnings, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib
import streamlit as st
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, confusion_matrix
from sklearn.tree import plot_tree
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🦁 Sea Lion ML Dashboard",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "ALLCOUNTS_v21_FED-fish.xlsx")

sys.path.insert(0, BASE_DIR)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(135deg, #1e3a5f 0%, #2d7dd2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        padding-bottom: 0.3rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9ff 0%, #e8f0ff 100%);
        border-left: 4px solid #2d7dd2; border-radius: 8px;
        padding: 1rem 1.2rem; margin: 0.3rem 0;
    }
    .metric-val { font-size: 1.8rem; font-weight: 700; color: #1e3a5f; }
    .metric-label { font-size: 0.85rem; color: #555; }
    .insight-box {
        background: #fff8e7; border-left: 4px solid #f0a500;
        border-radius: 8px; padding: 0.9rem 1.1rem; margin: 0.5rem 0;
        font-size: 0.95rem;
    }
    .section-header {
        font-size: 1.4rem; font-weight: 700; color: #1e3a5f;
        border-bottom: 2px solid #2d7dd2; padding-bottom: 0.3rem; margin: 1.2rem 0 0.8rem 0;
    }
    div[data-testid="stTabs"] > div > div > button {
        font-size: 1.0rem; font-weight: 600; padding: 0.6rem 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = joblib.load(os.path.join(MODEL_DIR, "processed_df.pkl"))
    return df

@st.cache_data
def load_splits():
    return joblib.load(os.path.join(MODEL_DIR, "splits.pkl"))

@st.cache_resource
def load_model(name: str):
    tag = name.replace(" ", "_").lower()
    path = os.path.join(MODEL_DIR, f"{tag}.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None

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
    sys.path.insert(0, BASE_DIR)
    from data_utils import get_long_format
    df = joblib.load(os.path.join(MODEL_DIR, "processed_df.pkl"))
    return get_long_format(df)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f5/Steller_sea_lion_and_California_sea_lion.jpg/320px-Steller_sea_lion_and_California_sea_lion.jpg", use_container_width=True)
    st.markdown("## 🦁 Sea Lion ML Dashboard")
    st.markdown("**MSIS 522 · HW1** | University of Washington")
    st.markdown("---")
    st.markdown("""
**Dataset:** Steller Sea Lion counts across Alaska & Pacific Coast rookeries
(1970–2024).

**Task:** Binary classification — predict whether a survey site shows a
**Declining** population trend.

**Target:** `is_declining = 1` if slope < 0 AND recent counts < 85 % of early counts.
""")
    st.markdown("---")
    st.markdown("*Data source: NOAA / National Marine Fisheries Service*")

# ── Load everything ────────────────────────────────────────────────────────────
df        = load_data()
X_train, X_test, y_train, y_test, feature_names = load_splits()
scaler    = load_scaler()
results   = load_results()
shap_data = load_shap_data()
cluster   = load_clustering()
mlp_hist  = load_mlp_history()
df_long   = load_long_format()

MODEL_NAMES = ["Logistic Regression", "LASSO", "Ridge", "CART", "Random Forest", "LightGBM", "MLP"]

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📋 Executive Summary",
    "📊 Data Visualization",
    "🏆 Model Performance",
    "🔍 SHAP & Interactive Prediction",
    "🗂️ Clustering",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: EXECUTIVE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="main-title">🦁 Steller Sea Lion Population Trend Analysis</div>', unsafe_allow_html=True)
    st.markdown("### Predicting Population Decline Using Machine Learning")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    n_declining   = int(df["is_declining"].sum())
    n_stable      = int((df["is_declining"] == 0).sum())
    n_sites       = len(df)
    best_auc      = max(results[m]["auc_roc"] for m in results)
    best_model    = max(results, key=lambda m: results[m]["auc_roc"])

    with col1:
        st.markdown(f"""<div class="metric-card"><div class="metric-val">{n_sites}</div><div class="metric-label">Survey Sites</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card"><div class="metric-val" style="color:#c0392b">{n_declining}</div><div class="metric-label">Declining Sites ({n_declining/n_sites*100:.0f}%)</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card"><div class="metric-val" style="color:#27ae60">{n_stable}</div><div class="metric-label">Stable / Recovering</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card"><div class="metric-val">{best_auc:.3f}</div><div class="metric-label">Best AUC-ROC ({best_model})</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.markdown("#### 🌊 Dataset Overview")
        st.markdown("""
**Steller sea lions** (*Eumetopias jubatus*) are a keystone predator of the North Pacific
and are critically important to the ecosystem health of Alaska and the US Pacific coast.
Two Distinct Population Segments (DPS) exist:

- **WDPS** (Western, *endangered*): Aleutian Islands and Gulf of Alaska — **severe decline** since the 1970s, particularly in the western Aleutians, where populations have dropped by over 80%.
- **EDPS** (Eastern, *delisted*): British Columbia, Oregon, Washington, California — gradually recovering.

This dataset compiles multi-decade aerial survey counts from NOAA/NMFS at hundreds of
rookeries and haul-out sites across the range.

#### 🎯 Prediction Task
Classify each survey site as **Declining** or **Stable / Recovering** based on
engineered features from the historical count time-series (region, population
segment, mean/peak counts, decade averages, variability, etc.).

This task supports conservation managers in **prioritizing sites for intervention**,
allocating monitoring resources, and forecasting future endangerment risk.
""")

    with col_b:
        st.markdown("#### 🔬 Approach")
        st.markdown("""
| Stage | Method |
|-------|--------|
| **Preprocessing** | Wide-to-long melt, per-site feature engineering |
| **Target** | Binary: Declining vs Stable/Recovering |
| **Baseline** | Logistic Regression (+ LASSO, Ridge) |
| **Tree models** | CART, Random Forest, LightGBM/GradientBoosting |
| **Neural net** | MLP (128-128-ReLU, Adam, early stopping) |
| **Tuning** | 5-fold cross-validation + GridSearchCV |
| **Explainability** | SHAP (Random Forest) |
| **Clustering** | KMeans (k=4) + PCA visualization |
""")

        st.markdown("#### 📈 Key Findings")
        st.markdown(f"""
<div class="insight-box">
🏆 <b>{best_model}</b> achieved the best AUC-ROC of <b>{best_auc:.3f}</b>.
<br><br>
📍 <b>Western Aleutians</b> sites are overwhelmingly Declining — consistent with NOAA
endangered status designations.
<br><br>
🧬 The most important predictors are <b>recent_mean</b> count, <b>pct_decline_from_peak</b>,
and <b>DPS membership</b> — confirming that the WDPS / EDPS split is a powerful discriminator.
<br><br>
🔮 Tree-based models substantially outperform linear models, indicating non-linear
interactions between region, count type, and temporal patterns.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📜 Feature Engineering Summary")
    feature_desc = {
        "mean_count": "Average count across all survey years",
        "recent_mean": "Average of last 5 observed counts",
        "early_mean": "Average of first 5 observed counts",
        "pct_decline_from_peak": "% decline from historical peak to recent mean",
        "cv": "Coefficient of variation (count variability)",
        "n_years": "Number of years with survey data",
        "r_squared": "R² of linear trend fit",
        "trend_pct_per_year": "% change per year from linear trend",
        "years_since_peak": "Years elapsed since peak population count",
        "dps": "Population segment: WDPS (endangered) or EDPS (delisted)",
        "region": "Geographic sub-region (e.g. W ALEU, E GULF, BC, CA…)",
        "count_type": "Non-Pup (adults) or Pup (juveniles)",
    }
    fd_df = pd.DataFrame(list(feature_desc.items()), columns=["Feature", "Description"])
    st.dataframe(fd_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: DATA VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("## 📊 Descriptive Analytics")

    # 2.1 Target distribution
    st.markdown('<div class="section-header">1 · Target Distribution</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.5])
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        labels  = ["Stable / Recovering", "Declining"]
        values  = [(df["is_declining"] == 0).sum(), df["is_declining"].sum()]
        colors  = ["#27ae60", "#c0392b"]
        bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=1.5)
        for bar, v in zip(bars, values):
            ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2,
                    f"{v} ({v/sum(values)*100:.1f}%)", va="center", fontsize=11, fontweight="bold")
        ax.set_xlabel("Number of Sites", fontsize=11)
        ax.set_title("Target Class Distribution", fontsize=13, fontweight="bold")
        ax.set_xlim(0, max(values) * 1.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown("""
<div class="insight-box">
The dataset is <b>slightly imbalanced</b> (~53% Declining), but not severely so.
The modest imbalance reflects the real-world situation: the WDPS is endangered
and declining while the EDPS is recovering.
</div>""", unsafe_allow_html=True)
    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        region_counts = df.groupby(["region", "is_declining"]).size().unstack(fill_value=0)
        region_counts.columns = ["Stable", "Declining"]
        region_counts.sort_values("Declining", ascending=True).plot(
            kind="barh", ax=ax, color=["#27ae60", "#c0392b"], edgecolor="white")
        ax.set_xlabel("Number of Sites"); ax.set_title("Decline by Region", fontweight="bold")
        ax.legend(loc="lower right"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown("""
<div class="insight-box">
<b>W ALEU</b> (Western Aleutians) is the most severely declining region, consistent with NOAA
endangered status. EDPS regions (BC, OR, WA, CA) show more stable/recovering trends.
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # 2.2 Time-series trends
    st.markdown('<div class="section-header">2 · Population Trends Over Time</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for dps, color, label in [("WDPS","#c0392b","WDPS (Endangered)"), ("EDPS","#27ae60","EDPS (Delisted)")]:
            sub = df_long[df_long["dps"] == dps]
            yr_mean = sub.groupby("year")["count"].sum().reset_index()
            ax.plot(yr_mean["year"], yr_mean["count"]/1000, color=color, lw=2.5, label=label)
            ax.fill_between(yr_mean["year"], yr_mean["count"]/1000, alpha=0.15, color=color)
        ax.set_xlabel("Year"); ax.set_ylabel("Total Count (thousands)")
        ax.set_title("Total Population by DPS Over Time", fontweight="bold")
        ax.legend(); ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown("""
<div class="insight-box">
The WDPS population peaked in the 1970s–80s and has declined sharply. The EDPS
shows a steady recovery since the late 1990s following federal protections.
</div>""", unsafe_allow_html=True)

    with col2:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for region in df_long[df_long["dps"] == "WDPS"]["region"].unique():
            sub = df_long[(df_long["dps"] == "WDPS") & (df_long["region"] == region)]
            yr = sub.groupby("year")["count"].sum()
            ax.plot(yr.index, yr.values/1000, lw=2, label=region, alpha=0.85)
        ax.set_xlabel("Year"); ax.set_ylabel("Count (thousands)")
        ax.set_title("WDPS Regional Trends", fontweight="bold")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown("""
<div class="insight-box">
Within the WDPS, the <b>W ALEU</b> (Western Aleutians) shows the steepest collapse,
while the Gulf of Alaska regions have partially stabilized.
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # 2.3 Feature distributions
    st.markdown('<div class="section-header">3 · Feature Distributions by Class</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    features_to_plot = [
        ("recent_mean", "Recent Mean Count"),
        ("pct_decline_from_peak", "% Decline from Peak"),
        ("cv", "Coefficient of Variation"),
        ("years_since_peak", "Years Since Peak"),
        ("n_years", "N Survey Years"),
        ("r_squared", "Trend R²"),
    ]
    palette = {0: "#27ae60", 1: "#c0392b"}
    for ax, (feat, title) in zip(axes.flat, features_to_plot):
        for cls, color in palette.items():
            subset = df[df["is_declining"] == cls][feat].dropna()
            ax.hist(subset, bins=25, alpha=0.6, color=color,
                    label="Declining" if cls == 1 else "Stable", edgecolor="white", linewidth=0.5)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xlabel("Value"); ax.set_ylabel("Count")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8)
    plt.suptitle("Feature Distributions: Declining vs Stable Sites", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    st.pyplot(fig); plt.close()
    st.markdown("""
<div class="insight-box">
<b>Recent Mean Count</b>: Declining sites tend to have very low recent counts, often near zero.
<b>% Decline from Peak</b>: Declining sites show dramatically higher peak-to-present losses.
<b>CV</b>: Declining sites have higher variability (erratic surveys as populations crash).
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # 2.4 Pup vs Non-Pup
    st.markdown('<div class="section-header">4 · Pup vs Non-Pup Decline Rates</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.2])
    with col1:
        pnp = df.groupby(["dps", "count_type", "is_declining"]).size().unstack(fill_value=0)
        pnp.columns = ["Stable", "Declining"]
        pnp["pct_declining"] = pnp["Declining"] / (pnp["Declining"] + pnp["Stable"]) * 100
        fig, ax = plt.subplots(figsize=(6, 4))
        x = range(len(pnp))
        bars = ax.bar(x, pnp["pct_declining"],
                      color=["#c0392b" if "WDPS" in str(i) else "#3498db" for i in pnp.index],
                      edgecolor="white", linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{i[0]}\n{i[1]}" for i in pnp.index], fontsize=9)
        ax.set_ylabel("% of Sites Declining"); ax.set_ylim(0, 110)
        ax.set_title("% Declining Sites by DPS & Count Type", fontweight="bold")
        for bar, pct in zip(bars, pnp["pct_declining"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{pct:.0f}%", ha="center", fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()
    with col2:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        df_box = df[["pct_decline_from_peak", "dps", "count_type"]].dropna()
        dps_type = df_box["dps"] + " " + df_box["count_type"]
        groups = {g: df_box[dps_type == g]["pct_decline_from_peak"].values for g in dps_type.unique()}
        bp = ax.boxplot(list(groups.values()), labels=list(groups.keys()),
                        patch_artist=True, notch=False, showfliers=True)
        colors = ["#c0392b", "#e74c3c", "#2980b9", "#3498db"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.set_ylabel("% Decline from Peak"); ax.set_title("Decline from Peak: DPS × Count Type", fontweight="bold")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()
    st.markdown("""
<div class="insight-box">
<b>WDPS Non-Pup</b> sites show the most severe peak-to-recent declines — adults have been
declining faster than pups in some subregions, indicating continued habitat pressure.
EDPS sites show much lower decline rates, confirming population recovery east of Alaska.
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # 2.5 Correlation heatmap
    st.markdown('<div class="section-header">5 · Correlation Heatmap</div>', unsafe_allow_html=True)
    num_features = [
        "mean_count", "max_count", "min_count", "std_count", "cv", "n_years",
        "trend_pct_per_year", "r_squared", "early_mean", "recent_mean",
        "pct_decline_from_peak", "years_since_peak", "is_declining"
    ]
    corr = df[num_features].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr), k=1)
    sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8},
                annot_kws={"size": 8})
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=12)
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig); plt.close()
    st.markdown("""
<div class="insight-box">
<b>Strong positive correlation (≈0.7+)</b>: <i>mean_count</i>, <i>early_mean</i>, <i>recent_mean</i>,
<i>max_count</i> — all collinear measures of overall site productivity.
<b>Negative correlation with is_declining</b>: <i>recent_mean</i> (−0.55) — lower recent counts
predict decline. <i>pct_decline_from_peak</i> is positively correlated with the target (+0.63).
This confirms that recent count levels and the magnitude of peak-to-present loss
are the strongest predictors of site decline status.
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("## 🏆 Model Performance Report")

    # Summary table
    st.markdown('<div class="section-header">Model Comparison — Test Set Metrics</div>', unsafe_allow_html=True)
    rows = []
    for m in MODEL_NAMES:
        if m not in results: continue
        r = results[m]
        rows.append({
            "Model": m,
            "Accuracy": r["accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1": r["f1"],
            "AUC-ROC": r["auc_roc"],
            "CV F1 (mean)": r["cv_f1_mean"],
            "CV F1 (±std)": r["cv_f1_std"],
        })
    metrics_df = pd.DataFrame(rows)
    st.dataframe(
        metrics_df.set_index("Model").style
            .background_gradient(cmap="Blues", subset=["AUC-ROC", "F1"])
            .format("{:.4f}"),
        use_container_width=True
    )
    st.markdown("*Note: LightGBM uses GradientBoostingClassifier when LightGBM is not installed.*")

    # Bar charts
    st.markdown("---")
    st.markdown('<div class="section-header">Metric Comparison Charts</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        models_sorted = metrics_df.sort_values("F1", ascending=True)
        colors = ["#c0392b" if m in ["CART","Random Forest","LightGBM"] else
                  "#f39c12" if m == "MLP" else "#2980b9" for m in models_sorted["Model"]]
        bars = ax.barh(models_sorted["Model"], models_sorted["F1"],
                       color=colors, edgecolor="white", linewidth=1.5)
        for bar, v in zip(bars, models_sorted["F1"]):
            ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                    f"{v:.3f}", va="center", fontsize=9, fontweight="bold")
        ax.set_xlabel("F1 Score"); ax.set_title("F1 Score by Model", fontweight="bold")
        ax.set_xlim(0.8, 1.02)
        ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()
    with col2:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.barh(models_sorted["Model"], models_sorted["AUC-ROC"],
                       color=colors, edgecolor="white", linewidth=1.5)
        for bar, v in zip(bars, models_sorted["AUC-ROC"]):
            ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                    f"{v:.3f}", va="center", fontsize=9, fontweight="bold")
        ax.set_xlabel("AUC-ROC"); ax.set_title("AUC-ROC by Model", fontweight="bold")
        ax.set_xlim(0.8, 1.05)
        ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # ROC Curves
    st.markdown("---")
    st.markdown('<div class="section-header">ROC Curves — All Models</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    color_map = {
        "Logistic Regression": "#3498db",
        "LASSO": "#9b59b6",
        "Ridge": "#1abc9c",
        "CART": "#e67e22",
        "Random Forest": "#c0392b",
        "LightGBM": "#e74c3c",
        "MLP": "#f39c12",
    }
    for m in MODEL_NAMES:
        if m not in results: continue
        r = results[m]
        ax.plot(r["roc_fpr"], r["roc_tpr"],
                label=f"{m} (AUC={r['auc_roc']:.3f})",
                color=color_map.get(m, "gray"), lw=2, alpha=0.85)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (AUC=0.5)")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontweight="bold", fontsize=13)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Per-model details
    st.markdown("---")
    st.markdown('<div class="section-header">Per-Model Detail</div>', unsafe_allow_html=True)
    selected_model = st.selectbox("Select model to inspect:", MODEL_NAMES)
    if selected_model in results:
        r = results[selected_model]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy",  f"{r['accuracy']:.3f}")
        c2.metric("Precision", f"{r['precision']:.3f}")
        c3.metric("Recall",    f"{r['recall']:.3f}")
        c4.metric("F1",        f"{r['f1']:.3f}")
        c5.metric("AUC-ROC",   f"{r['auc_roc']:.3f}")

        col_a, col_b = st.columns(2)
        with col_a:
            # Confusion Matrix
            cm_arr = np.array(r["confusion_matrix"])
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_arr,
                                          display_labels=["Stable", "Declining"])
            disp.plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(f"Confusion Matrix — {selected_model}", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig); plt.close()
        with col_b:
            # CV scores
            st.markdown(f"""
#### 5-Fold Cross-Validation (F1)
| Metric | Value |
|--------|-------|
| Mean F1 | **{r['cv_f1_mean']:.4f}** |
| Std Dev | {r['cv_f1_std']:.4f} |
""")
            if "best_params" in r:
                st.markdown("#### Best Hyperparameters")
                for k, v in r["best_params"].items():
                    st.write(f"- `{k}`: **{v}**")

        # CART tree visualization
        if selected_model == "CART":
            st.markdown("#### Decision Tree Visualization (top 3 levels)")
            cart_model = load_model("CART")
            if cart_model is not None:
                fig, ax = plt.subplots(figsize=(16, 6))
                plot_tree(cart_model, feature_names=feature_names,
                          class_names=["Stable", "Declining"],
                          filled=True, rounded=True, max_depth=3, ax=ax,
                          fontsize=7, impurity=False, proportion=True)
                ax.set_title("Decision Tree (depth ≤ 3)", fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig); plt.close()

        # MLP training history
        if selected_model == "MLP":
            st.markdown("#### Training History")
            fig, axes = plt.subplots(1, 2 if mlp_hist["val"] else 1,
                                     figsize=(10 if mlp_hist["val"] else 5, 4))
            if not isinstance(axes, np.ndarray): axes = [axes]
            axes[0].plot(mlp_hist["loss"], color="#c0392b", lw=2)
            axes[0].set_title("Training Loss (Log Loss)"); axes[0].set_xlabel("Epoch")
            axes[0].grid(True, alpha=0.3)
            if mlp_hist["val"] and len(axes) > 1:
                axes[1].plot(mlp_hist["val"], color="#27ae60", lw=2)
                axes[1].set_title("Validation Accuracy"); axes[1].set_xlabel("Epoch")
                axes[1].grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

    # Summary paragraph
    st.markdown("---")
    st.markdown('<div class="section-header">Model Comparison Discussion</div>', unsafe_allow_html=True)
    best_f1_model = max(results, key=lambda m: results[m]["f1"])
    best_f1 = results[best_f1_model]["f1"]
    st.markdown(f"""
**{best_f1_model}** achieved the highest F1 score ({best_f1:.3f}) and AUC-ROC, making it the
recommended model for deployment.

**Tree-based ensembles vs. linear models**: Random Forest and LightGBM/GradientBoosting
significantly outperform the linear baselines (Logistic, LASSO, Ridge), confirming that
the relationship between features and population decline is non-linear. For example,
the impact of `pct_decline_from_peak` on class membership is not uniform — sites with
extreme declines are categorically different from moderate ones.

**Bias-variance tradeoff**: CART is prone to overfitting (high F1 but lower CV stability)
while Random Forest's averaging reduces variance. The MLP neural network performs
comparably to linear models, likely due to the small dataset size limiting its generalization.

**Interpretability**: Linear models (LASSO especially) offer coefficient-level interpretability,
while tree-based models require SHAP for post-hoc explanation (see Explainability tab).
""")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: SHAP & INTERACTIVE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("## 🔍 Explainability & Interactive Prediction")

    # Feature importance (RF)
    st.markdown('<div class="section-header">Feature Importance — Random Forest</div>', unsafe_allow_html=True)
    rf_model = load_model("Random Forest")
    if rf_model is not None:
        importances = rf_model.feature_importances_
        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False).head(15)

        col1, col2 = st.columns([1.3, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(7, 5))
            colors = plt.cm.RdBu_r(np.linspace(0.1, 0.9, len(fi_df)))[::-1]
            bars = ax.barh(fi_df["Feature"][::-1], fi_df["Importance"][::-1],
                           color=colors[::-1], edgecolor="white", linewidth=0.8)
            ax.set_xlabel("Mean Decrease in Impurity")
            ax.set_title("Top 15 Feature Importances (Random Forest)", fontweight="bold")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig); plt.close()
        with col2:
            st.dataframe(fi_df.set_index("Feature").style.format("{:.4f}")
                         .background_gradient(cmap="Blues"), use_container_width=True)

        st.markdown("""
<div class="insight-box">
<b>pct_decline_from_peak</b> and <b>recent_mean</b> are the dominant predictors — sites with
large historical declines and low recent counts are nearly certain to be classified as Declining.
<b>DPS membership</b> (WDPS vs EDPS) is also highly predictive, reflecting the fundamental
biological and regulatory differences between the two segments.
<b>Region</b> encodes geographic proximity to prey, oceanographic conditions, and human impacts.
</div>""", unsafe_allow_html=True)

        # SHAP explanation (if SHAP available, otherwise use permutation importance visualization)
        sd = shap_data
        if sd["shap_values"] is not None:
            try:
                import shap
                st.markdown('<div class="section-header">SHAP Summary Plot</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(9, 6))
                shap.summary_plot(sd["shap_values"], sd["X_test"],
                                  feature_names=sd["feature_names"],
                                  show=False, plot_size=None, ax=ax)
                st.pyplot(fig); plt.close()
            except Exception as e:
                st.info(f"SHAP plots require `shap` package: {e}")
        else:
            st.info("ℹ️ Full SHAP beeswarm plots require the `shap` package (install with `pip install shap`). "
                    "Feature importance shown above is from Random Forest's built-in impurity decrease metric.")

    st.markdown("---")

    # LOGISTIC coefficients (linear model explainability)
    st.markdown('<div class="section-header">Logistic Regression Coefficients</div>', unsafe_allow_html=True)
    lr_model = load_model("Logistic Regression")
    if lr_model is not None:
        coefs = lr_model.coef_[0]
        coef_df = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": coefs
        }).sort_values("Coefficient", key=abs, ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(7, 5))
        bar_colors = ["#c0392b" if c > 0 else "#27ae60" for c in coef_df["Coefficient"][::-1]]
        ax.barh(coef_df["Feature"][::-1], coef_df["Coefficient"][::-1],
                color=bar_colors, edgecolor="white", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Coefficient Value (scaled)")
        ax.set_title("Logistic Regression — Feature Coefficients (top 15)", fontweight="bold")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown("""
<div class="insight-box">
Red bars → increase probability of <b>Declining</b>. Green bars → decrease it (associated with Stable).
High <b>pct_decline_from_peak</b> and low <b>recent_mean</b> most strongly drive the Declining prediction,
confirming the Random Forest finding. <b>WDPS membership</b> has a strong positive coefficient.
</div>""", unsafe_allow_html=True)

    # ── Interactive Prediction ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🎛️ Interactive Prediction</div>', unsafe_allow_html=True)
    st.markdown("Adjust the sliders below to define a hypothetical survey site and see what each model predicts.")

    # Build full feature vector (use medians as defaults)
    X_full = joblib.load(os.path.join(MODEL_DIR, "X_full.pkl"))
    medians = X_full.median()

    # Sliders for key features
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        recent_mean_val = st.slider("Recent Mean Count", 0, int(df["recent_mean"].quantile(0.99)), 500, step=10)
        pct_decline_val = st.slider("% Decline from Peak", 0.0, 100.0, 50.0, step=1.0)
        cv_val          = st.slider("Coefficient of Variation", 0.0, 3.0, 0.5, step=0.05)
    with col_s2:
        n_years_val     = st.slider("N Survey Years", 1, 50, 20)
        r2_val          = st.slider("Trend R²", 0.0, 1.0, 0.5, step=0.01)
        yrs_peak_val    = st.slider("Years Since Peak", 0, 50, 20)
    with col_s3:
        dps_val         = st.selectbox("Population Segment (DPS)", ["WDPS", "EDPS"])
        region_val      = st.selectbox("Region", sorted(df["region"].unique()))
        count_type_val  = st.selectbox("Count Type", ["Non-Pup", "Pup"])
        model_choice    = st.selectbox("Model to predict with", MODEL_NAMES)

    # Build feature vector
    X_custom = medians.copy()
    X_custom["recent_mean"]           = recent_mean_val
    X_custom["pct_decline_from_peak"] = pct_decline_val
    X_custom["cv"]                    = cv_val
    X_custom["n_years"]               = n_years_val
    X_custom["r_squared"]             = r2_val
    X_custom["years_since_peak"]      = yrs_peak_val
    X_custom["early_mean"]            = recent_mean_val * (1 + pct_decline_val / 100)

    # One-hot encode DPS
    for col in X_full.columns:
        if col.startswith("dps_"):
            X_custom[col] = 0
        if col.startswith("count_type_"):
            X_custom[col] = 0
        if col.startswith("region_"):
            X_custom[col] = 0
    if f"dps_{dps_val}" in X_custom.index:
        X_custom[f"dps_{dps_val}"] = 1
    if f"count_type_{count_type_val}" in X_custom.index:
        X_custom[f"count_type_{count_type_val}"] = 1
    if f"region_{region_val}" in X_custom.index:
        X_custom[f"region_{region_val}"] = 1

    X_vec = X_custom.values.reshape(1, -1)
    pred_model = load_model(model_choice)

    if pred_model is not None:
        needs_scaling = model_choice in ["Logistic Regression", "LASSO", "Ridge", "MLP"]
        if needs_scaling:
            X_vec_input = scaler.transform(X_vec)
        else:
            X_vec_input = X_vec

        pred_class = pred_model.predict(X_vec_input)[0]
        pred_prob  = pred_model.predict_proba(X_vec_input)[0]

        col_r1, col_r2, col_r3 = st.columns(3)
        outcome = "🔴 DECLINING" if pred_class == 1 else "🟢 STABLE / RECOVERING"
        col_r1.metric("Prediction", outcome)
        col_r2.metric("P(Declining)",  f"{pred_prob[1]:.3f}")
        col_r3.metric("P(Stable)",     f"{pred_prob[0]:.3f}")

        # Probability gauge
        fig, ax = plt.subplots(figsize=(7, 2))
        ax.barh(["Declining"],   [pred_prob[1]], color="#c0392b", height=0.4)
        ax.barh(["Stable"],      [pred_prob[0]], color="#27ae60", height=0.4)
        ax.set_xlim(0, 1); ax.set_xlabel("Probability")
        ax.set_title(f"Prediction Probabilities — {model_choice}", fontweight="bold")
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # Custom SHAP waterfall (approximate using feature importances)
        if rf_model is not None:
            st.markdown("#### Feature Contribution to Prediction")
            key_feats = fi_df.head(10)["Feature"].tolist()
            contribs = []
            for feat in key_feats:
                if feat in X_full.columns:
                    feat_idx = list(X_full.columns).index(feat)
                    baseline = float(medians.iloc[feat_idx])
                    custom   = float(X_custom.iloc[feat_idx])
                    direction = (custom - baseline) * rf_model.feature_importances_[feat_idx] * (1 if pred_class == 1 else -1)
                    contribs.append({"Feature": feat, "Contribution": direction})
            contrib_df = pd.DataFrame(contribs).sort_values("Contribution")
            fig, ax = plt.subplots(figsize=(7, 4))
            colors = ["#c0392b" if c > 0 else "#27ae60" for c in contrib_df["Contribution"]]
            ax.barh(contrib_df["Feature"], contrib_df["Contribution"], color=colors, edgecolor="white")
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title("Approximate Feature Contributions (custom input)", fontweight="bold")
            ax.set_xlabel("Direction × Importance")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("## 🗂️ Clustering Analysis")
    st.markdown("""
Clustering allows us to discover **natural groupings** of survey sites without relying
on the is_declining label. We use **KMeans (k=4)** on the full feature matrix and
visualize clusters via **PCA (2D projection)**.

This is especially useful for this dataset because regions and population segments
often co-vary in complex ways — clustering can reveal latent groupings not captured
by the simple Declining/Stable binary.
""")

    X_pca   = cluster["X_pca"]
    labels  = cluster["cluster_labels"]
    ev      = cluster["explained_variance"]

    col1, col2 = st.columns([1.3, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(7, 6))
        scatter_colors = ["#c0392b", "#2980b9", "#27ae60", "#f39c12"]
        for k in range(4):
            mask = labels == k
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       color=scatter_colors[k], label=f"Cluster {k+1}",
                       alpha=0.7, s=50, edgecolors="white", linewidth=0.5)
        # Overlay declining status
        dec_mask = df["is_declining"].values.astype(bool)
        ax.scatter(X_pca[dec_mask, 0], X_pca[dec_mask, 1],
                   marker="x", color="black", s=30, alpha=0.4, label="Declining (×)")
        ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% variance)")
        ax.set_title("KMeans Clusters (k=4) in PCA Space", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        # Cluster profiles
        df_cluster = df.copy()
        df_cluster["cluster"] = labels
        profile = df_cluster.groupby("cluster").agg({
            "is_declining": "mean",
            "mean_count": "median",
            "recent_mean": "median",
            "pct_decline_from_peak": "median",
            "dps": lambda x: x.value_counts().index[0],
        }).round(3)
        profile.index = [f"Cluster {i+1}" for i in profile.index]
        profile.columns = ["% Declining", "Median Count", "Recent Mean", "% Peak Decline", "Top DPS"]
        st.dataframe(profile.style.background_gradient(cmap="RdYlGn_r", subset=["% Declining"]),
                     use_container_width=True)

        st.markdown("""
<div class="insight-box">
<b>Cluster interpretation:</b>
- High % Declining + WDPS → Western Aleutian endangered sites
- Low % Declining + EDPS → Recovering Pacific coast sites
- Mixed clusters reflect intermediate regions (Gulf of Alaska, Eastern Aleutians)
Clustering reveals that decline is not random — it follows coherent geographic and ecological groupings.
</div>""", unsafe_allow_html=True)

    # Distribution of DPS and count type within clusters
    st.markdown("---")
    st.markdown('<div class="section-header">Cluster Composition</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df_cluster["cluster_label"] = df_cluster["cluster"].map({i: f"Cluster {i+1}" for i in range(4)})
    pd.crosstab(df_cluster["cluster_label"], df_cluster["dps"]).plot(
        kind="bar", ax=axes[0], color=["#27ae60", "#c0392b"], edgecolor="white", rot=0)
    axes[0].set_title("DPS by Cluster", fontweight="bold"); axes[0].set_xlabel("")
    axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)

    pd.crosstab(df_cluster["cluster_label"], df_cluster["region"]).plot(
        kind="bar", ax=axes[1], edgecolor="white", rot=30, colormap="tab10")
    axes[1].set_title("Region by Cluster", fontweight="bold"); axes[1].set_xlabel("")
    axes[1].legend(fontsize=7, bbox_to_anchor=(1, 1))
    axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("""
**Key Insight:** The clustering aligns well with the biological DPS boundaries —
WDPS sites naturally group into clusters with high decline rates while EDPS sites
cluster separately. This unsupervised finding validates the supervised classification
approach: the DPS split genuinely reflects different ecological trajectories.
""")
