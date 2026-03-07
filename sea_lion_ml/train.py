"""
train.py
Train all models, run cross-validation, compute SHAP values.
Run this script once before launching the Streamlit app.

Usage:
    python train.py
"""

import os
import warnings
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
import joblib

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "ALLCOUNTS_v21_FED-fish.xlsx")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42

# ── Import from data_utils ─────────────────────────────────────────────────────
import sys
sys.path.insert(0, BASE_DIR)
from data_utils import load_and_process_data, get_ml_features_target, get_long_format

# ── Try optional imports ───────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    HAS_LGB = True
    print("LightGBM available ✓")
except ImportError:
    HAS_LGB = False
    print("LightGBM not available – using GradientBoostingClassifier as fallback")

try:
    import shap
    HAS_SHAP = True
    print("SHAP available ✓")
except ImportError:
    HAS_SHAP = False
    print("SHAP not installed – SHAP values will use permutation importance fallback")


# ==============================================================================
def main():
    print("=" * 60)
    print("Sea Lion Population Trend Classification Pipeline")
    print("=" * 60)

    # 1. Load & process data
    print("\n[1/6] Loading and processing data …")
    df = load_and_process_data(DATA_PATH)
    print(f"      Total sites: {len(df)}, Declining: {df['is_declining'].sum()}, Not: {(df['is_declining']==0).sum()}")

    # Save processed dataframe
    df.to_pickle(os.path.join(MODEL_DIR, "processed_df.pkl"))
    print("      Saved processed_df.pkl")

    # 2. Feature matrix
    print("\n[2/6] Preparing features …")
    X, y = get_ml_features_target(df)
    feature_names = list(X.columns)
    print(f"      Features: {len(feature_names)}, Samples: {len(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )

    # Save splits & feature names
    joblib.dump((X_train, X_test, y_train, y_test, feature_names),
                os.path.join(MODEL_DIR, "splits.pkl"))
    joblib.dump(X, os.path.join(MODEL_DIR, "X_full.pkl"))

    # Scaler (fit on train)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {}

    # ── Helper ────────────────────────────────────────────────────────────────
    def eval_model(model, X_tr, X_te, y_tr, y_te, name, scaled=False):
        """Fit, predict, compute metrics, save model."""
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else y_pred

        acc  = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, zero_division=0)
        rec  = recall_score(y_te, y_pred, zero_division=0)
        f1   = f1_score(y_te, y_pred, zero_division=0)
        auc  = roc_auc_score(y_te, y_prob) if len(np.unique(y_te)) > 1 else 0.5
        cm   = confusion_matrix(y_te, y_pred).tolist()
        fpr, tpr, _ = roc_curve(y_te, y_prob)

        # CV score
        cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv5,
                                    scoring="f1", n_jobs=-1)
        cv_f1_mean = cv_scores.mean()
        cv_f1_std  = cv_scores.std()

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "auc_roc": round(auc, 4),
            "cv_f1_mean": round(cv_f1_mean, 4),
            "cv_f1_std": round(cv_f1_std, 4),
            "confusion_matrix": cm,
            "roc_fpr": fpr.tolist(),
            "roc_tpr": tpr.tolist(),
        }

        tag = name.replace(" ", "_").lower()
        joblib.dump(model, os.path.join(MODEL_DIR, f"{tag}.pkl"))
        print(f"      {name:30s}  F1={f1:.3f}  AUC={auc:.3f}  CV-F1={cv_f1_mean:.3f}±{cv_f1_std:.3f}")
        return model

    # ── 3. Models ─────────────────────────────────────────────────────────────
    print("\n[3/6] Training models …")

    # Logistic Regression (baseline)
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    eval_model(lr, X_train_sc, X_test_sc, y_train, y_test, "Logistic Regression")

    # LASSO (L1 Logistic)
    lasso = LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000,
                               C=0.1, random_state=RANDOM_STATE)
    eval_model(lasso, X_train_sc, X_test_sc, y_train, y_test, "LASSO")

    # Ridge (L2 Logistic)
    ridge = LogisticRegression(penalty="l2", max_iter=1000, C=0.1,
                               random_state=RANDOM_STATE)
    eval_model(ridge, X_train_sc, X_test_sc, y_train, y_test, "Ridge")

    # CART – GridSearchCV
    print("      Running GridSearchCV for CART …")
    cart_grid = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid={"max_depth": [3, 5, 7, 10],
                    "min_samples_leaf": [5, 10, 20, 50]},
        cv=cv5, scoring="f1", n_jobs=-1
    )
    cart_grid.fit(X_train, y_train)
    best_cart = cart_grid.best_estimator_
    results["CART"] = {}  # placeholder – re-eval
    cart_model = eval_model(best_cart, X_train, X_test, y_train, y_test, "CART")
    results["CART"]["best_params"] = cart_grid.best_params_
    joblib.dump(cart_grid, os.path.join(MODEL_DIR, "cart_gridsearch.pkl"))

    # Random Forest – GridSearchCV
    print("      Running GridSearchCV for Random Forest …")
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid={"n_estimators": [50, 100, 200],
                    "max_depth": [3, 5, 8]},
        cv=cv5, scoring="f1", n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    results["Random Forest"] = {}
    rf_model = eval_model(best_rf, X_train, X_test, y_train, y_test, "Random Forest")
    results["Random Forest"]["best_params"] = rf_grid.best_params_
    joblib.dump(rf_grid, os.path.join(MODEL_DIR, "rf_gridsearch.pkl"))

    # LightGBM or Gradient Boosting fallback
    if HAS_LGB:
        print("      Running GridSearchCV for LightGBM …")
        lgb_grid = GridSearchCV(
            lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
            param_grid={"n_estimators": [50, 100, 200],
                        "max_depth": [3, 4, 5, 6],
                        "learning_rate": [0.01, 0.05, 0.1]},
            cv=cv5, scoring="f1", n_jobs=-1
        )
        lgb_grid.fit(X_train, y_train)
        best_boost = lgb_grid.best_estimator_
        model_name = "LightGBM"
        results["LightGBM"] = {}
        boost_model = eval_model(best_boost, X_train, X_test, y_train, y_test, "LightGBM")
        results["LightGBM"]["best_params"] = lgb_grid.best_params_
        joblib.dump(lgb_grid, os.path.join(MODEL_DIR, "lgb_gridsearch.pkl"))
    else:
        print("      Running GridSearchCV for Gradient Boosting (LightGBM fallback) …")
        gb_grid = GridSearchCV(
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            param_grid={"n_estimators": [50, 100, 200],
                        "max_depth": [3, 4, 5],
                        "learning_rate": [0.05, 0.1, 0.2]},
            cv=cv5, scoring="f1", n_jobs=-1
        )
        gb_grid.fit(X_train, y_train)
        best_boost = gb_grid.best_estimator_
        model_name = "GradientBoosting"
        results["LightGBM"] = {}
        boost_model = eval_model(best_boost, X_train, X_test, y_train, y_test, "LightGBM")
        results["LightGBM"]["best_params"] = gb_grid.best_params_
        results["LightGBM"]["note"] = "GradientBoostingClassifier used (LightGBM not installed)"
        joblib.dump(gb_grid, os.path.join(MODEL_DIR, "lgb_gridsearch.pkl"))
        joblib.dump(best_boost, os.path.join(MODEL_DIR, "lightgbm.pkl"))

    # MLP Neural Network
    print("      Training MLP Neural Network …")
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 128),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
    )
    mlp.fit(X_train_sc, y_train)
    # Capture loss curve
    loss_curve = mlp.loss_curve_
    val_scores = mlp.validation_scores_ if hasattr(mlp, "validation_scores_") else []
    joblib.dump({"loss": loss_curve, "val": list(val_scores)},
                os.path.join(MODEL_DIR, "mlp_history.pkl"))
    eval_model(mlp, X_train_sc, X_test_sc, y_train, y_test, "MLP")

    # ── GAM (Generalized Additive Model) — inspired by agTrend.ssl methodology ──
    print("      Training GAM (LogisticGAM — agTrend.ssl-inspired) …")
    try:
        from pygam import LogisticGAM, s
        

        # Impute any NaNs in scaled data (GAM doesn't handle NaN)
        # Force to numpy array first, then replace all NaNs with 0
        X_tr_gam = np.array(X_train_sc, dtype=float)
        X_te_gam = np.array(X_test_sc, dtype=float)
        col_means = np.where(np.isfinite(X_tr_gam).any(axis=0),
                             np.nanmean(np.where(np.isfinite(X_tr_gam), X_tr_gam, np.nan), axis=0), 0)
        for col_idx in range(X_tr_gam.shape[1]):
            X_tr_gam[~np.isfinite(X_tr_gam[:, col_idx]), col_idx] = col_means[col_idx]
            X_te_gam[~np.isfinite(X_te_gam[:, col_idx]), col_idx] = col_means[col_idx]
        # Final safety net
        X_tr_gam = np.nan_to_num(X_tr_gam, nan=0.0, posinf=0.0, neginf=0.0)
        X_te_gam = np.nan_to_num(X_te_gam, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"      GAM input: {X_tr_gam.shape}, NaNs remaining: {np.isnan(X_tr_gam).sum()}")

        # Use only first 15 numerical features (spline terms for continuous vars)
        # Skip gridsearch — it generates NaN internally on this dataset.
        # Use only top 8 numerical features with high regularization to avoid divergence.
        gam = LogisticGAM(
            s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7),
            max_iter=200, lam=10.0,
        )
        gam.fit(X_tr_gam[:, :8], y_train)
        y_pred_gam = (gam.predict_proba(X_te_gam[:, :8]) > 0.5).astype(int)
        y_prob_gam = gam.predict_proba(X_te_gam[:, :8])

        acc  = accuracy_score(y_test, y_pred_gam)
        prec = precision_score(y_test, y_pred_gam, zero_division=0)
        rec  = recall_score(y_test, y_pred_gam, zero_division=0)
        f1   = f1_score(y_test, y_pred_gam, zero_division=0)
        auc  = roc_auc_score(y_test, y_prob_gam)
        fpr, tpr, _ = roc_curve(y_test, y_prob_gam)
        cm   = confusion_matrix(y_test, y_pred_gam).tolist()


        # Wrap GAM for cross_val_score (manual)
        cv_scores_gam = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        for tr_idx, val_idx in skf.split(X_tr_gam, y_train):
            g = LogisticGAM(s(0)+s(1)+s(2)+s(3)+s(4)+s(5)+s(6)+s(7), max_iter=100, lam=10.0)
            g.fit(X_tr_gam[tr_idx][:, :8], y_train[tr_idx])
            preds = (g.predict_proba(X_tr_gam[val_idx][:, :8]) > 0.5).astype(int)
            cv_scores_gam.append(f1_score(y_train[val_idx], preds, zero_division=0))
        cv_mean = np.mean(cv_scores_gam)
        cv_std  = np.std(cv_scores_gam)

        results["GAM"] = {
            "accuracy":    round(acc, 4),
            "precision":   round(prec, 4),
            "recall":      round(rec, 4),
            "f1":          round(f1, 4),
            "auc_roc":     round(auc, 4),
            "cv_f1_mean":  round(cv_mean, 4),
            "cv_f1_std":   round(cv_std, 4),
            "confusion_matrix": cm,
            "roc_fpr":     fpr.tolist(),
            "roc_tpr":     tpr.tolist(),
            "note": "LogisticGAM with penalized splines — Python equivalent of agTrend.ssl mgcv GAM",
        }
        joblib.dump(gam, os.path.join(MODEL_DIR, "gam.pkl"))
        print(f"      {'GAM':30s}  F1={f1:.3f}  AUC={auc:.3f}  CV-F1={cv_mean:.3f}±{cv_std:.3f}")
        HAS_GAM = True
    except ImportError:
        print("      pygam not installed — skipping GAM (run: pip install pygam)")
        HAS_GAM = False
    except Exception as e:
        print(f"      GAM failed: {e} — skipping")
        HAS_GAM = False

    # ── 4. SHAP ───────────────────────────────────────────────────────────────
    print("\n[4/6] Computing SHAP values …")
    if HAS_SHAP:
        try:
            explainer    = shap.TreeExplainer(best_rf)
            shap_values  = explainer.shap_values(X_test)
            sv_arr       = np.array(shap_values)

            # Handle all possible output shapes:
            # Legacy SHAP (<0.41): list of 2 arrays each (n, features)
            # New SHAP (>=0.41):   single array (n, features, 2)
            if isinstance(shap_values, list):
                sv = np.array(shap_values[1])        # class 1
                ev = explainer.expected_value[1] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value
            elif sv_arr.ndim == 3:                   # (n, features, 2)
                sv = sv_arr[:, :, 1]                 # class 1
                ev = explainer.expected_value[1] if hasattr(explainer.expected_value, '__len__') else float(explainer.expected_value)
            else:                                    # already (n, features)
                sv = sv_arr
                ev = float(np.array(explainer.expected_value).flat[-1])

            joblib.dump({
                "shap_values":    sv,          # always (n_test, n_features) for class 1
                "expected_value": float(ev),
                "X_test":         X_test,      # DataFrame preserved for feature names
                "feature_names":  feature_names,
            }, os.path.join(MODEL_DIR, "shap_data.pkl"))
            print(f"      SHAP saved. shap_values shape: {sv.shape}")
        except Exception as e:
            print(f"      SHAP computation failed: {e}")
            joblib.dump({"shap_values": None, "X_test": X_test,
                         "feature_names": feature_names, "expected_value": 0.0},
                        os.path.join(MODEL_DIR, "shap_data.pkl"))
    else:
        joblib.dump({"shap_values": None, "X_test": X_test,
                     "feature_names": feature_names, "expected_value": 0.0},
                    os.path.join(MODEL_DIR, "shap_data.pkl"))
        print("      SHAP unavailable; saved fallback.")

    # ── 5. Save results JSON ──────────────────────────────────────────────────
    print("\n[5/6] Saving metrics …")
    with open(os.path.join(MODEL_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("      results.json saved.")

    # ── 6. Clustering (KMeans) ────────────────────────────────────────────────
    print("\n[6/6] Clustering with KMeans …")
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    X_cluster = X.values
    kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(X_cluster)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(scaler.transform(X_cluster))

    joblib.dump({
        "kmeans": kmeans,
        "cluster_labels": cluster_labels,
        "X_pca": X_pca,
        "pca": pca,
        "explained_variance": pca.explained_variance_ratio_.tolist(),
    }, os.path.join(MODEL_DIR, "clustering.pkl"))
    print("      Clustering saved.")

    print("\n✅ Training complete! All artifacts saved to models/")
    print("   → Launch app with:  streamlit run app.py")


if __name__ == "__main__":
    main()
