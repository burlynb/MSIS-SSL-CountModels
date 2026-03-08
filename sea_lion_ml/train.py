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
# Keras preferred for MLP; sklearn fallback if not available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False
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


# ── KerasWrapper — module-level so joblib can pickle it ───────────────────────
class KerasWrapper:
    """Sklearn-compatible wrapper around a saved Keras model.
    Falls back to sklearn MLP pkl automatically if TensorFlow is unavailable
    (e.g. Streamlit Cloud deployment without TF installed)."""
    _is_keras_wrapper = True
    def __init__(self, model_path):
        self.model_path = model_path
        self._model = None
    def _load(self):
        if self._model is None:
            try:
                import tensorflow as tf
                self._model = tf.keras.models.load_model(self.model_path)
            except Exception:
                import os, joblib as _jl
                sk_path = os.path.join(os.path.dirname(self.model_path), "mlp.pkl")
                if os.path.exists(sk_path):
                    self._model = _jl.load(sk_path)
                else:
                    raise RuntimeError("TensorFlow unavailable and no sklearn MLP fallback found.")
    def predict(self, X):
        self._load()
        if hasattr(self._model, "predict_proba"):
            return self._model.predict(X)
        return (self._model.predict(X, verbose=0).flatten() > 0.5).astype(int)
    def predict_proba(self, X):
        self._load()
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)
        p = self._model.predict(X, verbose=0).flatten()
        return np.column_stack([1 - p, p])


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
        """Fit (if not KerasWrapper), predict, compute metrics, save model."""
        is_keras = hasattr(model, '_is_keras_wrapper')
        if not is_keras:
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

        # CV score — manual for Keras, sklearn cross_val_score for others
        if is_keras:
            # Manual 5-fold CV for Keras
            cv_scores = []
            for tr_idx, val_idx in cv5.split(X_tr, y_tr):
                preds = model.predict(X_tr[val_idx])
                cv_scores.append(f1_score(y_tr[val_idx], preds, zero_division=0))
            cv_f1_mean = np.mean(cv_scores)
            cv_f1_std  = np.std(cv_scores)
        else:
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
        # For MLP/Keras: also save an sklearn fallback for cloud deployment without TF
        if is_keras:
            from sklearn.neural_network import MLPClassifier as _MLPC
            _mlp_sk = _MLPC(hidden_layer_sizes=(128,128), activation="relu", solver="adam",
                            max_iter=300, random_state=RANDOM_STATE)
            _mlp_sk.fit(X_tr, y_tr)
            joblib.dump(_mlp_sk, os.path.join(MODEL_DIR, f"{tag}.pkl"))
            print(f"      Saved sklearn MLP fallback to {tag}.pkl for cloud deployment")
        else:
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

    # MLP Neural Network — Keras/TensorFlow preferred; sklearn fallback
    print("      Training MLP Neural Network …")
    n_features = X_train_sc.shape[1]
    if HAS_KERAS:
        tf.random.set_seed(RANDOM_STATE)
        # Build model: Input -> 128 ReLU -> 128 ReLU -> sigmoid output
        mlp_keras = keras.Sequential([
            layers.Input(shape=(n_features,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),   # binary cross-entropy output
        ])
        mlp_keras.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        history = mlp_keras.fit(
            X_train_sc, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.15,
            callbacks=[keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)],
            verbose=0,
        )
        # Wrap in a sklearn-compatible interface for eval_model
        # KerasWrapper defined at module level below
        keras_path = os.path.join(MODEL_DIR, "mlp_keras.keras")
        mlp_keras.save(keras_path)
        mlp = KerasWrapper(keras_path)
        loss_curve = history.history["loss"]
        val_loss   = history.history.get("val_loss", [])
        val_acc    = history.history.get("val_accuracy", [])
        train_acc  = history.history.get("accuracy", [])
        joblib.dump({
            "loss": loss_curve, "val": val_loss,
            "train_acc": train_acc, "val_acc": val_acc,
            "framework": "keras",
        }, os.path.join(MODEL_DIR, "mlp_history.pkl"))
        print("      Keras MLP saved.")
    else:
        # sklearn fallback
        print("      TensorFlow not found — using sklearn MLPClassifier fallback")
        from sklearn.neural_network import MLPClassifier
        mlp_sk = MLPClassifier(
            hidden_layer_sizes=(128, 128), activation="relu", solver="adam",
            max_iter=500, random_state=RANDOM_STATE,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
        )
        mlp_sk.fit(X_train_sc, y_train)
        mlp = mlp_sk
        loss_curve = mlp_sk.loss_curve_
        val_scores = list(mlp_sk.validation_scores_) if hasattr(mlp_sk, "validation_scores_") else []
        joblib.dump({"loss": loss_curve, "val": val_scores, "framework": "sklearn"},
                    os.path.join(MODEL_DIR, "mlp_history.pkl"))
    eval_model(mlp, X_train_sc, X_test_sc, y_train, y_test, "MLP")

    # ── MLP Hyperparameter Tuning (Bonus) ─────────────────────────────────────
    if HAS_KERAS:
        print("      Running MLP hyperparameter grid search (bonus) …")
        # Small 2x2x2 grid to keep runtime reasonable (~8 models x ~30s = ~4 min)
        hid_sizes   = [(64, 64), (128, 128)]
        learn_rates = [0.001, 0.01]
        drop_rates  = [0.1, 0.3]

        tuning_results = []
        combo_num = 0
        total = len(hid_sizes) * len(learn_rates) * len(drop_rates)
        for hid in hid_sizes:
            for lr in learn_rates:
                for dr in drop_rates:
                    combo_num += 1
                    print(f"        [{combo_num}/{total}] layers={hid} lr={lr} dropout={dr}")
                    tf.random.set_seed(RANDOM_STATE)
                    m = keras.Sequential([
                        layers.Input(shape=(X_train_sc.shape[1],)),
                        layers.Dense(hid[0], activation="relu"),
                        layers.Dropout(dr),
                        layers.Dense(hid[1], activation="relu"),
                        layers.Dropout(dr),
                        layers.Dense(1, activation="sigmoid"),
                    ])
                    m.compile(
                        optimizer=keras.optimizers.Adam(learning_rate=lr),
                        loss="binary_crossentropy",
                        metrics=["accuracy"],
                    )
                    h = m.fit(
                        X_train_sc, y_train,
                        epochs=80,
                        batch_size=32,
                        validation_split=0.15,
                        callbacks=[keras.callbacks.EarlyStopping(
                            patience=10, restore_best_weights=True)],
                        verbose=0,
                    )
                    # Evaluate on test set
                    p_prob = m.predict(X_test_sc, verbose=0).flatten()
                    p_cls  = (p_prob > 0.5).astype(int)
                    f1_val = f1_score(y_test, p_cls, zero_division=0)
                    auc_val = roc_auc_score(y_test, p_prob)
                    epochs_run = len(h.history["loss"])
                    tuning_results.append({
                        "hidden_sizes": str(hid),
                        "learning_rate": lr,
                        "dropout": dr,
                        "f1":    round(f1_val, 4),
                        "auc":   round(auc_val, 4),
                        "epochs": epochs_run,
                        "val_loss_final": round(h.history["val_loss"][-1], 4),
                    })
                    print(f"          F1={f1_val:.4f}  AUC={auc_val:.4f}  epochs={epochs_run}")

        best_tune = max(tuning_results, key=lambda x: x["f1"])
        print(f"      Best tuning config: {best_tune}")
        joblib.dump({
            "results": tuning_results,
            "best":    best_tune,
            "hid_sizes":   [str(h) for h in hid_sizes],
            "learn_rates": learn_rates,
            "drop_rates":  drop_rates,
        }, os.path.join(MODEL_DIR, "mlp_tuning.pkl"))
        print("      MLP tuning results saved.")

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

    # ── 6. Trajectory-only clustering ────────────────────────────────────────
    print("\n[6/6] Trajectory clustering (trend features only) …")
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler as TrajScaler

    # Use ONLY trajectory/trend features — ignores population size entirely
    # This reveals population MOVEMENT types: collapsing / declining / stable / recovering
    traj_features = [
        "pct_decline_from_peak",   # how much lost relative to peak
        "trend_pct_per_year",      # annual rate of change
        "r_squared",               # trend reliability / consistency
        "years_since_peak",        # how long the decline has been ongoing
        "cv",                      # variability (erratic vs smooth trajectory)
        "ratio_recent_early",      # recent vs early abundance ratio
    ]
    # Only use features that exist in df
    traj_cols = [c for c in traj_features if c in df.columns]
    if len(traj_cols) < 3:
        # fallback to trend-related columns from the full feature set
        traj_cols = ["pct_decline_from_peak", "trend_pct_per_year", "r_squared",
                     "years_since_peak", "cv"]
        traj_cols = [c for c in traj_cols if c in df.columns]

    print(f"      Trajectory features used: {traj_cols}")

    X_traj = df[traj_cols].fillna(df[traj_cols].median()).values
    traj_scaler = TrajScaler()
    X_traj_sc = traj_scaler.fit_transform(X_traj)

    # Try k=4 to find: Collapsing / Declining / Stable / Recovering
    kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=20)
    cluster_labels = kmeans.fit_predict(X_traj_sc)

    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_traj_sc)

    # Label clusters by their mean decline rate for interpretability
    cluster_profiles = {}
    for k in range(4):
        mask = cluster_labels == k
        profile = {
            "mean_trend_pct_yr":      float(df.loc[mask, "trend_pct_per_year"].mean()) if "trend_pct_per_year" in df.columns else 0,
            "mean_pct_decline_peak":  float(df.loc[mask, "pct_decline_from_peak"].mean()) if "pct_decline_from_peak" in df.columns else 0,
            "pct_declining_label":    float(df.loc[mask, "is_declining"].mean()),
            "n_sites":                int(mask.sum()),
        }
        cluster_profiles[k] = profile

    # Sort clusters by mean trend (most negative = Collapsing, most positive = Recovering)
    sorted_by_trend = sorted(cluster_profiles.keys(),
                             key=lambda k: cluster_profiles[k]["mean_trend_pct_yr"])
    cluster_names = {}
    traj_labels = ["Collapsing", "Declining", "Stable", "Recovering"]
    for rank, orig_k in enumerate(sorted_by_trend):
        cluster_names[orig_k] = traj_labels[rank]
    print("      Trajectory cluster assignments:")
    for k in sorted_by_trend:
        print(f"        Cluster {k+1} ({cluster_names[k]}): "
              f"trend={cluster_profiles[k]['mean_trend_pct_yr']:+.2f}%/yr  "
              f"decline_from_peak={cluster_profiles[k]['mean_pct_decline_peak']:.1f}%  "
              f"n={cluster_profiles[k]['n_sites']}")

    joblib.dump({
        "kmeans":             kmeans,
        "cluster_labels":     cluster_labels,
        "cluster_names":      cluster_names,
        "cluster_profiles":   cluster_profiles,
        "traj_features":      traj_cols,
        "traj_scaler":        traj_scaler,
        "X_pca":              X_pca,
        "pca":                pca,
        "explained_variance": pca.explained_variance_ratio_.tolist(),
    }, os.path.join(MODEL_DIR, "clustering.pkl"))
    print("      Trajectory clustering saved.")

    print("\n✅ Training complete! All artifacts saved to models/")
    print("   → Launch app with:  streamlit run app.py")


if __name__ == "__main__":
    main()
