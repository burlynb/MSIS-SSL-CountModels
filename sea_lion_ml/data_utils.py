"""
data_utils.py
Sea Lion Population Data Loading and Feature Engineering
MSIS 522 - HW1
"""

import pandas as pd
import numpy as np
from scipy import stats


def load_and_process_data(excel_path: str) -> pd.DataFrame:
    """
    Load 4 sheets from the ALLCOUNTS Excel file and engineer features.
    Returns one row per survey site with derived ML features and a
    binary classification target: 'is_declining'.
    """
    sheets = {
        "wdpsnp":     {"dps": "WDPS", "type": "Non-Pup"},
        "wdpspup":    {"dps": "WDPS", "type": "Pup"},
        "edpsnp_corr":{"dps": "EDPS", "type": "Non-Pup"},
        "edpspup":    {"dps": "EDPS", "type": "Pup"},
    }

    records = []

    for sheet_name, meta in sheets.items():
        df = pd.read_excel(excel_path, sheet_name=sheet_name)

        # Identify year columns (numeric)
        year_cols = [c for c in df.columns if isinstance(c, (int, float)) and 1960 <= int(c) <= 2030]
        year_cols = sorted(year_cols)

        region_col = "REGION" if "REGION" in df.columns else None

        for _, row in df.iterrows():
            site = row["SITE"]
            region = row[region_col] if region_col else "Unknown"

            # Extract time series
            ts = {}
            for yr in year_cols:
                val = row[yr]
                if pd.notna(val) and val >= 0:
                    ts[int(yr)] = float(val)

            if len(ts) < 5:
                continue  # need at least 5 data points

            years_arr = np.array(sorted(ts.keys()))
            counts_arr = np.array([ts[y] for y in years_arr])

            # --- Feature Engineering ---
            mean_count = counts_arr.mean()
            max_count = counts_arr.max()
            min_count = counts_arr.min()
            std_count = counts_arr.std()
            cv = (std_count / mean_count) if mean_count > 0 else 0.0
            n_years = len(years_arr)

            # Linear trend: slope and r-value
            slope, intercept, r_val, p_val, _ = stats.linregress(years_arr, counts_arr)
            trend_pct_per_year = (slope / mean_count * 100) if mean_count > 0 else 0.0

            # Early vs recent comparison (first/last 5 obs)
            early_mean = counts_arr[:5].mean()
            recent_mean = counts_arr[-5:].mean()
            ratio_recent_early = (recent_mean / early_mean) if early_mean > 0 else 1.0

            # Peak analysis
            peak_idx = counts_arr.argmax()
            peak_year = int(years_arr[peak_idx])
            years_since_peak = 2023 - peak_year
            count_at_peak = counts_arr[peak_idx]
            pct_decline_from_peak = ((count_at_peak - recent_mean) / count_at_peak * 100
                                     if count_at_peak > 0 else 0.0)

            # Decade averages
            avg_1980s = counts_arr[(years_arr >= 1980) & (years_arr < 1990)].mean() if np.any((years_arr >= 1980) & (years_arr < 1990)) else np.nan
            avg_2000s = counts_arr[(years_arr >= 2000) & (years_arr < 2010)].mean() if np.any((years_arr >= 2000) & (years_arr < 2010)) else np.nan
            avg_2010s = counts_arr[(years_arr >= 2010) & (years_arr < 2020)].mean() if np.any((years_arr >= 2010) & (years_arr < 2020)) else np.nan

            records.append({
                "site": site,
                "region": region,
                "dps": meta["dps"],
                "count_type": meta["type"],
                "mean_count": mean_count,
                "max_count": max_count,
                "min_count": min_count,
                "std_count": std_count,
                "cv": cv,
                "n_years": n_years,
                "trend_slope": slope,
                "trend_pct_per_year": trend_pct_per_year,
                "r_squared": r_val ** 2,
                "trend_pvalue": p_val,
                "early_mean": early_mean,
                "recent_mean": recent_mean,
                "ratio_recent_early": ratio_recent_early,
                "peak_year": peak_year,
                "years_since_peak": years_since_peak,
                "pct_decline_from_peak": pct_decline_from_peak,
                "avg_1980s": avg_1980s,
                "avg_2000s": avg_2000s,
                "avg_2010s": avg_2010s,
                # store raw time series for visualization
                "_years": years_arr.tolist(),
                "_counts": counts_arr.tolist(),
            })

    df_feat = pd.DataFrame(records)

    # -------------------------------------------------------------------
    # Classification target: is_declining
    #   Declining = negative slope AND recent/early ratio < 0.85
    # -------------------------------------------------------------------
    df_feat["is_declining"] = (
        (df_feat["trend_slope"] < 0) &
        (df_feat["ratio_recent_early"] < 0.85)
    ).astype(int)

    # Region sub-categories
    df_feat["region_group"] = df_feat["region"].apply(_region_group)

    return df_feat


def _region_group(region: str) -> str:
    r = str(region).upper()
    if "W ALEU" in r or "W_ALEU" in r:
        return "W Aleutians"
    elif "C ALEU" in r:
        return "C Aleutians"
    elif "E ALEU" in r:
        return "E Aleutians"
    elif "GULF" in r:
        return "Gulf of Alaska"
    elif r in ("BC", "CA", "OR", "WA", "SE AK"):
        return f"EDPS-{r}"
    return "Other"


def get_ml_features_target(df: pd.DataFrame):
    """
    Returns X (feature matrix) and y (target) ready for sklearn.
    """
    # One-hot encode categoricals
    cat_cols = ["dps", "count_type", "region"]
    # NOTE: trend_slope and ratio_recent_early directly define the target
    # (is_declining = slope<0 AND ratio<0.85), so they are excluded to prevent
    # data leakage. The models must learn from other correlated features.
    num_cols = [
        "mean_count", "max_count", "min_count", "std_count", "cv",
        "n_years", "r_squared", "trend_pct_per_year",
        "early_mean", "recent_mean",
        "peak_year", "years_since_peak", "pct_decline_from_peak",
        "avg_1980s", "avg_2000s", "avg_2010s",
    ]

    # Fill NaN in numeric cols with median
    df_ml = df[num_cols + cat_cols].copy()
    for c in num_cols:
        df_ml[c] = df_ml[c].fillna(df_ml[c].median())

    X = pd.get_dummies(df_ml, columns=cat_cols, drop_first=False)
    y = df["is_declining"].values

    return X, y


def get_long_format(df: pd.DataFrame, interpolate: bool = True) -> pd.DataFrame:
    """
    Explode the raw time-series back to long form.

    When interpolate=True (default), missing years between the first and last
    observed survey are filled using PCHIP spline interpolation rather than
    being left as gaps (which matplotlib renders as drop-to-zero spikes).

    This mirrors the agTrend.ssl philosophy: treat unsurveyed years as
    missing-at-random, not as zero-count events. Interpolation is done in
    log(count+1) space to avoid negatives and respect log-normal count distributions.
    """
    from scipy.interpolate import PchipInterpolator

    rows = []
    for _, r in df.iterrows():
        years_obs  = np.array(r["_years"], dtype=int)
        counts_obs = np.array(r["_counts"], dtype=float)

        if interpolate and len(years_obs) >= 3:
            yr_min, yr_max = years_obs.min(), years_obs.max()
            all_years = np.arange(yr_min, yr_max + 1)
            log_counts = np.log1p(counts_obs)
            interp_fn  = PchipInterpolator(years_obs, log_counts, extrapolate=False)
            log_interp = interp_fn(all_years)
            interp_counts = np.expm1(np.clip(log_interp, 0, None))
            for yr, cnt in zip(all_years, interp_counts):
                rows.append({
                    "site":         r["site"],
                    "region":       r["region"],
                    "dps":          r["dps"],
                    "count_type":   r["count_type"],
                    "year":         int(yr),
                    "count":        float(cnt) if not np.isnan(cnt) else np.nan,
                    "is_observed":  bool(yr in years_obs),
                    "is_declining": r["is_declining"],
                })
        else:
            for yr, cnt in zip(years_obs, counts_obs):
                rows.append({
                    "site":         r["site"],
                    "region":       r["region"],
                    "dps":          r["dps"],
                    "count_type":   r["count_type"],
                    "year":         int(yr),
                    "count":        cnt,
                    "is_observed":  True,
                    "is_declining": r["is_declining"],
                })

    long_df = pd.DataFrame(rows)
    long_df = long_df.dropna(subset=["count"])
    return long_df
