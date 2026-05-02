"""
featureStore.py
Pulls all collectors, aligns to daily business-day frequency,
handles missing data, saves to SQLite.

Modes:
  python featureStore.py --full   → full rebuild (first run)
  python featureStore.py          → incremental daily update
"""

import os
import sqlite3
import pandas as pd
from collectors import (
    fetch_crude_oil, fetch_inr_usd, fetch_fii_flow,
    fetch_geopolitics_sentiment, fetch_policy_sentiment,
    fetch_rbi_repo_rate, fetch_stock_target
)
from config import DB_PATH, HISTORY_DAYS


def _ensure_db_dir():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def _load_existing() -> pd.DataFrame:
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM features", conn, index_col="index")
        conn.close()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return pd.DataFrame()


def _save(df: pd.DataFrame):
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("features", conn, if_exists="replace")
    conn.close()


def _pct(s: pd.Series) -> pd.Series:
    return (s.pct_change()
             .replace([float("inf"), float("-inf")], pd.NA)
             .ffill()
             .fillna(0))


def build_feature_store(incremental: bool = True):
    _ensure_db_dir()

    days_arg = None if incremental else HISTORY_DAYS
    print(f"[featureStore] Mode: {'incremental' if incremental else 'full'}")
    print("Fetching data sources...")

    crude  = fetch_crude_oil(days=days_arg)
    inr    = fetch_inr_usd(days=days_arg)
    fii    = fetch_fii_flow(days=days_arg)
    geo    = fetch_geopolitics_sentiment(days=days_arg)
    policy = fetch_policy_sentiment(days=days_arg)
    repo   = fetch_rbi_repo_rate(days=days_arg)
    target = fetch_stock_target(days=days_arg)

    if target.empty:
        print("[featureStore] ERROR: TATASTEEL price fetch failed. Aborting.")
        return None

    print("Aligning to daily frequency...")

    bdays = pd.DatetimeIndex([pd.Timestamp(d) for d in target.index])
    bdays = bdays[~bdays.duplicated(keep="last")]

    def align(df, col, fill=0):
        if df.empty:
            print(f"  WARNING: {col} empty — carry-forward or fill {fill}")
            return pd.Series(dtype=float, name=col)
        s = df[col].copy()
        s.index = pd.DatetimeIndex([pd.Timestamp(d) for d in s.index])
        s = s[~s.index.duplicated(keep="last")]
        return s.reindex(bdays).ffill(limit=5)

    new = pd.DataFrame(index=bdays)
    new["crude_usd"]             = align(crude,  "crude_usd")
    new["inr_usd"]               = align(inr,    "inr_usd")
    new["fii_net_crore"]         = align(fii,    "fii_net_crore")
    new["fii_ls_ratio"]          = align(fii,    "fii_ls_ratio")
    new["fii_net_futures_value"] = align(fii,    "fii_net_futures_value")
    new["geo_score"]             = align(geo,    "geo_score")
    new["policy_score"]          = align(policy, "policy_score")
    new["rbi_repo_rate"]         = align(repo,   "rbi_repo_rate")

    # Derived features
    new["crude_chg_pct"]         = _pct(new["crude_usd"])
    new["inr_chg_pct"]           = _pct(new["inr_usd"])
    new["fii_3d_avg"]            = new["fii_net_crore"].rolling(3).mean().fillna(0)
    new["fii_ls_5d_avg"]         = new["fii_ls_ratio"].rolling(5).mean().fillna(1)
    new["geo_5d_avg"]            = new["geo_score"].rolling(5).mean().fillna(0)
    new["policy_5d_avg"]         = new["policy_score"].rolling(5).mean().fillna(0)
    new["repo_chg"]              = new["rbi_repo_rate"].diff().fillna(0)

    # Lag features — delayed causal impact
    new["crude_lag3"]            = new["crude_usd"].shift(3)        # crude takes ~3 days to reflect
    new["crude_lag7"]            = new["crude_usd"].shift(7)        # longer supply-chain lag
    new["inr_lag2"]              = new["inr_usd"].shift(2)          # forex settles T+2
    new["fii_lag1"]              = new["fii_net_crore"].shift(1)    # next-day FII impact
    new["fii_ls_lag2"]           = new["fii_ls_ratio"].shift(2)     # positioning lag
    new["geo_lag5"]              = new["geo_score"].shift(5)        # geopolitics slow-burn
    new["policy_lag10"]          = new["policy_score"].shift(10)    # policy takes time to price in
    new["repo_lag30"]            = new["rbi_repo_rate"].shift(30)   # rate hike impact ~1 month  # rate change signal

    new["target_return"]    = target["target_return"]
    new["target_direction"] = target["target_direction"]

    if incremental:
        existing = _load_existing()
        if not existing.empty:
            combined = pd.concat([existing, new])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()

            # Carry-forward any columns that failed in this run
            for col in ["crude_usd", "inr_usd", "fii_net_crore", "fii_ls_ratio",
                        "fii_net_futures_value", "geo_score", "policy_score", "rbi_repo_rate"]:
                if col in combined.columns:
                    combined[col] = combined[col].ffill(limit=5).fillna(0)

            # Recompute derived on full series
            combined["crude_chg_pct"]  = _pct(combined["crude_usd"])
            combined["inr_chg_pct"]    = _pct(combined["inr_usd"])
            combined["fii_3d_avg"]     = combined["fii_net_crore"].rolling(3).mean().fillna(0)
            combined["fii_ls_5d_avg"]  = combined["fii_ls_ratio"].rolling(5).mean().fillna(1)
            combined["geo_5d_avg"]     = combined["geo_score"].rolling(5).mean().fillna(0)
            combined["policy_5d_avg"]  = combined["policy_score"].rolling(5).mean().fillna(0)
            combined["repo_chg"]       = combined["rbi_repo_rate"].diff().fillna(0)
            combined["crude_lag3"]     = combined["crude_usd"].shift(3)
            combined["crude_lag7"]     = combined["crude_usd"].shift(7)
            combined["inr_lag2"]       = combined["inr_usd"].shift(2)
            combined["fii_lag1"]       = combined["fii_net_crore"].shift(1)
            combined["fii_ls_lag2"]    = combined["fii_ls_ratio"].shift(2)
            combined["geo_lag5"]       = combined["geo_score"].shift(5)
            combined["policy_lag10"]   = combined["policy_score"].shift(10)
            combined["repo_lag30"]     = combined["rbi_repo_rate"].shift(30)

            master = combined
        else:
            master = new
    else:
        master = new
        for col in ["crude_usd", "inr_usd", "fii_net_crore", "fii_ls_ratio",
                    "fii_net_futures_value", "geo_score", "policy_score", "rbi_repo_rate"]:
                if col in master.columns:
                    master[col] = master[col].ffill(limit=5).fillna(0)

    master.dropna(subset=["target_direction"], inplace=True)

    print(f"Feature store shape: {master.shape}")
    print(master.tail(3).to_string())

    _save(master)
    print(f"Saved to {DB_PATH}")
    return master


if __name__ == "__main__":
    import sys
    full = "--full" in sys.argv
    build_feature_store(incremental=not full)
