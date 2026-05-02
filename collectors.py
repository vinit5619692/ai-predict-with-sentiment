"""
collectors.py — optimized for CPU (Intel iGPU not supported by PyTorch)
- FinBERT on CPU with batch processing + SQLite sentiment cache (biggest speedup)
- yfinance with exponential backoff + nsepy fallback
- FII cache (csv) survives failed runs
"""

import hashlib
import sqlite3
import time
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from config import (
    NEWSDATA_API_KEY, GUARDIAN_API_KEY, EIA_API_KEY,
    ALPHA_VANTAGE_KEY, FRED_STLOUISFED_KEY,
    GEO_KEYWORDS, POLICY_KEYWORDS, HISTORY_DAYS,
    DB_PATH, CRUDE_CSV_PATH
)
try:
    from config import NEWSAPI_KEY
except ImportError:
    NEWSAPI_KEY = ""

# ═══════════════════════════════════════════
# FinBERT — CPU optimized, batch + SQLite cache
# Intel iGPU is NOT supported by PyTorch CUDA.
# The cache is the real speedup: same text never scored twice across runs.
# ═══════════════════════════════════════════
_finbert = None


def _ensure_sentiment_table():
    try:
        import os
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_cache (
                hash TEXT PRIMARY KEY,
                score REAL
            )
        """)
        conn.commit()
        conn.close()
    except Exception:
        pass


def _get_finbert():
    global _finbert
    if _finbert is None:
        try:
            import torch
            # Intel iGPU not supported — always CPU
            torch.set_num_threads(4)  # use 4 CPU threads for inference
            from transformers import pipeline
            print("[finbert] Loading ProsusAI/finbert on CPU (first run ~500MB download)...")
            _finbert = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                truncation=True,
                max_length=512,
                device=-1,  # -1 = CPU always
            )
            print("[finbert] Ready.")
        except Exception as e:
            print(f"[finbert] Load failed: {e}")
            _finbert = None
    return _finbert


def _hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _load_cached_scores(hashes: list) -> dict:
    if not hashes:
        return {}
    try:
        conn = sqlite3.connect(DB_PATH)
        placeholders = ",".join("?" * len(hashes))
        rows = conn.execute(
            f"SELECT hash, score FROM sentiment_cache WHERE hash IN ({placeholders})",
            hashes
        ).fetchall()
        conn.close()
        return {r[0]: r[1] for r in rows}
    except Exception:
        return {}


def _save_cached_scores(pairs: list):
    if not pairs:
        return
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.executemany(
            "INSERT OR REPLACE INTO sentiment_cache (hash, score) VALUES (?, ?)",
            pairs
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[finbert/cache] Save error: {e}")


def _finbert_score_batch(texts: list, batch_size: int = 16) -> list:
    """
    Score a list of texts. Checks SQLite cache first — uncached texts
    go through FinBERT in batches of 16 (safe for CPU RAM).
    Returns scores in same order as input.
    """
    _ensure_sentiment_table()
    texts = [str(t)[:512] for t in texts]
    hashes = [_hash(t) for t in texts]
    cached = _load_cached_scores(hashes)

    uncached_idx = [i for i, h in enumerate(hashes) if h not in cached]
    uncached_texts = [texts[i] for i in uncached_idx]

    if uncached_texts:
        model = _get_finbert()
        new_pairs = []
        if model is not None:
            for i in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[i:i + batch_size]
                try:
                    results = model(batch)
                    for j, res in enumerate(results):
                        label = res["label"].lower()
                        sc = res["score"]
                        score = sc if label == "positive" else (-sc if label == "negative" else 0.0)
                        h = _hash(batch[j])
                        cached[h] = round(score, 4)
                        new_pairs.append((h, round(score, 4)))
                except Exception as e:
                    print(f"[finbert] Batch {i} error: {e}")
                    for t in batch:
                        cached[_hash(t)] = 0.0
        else:
            for t in uncached_texts:
                cached[_hash(t)] = 0.0
        _save_cached_scores(new_pairs)
        cached_count = len(hashes) - len(uncached_idx)
        print(f"[finbert] {cached_count} from cache, {len(uncached_idx)} newly scored")

    return [cached.get(h, 0.0) for h in hashes]


def _finbert_score(text: str) -> float:
    return _finbert_score_batch([text])[0]


# ═══════════════════════════════════════════
# HELPER: last stored date for incremental
# ═══════════════════════════════════════════
def _last_stored_date(col: str) -> datetime:
    default = datetime.today() - timedelta(days=HISTORY_DAYS)
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql(
            f'SELECT MAX("index") as last FROM features WHERE "{col}" IS NOT NULL',
            conn
        )
        conn.close()
        val = df["last"].iloc[0]
        if val:
            return pd.to_datetime(val) + timedelta(days=1)
    except Exception:
        pass
    return default


def _clean(df) -> pd.DataFrame:
    if df.empty:
        return df
    return df[~df.index.duplicated(keep="last")].sort_index()


# ═══════════════════════════════════════════
# 1. CRUDE OIL — CSV seed + EIA + AV + stooq
# ═══════════════════════════════════════════
def fetch_crude_oil(days=None):
    start = _last_stored_date("crude_usd") if days is None else datetime.today() - timedelta(days=days)
    end   = datetime.today()
    csv_df = _crude_csv()
    api_df = _crude_eia(start, end)
    if api_df.empty:
        print("[crude] EIA failed, trying Alpha Vantage...")
        api_df = _crude_alphavantage(start, end)
    if api_df.empty:
        print("[crude] Alpha Vantage failed, trying stooq...")
        api_df = _crude_stooq(start, end)
    if not csv_df.empty and not api_df.empty:
        combined = pd.concat([csv_df, api_df])
        return _clean(combined[~combined.index.duplicated(keep="last")])
    return _clean(api_df) if not api_df.empty else _clean(csv_df)


def _crude_csv() -> pd.DataFrame:
    try:
        df = pd.read_csv(CRUDE_CSV_PATH)
        df = df.rename(columns={"Date": "date", "Price": "crude_usd"})[["date", "crude_usd"]]
        df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
        df["crude_usd"] = pd.to_numeric(df["crude_usd"].astype(str).str.replace(",", ""), errors="coerce")
        df = df.dropna().set_index("date")
        print(f"[crude/csv] Loaded {len(df)} rows")
        return df
    except Exception as e:
        print(f"[crude/csv] Skipped: {e}")
        return pd.DataFrame()


def _crude_eia(start, end) -> pd.DataFrame:
    url = (
        f"https://api.eia.gov/v2/petroleum/pri/spt/data/?api_key={EIA_API_KEY}"
        "&frequency=daily&data[0]=value&facets[series][]=RBRTE"
        f"&start={start.strftime('%Y-%m-%d')}&end={end.strftime('%Y-%m-%d')}"
        "&sort[0][column]=period&sort[0][direction]=asc&offset=0&length=5000"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()["response"]["data"]
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)[["period", "value"]].rename(columns={"period": "date", "value": "crude_usd"})
        df["date"] = pd.to_datetime(df["date"])
        df["crude_usd"] = pd.to_numeric(df["crude_usd"], errors="coerce")
        return df.dropna().set_index("date")
    except Exception as e:
        print(f"[crude/eia] Error: {e}")
        return pd.DataFrame()


def _crude_alphavantage(start, end) -> pd.DataFrame:
    url = f"https://www.alphavantage.co/query?function=BRENT&interval=daily&apikey={ALPHA_VANTAGE_KEY}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json().get("data", [])
        rows = [{"date": d["date"], "crude_usd": float(d["value"])}
                for d in data if d.get("value") not in (None, ".", "")]
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
        return df.dropna().set_index("date")
    except Exception as e:
        print(f"[crude/alphavantage] Error: {e}")
        return pd.DataFrame()


def _crude_stooq(start, end) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s=brt.f&d1={start.strftime('%Y%m%d')}&d2={end.strftime('%Y%m%d')}&i=d"
    try:
        df = pd.read_csv(url, parse_dates=["Date"])
        return df.rename(columns={"Date": "date", "Close": "crude_usd"})[["date", "crude_usd"]].dropna().set_index("date")
    except Exception as e:
        print(f"[crude/stooq] Error: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════
# 2. INR/USD — Alpha Vantage → frankfurter → stooq
# ═══════════════════════════════════════════
def fetch_inr_usd(days=None):
    start = _last_stored_date("inr_usd") if days is None else datetime.today() - timedelta(days=days)
    end   = datetime.today()
    df = _inr_alphavantage(start, end)
    if df.empty:
        print("[inr_usd] Alpha Vantage failed, trying frankfurter...")
        df = _inr_frankfurter(start, end)
    if df.empty:
        print("[inr_usd] frankfurter failed, trying stooq...")
        df = _inr_stooq(start, end)
    return _clean(df)


def _inr_alphavantage(start, end) -> pd.DataFrame:
    url = (f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol=USD"
           f"&to_symbol=INR&outputsize=full&apikey={ALPHA_VANTAGE_KEY}")
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        ts = r.json().get("Time Series FX (Daily)", {})
        rows = [{"date": d, "inr_usd": float(v["4. close"])} for d, v in ts.items()]
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
        return df.dropna().set_index("date")
    except Exception as e:
        print(f"[inr_usd/alphavantage] Error: {e}")
        return pd.DataFrame()


def _inr_frankfurter(start, end) -> pd.DataFrame:
    url = f"https://api.frankfurter.app/{start.strftime('%Y-%m-%d')}..{end.strftime('%Y-%m-%d')}?from=USD&to=INR"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json().get("rates", {})
        df = pd.DataFrame([{"date": d, "inr_usd": v["INR"]} for d, v in data.items()])
        df["date"] = pd.to_datetime(df["date"])
        return df.dropna().set_index("date")
    except Exception as e:
        print(f"[inr_usd/frankfurter] Error: {e}")
        return pd.DataFrame()


def _inr_stooq(start, end) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s=usdinr&d1={start.strftime('%Y%m%d')}&d2={end.strftime('%Y%m%d')}&i=d"
    try:
        df = pd.read_csv(url, parse_dates=["Date"])
        return df.rename(columns={"Date": "date", "Close": "inr_usd"})[["date", "inr_usd"]].dropna().set_index("date")
    except Exception as e:
        print(f"[inr_usd/stooq] Error: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════
# 3. FII/DII — nselib + CSV cache
# ═══════════════════════════════════════════
FII_CACHE = "fii_cache.csv"


def _load_fii_cache() -> pd.DataFrame:
    try:
        df = pd.read_csv(FII_CACHE, index_col="date", parse_dates=True)
        print(f"[fii/cache] Loaded {len(df)} cached rows")
        return df
    except Exception:
        return pd.DataFrame()


def _save_fii_cache(df: pd.DataFrame):
    try:
        existing = _load_fii_cache()
        if not existing.empty:
            df = pd.concat([existing, df])
            df = df[~df.index.duplicated(keep="last")].sort_index()
        df.to_csv(FII_CACHE, index_label="date")
    except Exception as e:
        print(f"[fii/cache] Save failed: {e}")


def fetch_fii_flow(days=None):
    cached = _load_fii_cache()
    if not cached.empty:
        start = cached.index.max() + timedelta(days=1)
    else:
        start = _last_stored_date("fii_net_crore") if days is None else datetime.today() - timedelta(days=days)
    end = datetime.today()

    if pd.Timestamp(start).date() >= pd.Timestamp(end).date():
        print("[fii] Cache up to date.")
        return _clean(cached) if not cached.empty else pd.DataFrame()

    print(f"[fii] Fetching {start.strftime('%d-%m-%Y')} → {end.strftime('%d-%m-%Y')}...")
    new_df = _fii_nselib(start, end)
    if new_df.empty:
        print("[fii] nselib failed, trying NSE API...")
        new_df = _fii_nse(start, end)

    if not new_df.empty:
        _save_fii_cache(new_df)
        combined = pd.concat([cached, new_df]) if not cached.empty else new_df
        return _clean(combined[~combined.index.duplicated(keep="last")])

    if not cached.empty:
        print("[fii] Fetch failed — using cache only.")
        return _clean(cached)

    print("[fii] All sources failed — carry-forward in featureStore.")
    return pd.DataFrame()


def _fii_nselib(start, end) -> pd.DataFrame:
    try:
        from nselib import derivatives, capital_market
    except ImportError:
        print("[fii/nselib] nselib not installed.")
        return pd.DataFrame()

    rows = []
    current = start if isinstance(start, datetime) else datetime.combine(start, datetime.min.time())
    end_dt  = end   if isinstance(end,   datetime) else datetime.combine(end,   datetime.min.time())

    while current <= end_dt:
        if current.weekday() < 5:
            d_str = current.strftime("%d-%m-%Y")
            try:
                df_oi     = derivatives.participant_wise_open_interest(d_str)
                fii_row   = df_oi[df_oi['Client Type'] == 'FII'].iloc[0]
                fii_long  = float(str(fii_row['Future Index Long']).replace(',', '') or 0)
                fii_short = float(str(fii_row['Future Index Short']).replace(',', '') or 0)
                ls_ratio  = fii_long / fii_short if fii_short != 0 else 1.0

                df_stats    = derivatives.fii_derivatives_statistics(d_str)
                fut_mask    = df_stats['fii_derivatives'].str.contains('FUTURES', case=False, na=False)
                buy_vals    = pd.to_numeric(df_stats.loc[fut_mask, 'buy_value_in_Cr'].astype(str).str.replace(',', ''), errors='coerce')
                sell_vals   = pd.to_numeric(df_stats.loc[fut_mask, 'sell_value_in_Cr'].astype(str).str.replace(',', ''), errors='coerce')
                net_futures = float(buy_vals.sum() - sell_vals.sum())

                try:
                    df_cash   = capital_market.fii_dii_trading_activity(d_str)
                    cat_col   = df_cash.columns[0]
                    fii_cash  = df_cash[df_cash[cat_col].astype(str).str.contains('FII|FPI', case=False)]
                    net_col   = [c for c in df_cash.columns if 'net' in c.lower()][0]
                    net_crore = float(str(fii_cash[net_col].iloc[0]).replace(',', '') or 0)
                except Exception:
                    net_crore = 0.0

                rows.append({
                    "date": pd.to_datetime(current.date()),
                    "fii_net_crore": net_crore,
                    "fii_ls_ratio": ls_ratio,
                    "fii_net_futures_value": net_futures,
                })
                print(f"  [fii/nselib] ✅ {d_str}")
            except Exception as e:
                print(f"  [fii/nselib] ❌ {d_str}: {e}")
            time.sleep(1)
        current += timedelta(days=1)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date")


def _fii_nse(start, end) -> pd.DataFrame:
    url = (
        "https://www.nseindia.com/api/fiidiiTradeReact"
        f"?startDate={start.strftime('%d-%m-%Y')}&endDate={end.strftime('%d-%m-%Y')}"
    )
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com"}
    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        r = session.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        rows = [{
            "date": pd.to_datetime(item["date"], format="%d-%b-%Y"),
            "fii_net_crore": float(str(item.get("fiinet", "0")).replace(",", "") or 0),
            "fii_ls_ratio": 1.0,
            "fii_net_futures_value": 0.0,
        } for item in r.json()]
        return pd.DataFrame(rows).set_index("date").sort_index()
    except Exception as e:
        print(f"[fii/nse] Error: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════
# 4. GEOPOLITICS — Guardian → NewsAPI
#    Batch-scores all articles at once before grouping
# ═══════════════════════════════════════════
def fetch_geopolitics_sentiment(days=None):
    start = _last_stored_date("geo_score") if days is None else datetime.today() - timedelta(days=days)
    df = _geo_guardian(start)
    if df.empty:
        print("[geo] Guardian failed, trying NewsAPI...")
        df = _geo_newsapi(start)
    return _clean(df)


def _geo_guardian(start) -> pd.DataFrame:
    query = " OR ".join(GEO_KEYWORDS)
    url = (
        f"https://content.guardianapis.com/search"
        f"?q={requests.utils.quote(query)}&from-date={start.strftime('%Y-%m-%d')}"
        f"&api-key={GUARDIAN_API_KEY}&page-size=200"
        f"&show-fields=headline,bodyText&order-by=newest"
    )
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        resp = r.json().get("response", {})
        if resp.get("status") != "ok":
            return pd.DataFrame()
        articles = resp.get("results", [])
        if not articles:
            return pd.DataFrame()
        texts = [
            (a.get("fields", {}).get("headline", "") + " " +
             a.get("fields", {}).get("bodyText", "")[:500])
            for a in articles
        ]
        dates  = [a["webPublicationDate"][:10] for a in articles]
        scores = _finbert_score_batch(texts)  # single batch call
        df = pd.DataFrame({"date": dates, "geo_score": scores})
        df["date"] = pd.to_datetime(df["date"])
        return df.groupby("date")["geo_score"].mean().to_frame()
    except Exception as e:
        print(f"[geo/guardian] Error: {e}")
        return pd.DataFrame()


def _geo_newsapi(start) -> pd.DataFrame:
    if not NEWSAPI_KEY:
        return pd.DataFrame()
    url = (
        f"https://newsapi.org/v2/everything"
        f"?q={requests.utils.quote(' OR '.join(GEO_KEYWORDS[:3]))}"
        f"&from={start.strftime('%Y-%m-%d')}&language=en&sortBy=publishedAt"
        f"&apiKey={NEWSAPI_KEY}&pageSize=100"
    )
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        articles = r.json().get("articles", [])
        if not articles:
            return pd.DataFrame()
        texts  = [a.get("title", "") + " " + (a.get("description") or "") for a in articles]
        dates  = [(a.get("publishedAt") or "")[:10] for a in articles]
        scores = _finbert_score_batch(texts)
        df = pd.DataFrame({"date": dates, "geo_score": scores})
        df["date"] = pd.to_datetime(df["date"])
        return df.groupby("date")["geo_score"].mean().to_frame()
    except Exception as e:
        print(f"[geo/newsapi] Error: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════
# 5. POLICY — NewsData.io → Guardian
# ═══════════════════════════════════════════
def fetch_policy_sentiment(days=None):
    start = _last_stored_date("policy_score") if days is None else datetime.today() - timedelta(days=days)
    df = _policy_newsdata(start)
    if df.empty:
        print("[policy] NewsData.io failed, trying Guardian...")
        df = _policy_guardian(start)
    return _clean(df)


def _policy_newsdata(start) -> pd.DataFrame:
    url = (
        f"https://newsdata.io/api/1/news"
        f"?apikey={NEWSDATA_API_KEY}"
        f"&q={requests.utils.quote(' OR '.join(POLICY_KEYWORDS[:4]))}"
        f"&country=in&language=en&category=business,politics"
    )
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        body = r.json()
        if body.get("status") == "error":
            print(f"[policy/newsdata] {body.get('results', {}).get('message', '')}")
            return pd.DataFrame()
        articles = [
            a for a in body.get("results", [])
            if (a.get("pubDate") or "")[:10]
            and pd.to_datetime((a.get("pubDate") or "")[:10]) >= pd.Timestamp(start)
        ]
        if not articles:
            return pd.DataFrame()
        texts  = [a.get("title", "") + " " + (a.get("description") or "")[:300] for a in articles]
        dates  = [(a.get("pubDate") or "")[:10] for a in articles]
        scores = _finbert_score_batch(texts)
        df = pd.DataFrame({"date": dates, "policy_score": scores})
        df["date"] = pd.to_datetime(df["date"])
        return df.groupby("date")["policy_score"].mean().to_frame()
    except Exception as e:
        print(f"[policy/newsdata] Error: {e}")
        return pd.DataFrame()


def _policy_guardian(start) -> pd.DataFrame:
    query = " OR ".join(POLICY_KEYWORDS[:3])
    url = (
        f"https://content.guardianapis.com/search"
        f"?q={requests.utils.quote(query)}&from-date={start.strftime('%Y-%m-%d')}"
        f"&api-key={GUARDIAN_API_KEY}&page-size=200"
        f"&show-fields=headline,bodyText&section=business,politics&order-by=newest"
    )
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        resp = r.json().get("response", {})
        if resp.get("status") != "ok":
            return pd.DataFrame()
        articles = resp.get("results", [])
        if not articles:
            return pd.DataFrame()
        texts  = [
            a.get("fields", {}).get("headline", "") + " " +
            a.get("fields", {}).get("bodyText", "")[:500]
            for a in articles
        ]
        dates  = [a["webPublicationDate"][:10] for a in articles]
        scores = _finbert_score_batch(texts)
        df = pd.DataFrame({"date": dates, "policy_score": scores})
        df["date"] = pd.to_datetime(df["date"])
        return df.groupby("date")["policy_score"].mean().to_frame()
    except Exception as e:
        print(f"[policy/guardian] Error: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════
# 6. RBI REPO RATE — FRED → hardcoded table
# ═══════════════════════════════════════════
def fetch_rbi_repo_rate(days=None):
    start = _last_stored_date("rbi_repo_rate") if days is None else datetime.today() - timedelta(days=days)
    end   = datetime.today()
    df = _repo_fred(start, end)
    if df.empty:
        print("[repo] FRED failed, using hardcoded RBI table...")
        df = _repo_rbi_scrape()
    return _clean(df)


def _repo_fred(start, end) -> pd.DataFrame:
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id=IRSTCI01INM156N&api_key={FRED_STLOUISFED_KEY}&file_type=json"
        f"&observation_start={start.strftime('%Y-%m-%d')}"
        f"&observation_end={end.strftime('%Y-%m-%d')}"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        rows = [{"date": o["date"], "rbi_repo_rate": float(o["value"])}
                for o in obs if o.get("value") not in (".", None, "")]
        if rows:
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            return df.set_index("date")
    except Exception as e:
        print(f"[repo/fred] Error: {e}")
    return pd.DataFrame()


def _repo_rbi_scrape() -> pd.DataFrame:
    """Hardcoded RBI repo rate change dates. Update when RBI announces new rate."""
    rbi_rates = [
        ("2020-03-27", 4.40), ("2020-05-22", 4.00),
        ("2022-05-04", 4.40), ("2022-06-08", 4.90),
        ("2022-08-05", 5.40), ("2022-09-30", 5.90),
        ("2022-12-07", 6.25), ("2023-02-08", 6.50),
        ("2024-02-08", 6.50), ("2024-04-05", 6.50),
        ("2024-06-07", 6.50), ("2024-08-08", 6.50),
        ("2024-10-09", 6.50), ("2024-12-06", 6.50),
        ("2025-02-07", 6.25), ("2025-04-09", 6.00),
    ]
    df = pd.DataFrame(rbi_rates, columns=["date", "rbi_repo_rate"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


# ═══════════════════════════════════════════
# 7. TATASTEEL — yfinance with exponential backoff
#    yfinance rate-limits aggressively after long runs.
#    Waits up to 3 min total before giving up.
# ═══════════════════════════════════════════
def fetch_stock_target(days=None):
    if days is None:
        start     = _last_stored_date("target_direction")
        kwargs    = {
            "start": start.strftime("%Y-%m-%d"),
            "end":   (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        }
    else:
        kwargs = {"period": f"{days}d"}

    ticker = yf.Ticker("TATASTEEL.NS")
    df = pd.DataFrame()
    waits = [60, 90, 120]  # seconds between retries — longer waits work better

    for attempt, wait in enumerate(waits, 1):
        try:
            df = ticker.history(**kwargs)[["Close"]].copy()
            if not df.empty:
                break
        except Exception as e:
            print(f"[target] Attempt {attempt} failed: {e}. Waiting {wait}s...")
            time.sleep(wait)
            ticker = yf.Ticker("TATASTEEL.NS")  # fresh ticker object each retry

    if df.empty:
        print("[target] yfinance failed after all retries.")
        return pd.DataFrame()

    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df.index = pd.to_datetime(df.index.date)
    df["target_return"]    = df["Close"].pct_change().shift(-1)
    df["target_direction"] = (df["target_return"] > 0).astype(int)
    return df[["target_return", "target_direction"]]
