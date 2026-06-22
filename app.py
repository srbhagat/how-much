
import os
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="How Much Possible Which Side", page_icon="₿", layout="wide")

COINDCX_CANDLES = "https://public.coindcx.com/market_data/candles/"
COINDCX_FUTURES_TRADES = "https://api.coindcx.com/exchange/v1/derivatives/futures/data/trades"

PAIR = "B-BTC_USDT"
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
CACHE_FILE = os.path.join(DATA_DIR, "coindcx_btcusdt_side_points_one_year.csv")

STEP_MS = 15 * 60 * 1000

def clean_datetime_column(df):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if "open_time" in df.columns:
        df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
        df = df.dropna(subset=["open_time"])
        if df.empty:
            return pd.DataFrame()
        df["open_time"] = df["open_time"].astype("int64")
        df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True, errors="coerce")
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df = df.dropna(subset=["datetime"])
        if df.empty:
            return pd.DataFrame()
        df["open_time"] = (df["datetime"].astype("int64") // 10**6)
    else:
        return pd.DataFrame()

    df = df.dropna(subset=["datetime", "open_time"])
    if df.empty:
        return pd.DataFrame()

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce").dt.tz_convert("Asia/Kolkata")
    df = df.dropna(subset=["datetime"])

    return df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)

def load_cache():
    if not os.path.exists(CACHE_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(CACHE_FILE)
        return clean_datetime_column(df)
    except Exception:
        return pd.DataFrame()

def save_cache(df):
    df = clean_datetime_column(df)
    if not df.empty:
        df.to_csv(CACHE_FILE, index=False)

def normalize_candles(data):
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if "time" in df.columns:
        df = df.rename(columns={"time": "open_time"})
    elif "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "open_time"})

    required = ["open_time", "open", "high", "low", "close", "volume"]
    if any(c not in df.columns for c in required):
        return pd.DataFrame()

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=required)
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
    df = df[(df["high"] >= df[["open", "close", "low"]].max(axis=1)) &
            (df["low"] <= df[["open", "close", "high"]].min(axis=1))]

    df["open_time"] = df["open_time"].astype("int64")
    df = clean_datetime_column(df)
    if df.empty:
        return pd.DataFrame()

    df["quote_volume"] = np.nan
    df["trades"] = np.nan

    return df[["open_time", "datetime", "open", "high", "low", "close", "volume", "quote_volume", "trades"]].drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)

def fetch_coindcx_candles(start_ms=None, end_ms=None, limit=1000):
    params = {"pair": PAIR, "interval": "15m", "limit": min(int(limit), 1000)}
    if start_ms is not None:
        params["startTime"] = int(start_ms)
    if end_ms is not None:
        params["endTime"] = int(end_ms)

    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(COINDCX_CANDLES, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    return normalize_candles(r.json())

def fetch_live_price():
    try:
        r = requests.get(COINDCX_FUTURES_TRADES, params={"pair": PAIR}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            return float(data[0]["price"]), "CoinDCX futures live"
    except Exception:
        pass

    try:
        d = fetch_coindcx_candles(limit=2)
        if not d.empty:
            return float(d["close"].iloc[-1]), "CoinDCX latest candle"
    except Exception:
        pass

    return np.nan, "No live price"

def build_or_update_database(years_requested=2, force=False):

    status = st.empty()
    progress = st.empty()

    target_start = pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=years_requested)
    target_start_ms = int(target_start.timestamp() * 1000)
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)

    existing = pd.DataFrame() if force else load_cache()

    if not existing.empty and len(existing) < 20000:
        status.warning(f"Small cache detected ({len(existing):,} candles). Auto-rebuilding database.")
        existing = pd.DataFrame()

    frames = []

    if not existing.empty:
        frames.append(existing)
        existing_min = int(existing["open_time"].min())
        existing_max = int(existing["open_time"].max())
        status.info(f"Cache found: {len(existing):,} candles from {existing['datetime'].min()} to {existing['datetime'].max()}")
    else:
        existing_min = None
        existing_max = None

    try:
        latest = fetch_coindcx_candles(limit=1000)
        if not latest.empty:
            frames.append(latest)
    except Exception as e:
        st.warning(f"Latest CoinDCX candle fetch failed: {e}")

    window_ms = STEP_MS * 1000  # 1000 candles per request
    cursor = target_start_ms
    total_span = max(1, now_ms - target_start_ms)
    batch = 0
    empty_batches = 0
    downloaded_rows = 0

    while cursor < now_ms:
        end_ms = min(cursor + window_ms - STEP_MS, now_ms)

        if existing_min is not None and existing_max is not None:
            if cursor >= existing_min and end_ms <= existing_max:
                cursor = end_ms + STEP_MS
                continue

        try:
            chunk = fetch_coindcx_candles(start_ms=cursor, end_ms=end_ms, limit=1000)
        except Exception as e:
            status.warning(f"CoinDCX history fetch stopped at batch {batch}: {e}")
            break

        batch += 1

        if chunk.empty:
            empty_batches += 1
        else:
            empty_batches = 0
            downloaded_rows += len(chunk)
            frames.append(chunk)

        if batch % 5 == 0 and frames:
            temp = pd.concat(frames, ignore_index=True).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
            save_cache(temp)

        done = min(1.0, max(0.0, (cursor - target_start_ms) / total_span))
        candle_count = 0
        if frames:
            try:
                candle_count = len(pd.concat(frames, ignore_index=True).drop_duplicates("open_time"))
            except Exception:
                candle_count = 0

        progress.progress(
            done,
            text=f"CoinDCX forward history: batch {batch}, candles {candle_count:,}, empty windows {empty_batches}"
        )

        if empty_batches >= 20 and not frames:
            status.warning("CoinDCX returned many empty windows. Using available data.")
            break

        cursor = end_ms + STEP_MS
        time.sleep(0.04)

    if not frames:
        st.error("CoinDCX did not return candle data.")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)

    df = df[(df["open_time"] >= target_start_ms) & (df["open_time"] <= now_ms)].copy()
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)

    save_cache(df)

    latest_dt = pd.to_datetime(df["datetime"].max())
    age_minutes = (pd.Timestamp.now(tz=latest_dt.tz) - latest_dt).total_seconds() / 60

    expected = min(int(years_requested * 365 * 96), len(df)) if years_requested == 1 else int(years_requested * 365 * 96)
    coverage = len(df) / expected * 100 if expected else 0

    if age_minutes <= 120:
        status.success(f"CoinDCX database ready with {len(df):,} candles | coverage {coverage:.1f}% | latest {latest_dt}")
    else:
        status.warning(f"CoinDCX database may be stale. Latest candle {latest_dt} ({age_minutes:.0f} min old)")

    if len(df) < 5000:
        st.warning(
            f"CoinDCX returned only {len(df):,} candles. "
            "This may mean CoinDCX public history is limited for this pair/timeframe."
        )

    return df

def resample_ohlcv(df, rule):
    d = clean_datetime_column(df)
    if d.empty:
        return pd.DataFrame()
    d = d.set_index("datetime").sort_index()

    out = pd.DataFrame()
    out["open"] = d["open"].resample(rule).first()
    out["high"] = d["high"].resample(rule).max()
    out["low"] = d["low"].resample(rule).min()
    out["close"] = d["close"].resample(rule).last()
    out["volume"] = d["volume"].resample(rule).sum()
    out = out.dropna().reset_index()
    out["open_time"] = (pd.to_datetime(out["datetime"], utc=True, errors="coerce").astype("int64") // 10**6)
    out["quote_volume"] = np.nan
    out["trades"] = np.nan
    return out.reset_index(drop=True)

@st.cache_data(show_spinner=False)
def add_indicators(df):
    df = clean_datetime_column(df)
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume", "datetime"])
    if len(df) < 250:
        return pd.DataFrame()

    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()

    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    atr_sum = df["atr14"].rolling(14).sum()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14).sum() / atr_sum
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14).sum() / atr_sum
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    df["adx14"] = dx.rolling(14).mean()

    typical = (df["high"] + df["low"] + df["close"]) / 3
    day = pd.to_datetime(df["datetime"], utc=True, errors="coerce").dt.tz_convert("Asia/Kolkata").dt.date
    df["vwap"] = (typical * df["volume"]).groupby(day).cumsum() / df["volume"].groupby(day).cumsum()

    df["vol_ratio"] = df["volume"] / df["volume"].rolling(50).mean()
    df["trend"] = np.where(df["ema20"] > df["ema50"], "Bullish", "Bearish")
    df["major_trend"] = np.where(df["ema50"] > df["ema200"], "Bullish", "Bearish")

    hour = pd.to_datetime(df["datetime"]).dt.hour
    df["india_session"] = np.select(
        [(hour >= 5) & (hour < 12), (hour >= 12) & (hour < 18), (hour >= 18) & (hour < 24)],
        ["Asia/India Morning", "London Open", "New York"],
        default="Late US"
    )
    required_clean_cols = [
        "datetime", "open", "high", "low", "close", "volume",
        "ema20", "ema50", "ema200", "rsi14", "atr14", "adx14",
        "vwap", "vol_ratio", "trend", "major_trend", "india_session"
    ]
    return df.dropna(subset=required_clean_cols).reset_index(drop=True)

def mtf_bias(dfs):
    votes = []
    for tf, raw in dfs.items():
        d = add_indicators(raw)
        if d.empty:
            continue
        x = d.iloc[-1]
        score = 0
        score += 1 if x["ema20"] > x["ema50"] else -1
        score += 1 if x["close"] > x["ema200"] else -1
        score += 1 if x["close"] > x["vwap"] else -1
        votes.append((tf, "Bullish" if score > 0 else "Bearish" if score < 0 else "Neutral"))

    bull = sum(1 for _, v in votes if v == "Bullish")
    bear = sum(1 for _, v in votes if v == "Bearish")
    return ("Bullish" if bull > bear else "Bearish" if bear > bull else "Mixed"), votes

def structure_levels(df, live_price):
    d = df.tail(600).reset_index(drop=True)
    highs, lows = [], []
    for i in range(3, len(d)-3):
        if d.loc[i, "high"] == d.loc[i-3:i+3, "high"].max():
            highs.append(float(d.loc[i, "high"]))
        if d.loc[i, "low"] == d.loc[i-3:i+3, "low"].min():
            lows.append(float(d.loc[i, "low"]))

    structure = "Mixed"
    if len(highs) >= 2 and len(lows) >= 2:
        if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
            structure = "Bullish HH-HL"
        elif highs[-1] < highs[-2] and lows[-1] < lows[-2]:
            structure = "Bearish LH-LL"

    swing_lows = sorted([x for x in lows if x < live_price], reverse=True)
    swing_highs = sorted([x for x in highs if x > live_price])

    atr = float(d["atr14"].iloc[-1]) if "atr14" in d.columns and pd.notna(d["atr14"].iloc[-1]) else max(300, live_price * 0.004)
    support = swing_lows[0] if swing_lows else live_price - max(300, atr * 2)
    resistance = swing_highs[0] if swing_highs else live_price + max(300, atr * 2)

    return structure, float(support), float(resistance), "CoinDCX recent swing levels"

def advanced_market_context(df, live_price):
    d = df.tail(480).reset_index(drop=True)
    if len(d) < 80:
        return {"bos": "None", "choch": "None", "sweep": "None", "volume_spike": "No", "volume_zone": "Unknown"}

    swing_highs, swing_lows = [], []
    for i in range(3, len(d)-3):
        if d.loc[i, "high"] == d.loc[i-3:i+3, "high"].max():
            swing_highs.append((i, float(d.loc[i, "high"])))
        if d.loc[i, "low"] == d.loc[i-3:i+3, "low"].min():
            swing_lows.append((i, float(d.loc[i, "low"])))

    close = float(d["close"].iloc[-1])
    prev_high = max([x[1] for x in swing_highs[-8:]], default=float(d["high"].tail(80).max()))
    prev_low = min([x[1] for x in swing_lows[-8:]], default=float(d["low"].tail(80).min()))

    bos = "Bullish BOS" if close > prev_high else "Bearish BOS" if close < prev_low else "None"

    choch = "None"
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        hh = swing_highs[-1][1] > swing_highs[-2][1]
        hl = swing_lows[-1][1] > swing_lows[-2][1]
        lh = swing_highs[-1][1] < swing_highs[-2][1]
        ll = swing_lows[-1][1] < swing_lows[-2][1]
        if hh and hl and close < prev_low:
            choch = "Bearish CHOCH"
        elif lh and ll and close > prev_high:
            choch = "Bullish CHOCH"

    last = d.iloc[-1]
    prev_80_high = float(d["high"].iloc[:-1].tail(80).max())
    prev_80_low = float(d["low"].iloc[:-1].tail(80).min())
    sweep = "None"
    if last["high"] > prev_80_high and last["close"] < prev_80_high:
        sweep = "Bearish sweep"
    elif last["low"] < prev_80_low and last["close"] > prev_80_low:
        sweep = "Bullish sweep"

    vol_avg = float(d["volume"].tail(50).mean())
    vol_now = float(d["volume"].iloc[-1])
    volume_spike = "Yes" if vol_avg > 0 and vol_now >= vol_avg * 1.75 else "No"

    recent = d.tail(320).copy()
    atr = float(recent["atr14"].iloc[-1]) if "atr14" in recent.columns and pd.notna(recent["atr14"].iloc[-1]) else max(100.0, live_price * 0.002)
    bucket = max(50, round(atr / 2 / 50) * 50)
    recent["price_bucket"] = (recent["close"] / bucket).round() * bucket
    vp = recent.groupby("price_bucket")["volume"].sum().sort_values(ascending=False)

    volume_zone = "Neutral"
    if not vp.empty:
        buckets = list(vp.index.astype(float))
        below = sorted([b for b in buckets if b < live_price], reverse=True)
        above = sorted([b for b in buckets if b > live_price])
        if below and abs(live_price - below[0]) <= bucket:
            volume_zone = "Near HV support"
        if above and abs(live_price - above[0]) <= bucket:
            volume_zone = "Near HV resistance"

    return {"bos": bos, "choch": choch, "sweep": sweep, "volume_spike": volume_spike, "volume_zone": volume_zone}

def first_hit(df, idx, direction, tp_points, sl_points, horizon):
    entry = df.loc[idx, "close"]
    fut = df.iloc[idx+1:idx+1+horizon]
    if fut.empty:
        return None
    if direction == "LONG":
        tp, sl = entry + tp_points, entry - sl_points
        for _, r in fut.iterrows():
            if r["high"] >= tp and r["low"] <= sl:
                return "Ambiguous"
            if r["high"] >= tp:
                return "Win"
            if r["low"] <= sl:
                return "Loss"
    else:
        tp, sl = entry - tp_points, entry + sl_points
        for _, r in fut.iterrows():
            if r["low"] <= tp and r["high"] >= sl:
                return "Ambiguous"
            if r["low"] <= tp:
                return "Win"
            if r["high"] >= sl:
                return "Loss"
    return "No hit"

def recency_weight(dt):
    now = pd.Timestamp.now(tz="Asia/Kolkata")
    d = pd.to_datetime(dt)
    if d.tzinfo is None:
        d = d.tz_localize("Asia/Kolkata")
    age_days = (now - d).days
    if age_days <= 90:
        return 1.0
    if age_days <= 365:
        return 0.7
    if age_days <= 365 * 2:
        return 0.35
    return 0.15

def analyze(df, price, direction, tp, sl, horizon, tol, session_match):

    latest = df.iloc[-1]

    match_steps = [
        {
            "name": "Strict",
            "price_tol": tol,
            "rsi_band": 12,
            "atr_low": 0.45,
            "atr_high": 1.65,
            "adx_band": 12,
            "vol_low": 0.35,
            "vol_high": 2.25,
            "trend": True,
            "major_trend": True,
            "session": session_match,
        },
        {
            "name": "Relaxed",
            "price_tol": max(tol, 5.0),
            "rsi_band": 18,
            "atr_low": 0.30,
            "atr_high": 2.20,
            "adx_band": 18,
            "vol_low": 0.20,
            "vol_high": 3.50,
            "trend": True,
            "major_trend": False,
            "session": False,
        },
        {
            "name": "Wide",
            "price_tol": max(tol, 8.0),
            "rsi_band": 25,
            "atr_low": 0.20,
            "atr_high": 3.00,
            "adx_band": 25,
            "vol_low": 0.10,
            "vol_high": 5.00,
            "trend": False,
            "major_trend": False,
            "session": False,
        },
    ]

    selected = pd.DataFrame()
    match_mode = "No match"

    for step in match_steps:
        s = df[(df["close"] >= price*(1-step["price_tol"]/100)) & (df["close"] <= price*(1+step["price_tol"]/100))].copy()

        if step["trend"]:
            s = s[s["trend"] == latest["trend"]]
        if step["major_trend"]:
            s = s[s["major_trend"] == latest["major_trend"]]

        s = s[(s["rsi14"] >= latest["rsi14"]-step["rsi_band"]) & (s["rsi14"] <= latest["rsi14"]+step["rsi_band"])]
        s = s[(s["atr14"] >= latest["atr14"]*step["atr_low"]) & (s["atr14"] <= latest["atr14"]*step["atr_high"])]
        s = s[(s["adx14"] >= latest["adx14"]-step["adx_band"]) & (s["adx14"] <= latest["adx14"]+step["adx_band"])]
        s = s[(s["vol_ratio"] >= latest["vol_ratio"]*step["vol_low"]) & (s["vol_ratio"] <= latest["vol_ratio"]*step["vol_high"])]

        if step["session"]:
            s = s[s["india_session"] == latest["india_session"]]

        s = s[s.index <= len(df)-horizon-2]

        if len(s) >= 50 or step["name"] == "Wide":
            selected = s
            match_mode = step["name"]
            break

    outcomes, weights = [], []
    for idx in selected.index:
        outcome = first_hit(df, idx, direction, tp, sl, horizon)
        if outcome:
            outcomes.append(outcome)
            weights.append(recency_weight(df.loc[idx, "datetime"]))

    if not outcomes:
        return {"direction": direction, "matches": 0, "wins": 0, "losses": 0,
                "no_hit": 0, "ambiguous": 0, "decided": 0, "probability": 0.0,
                "weighted_probability": 0.0, "weighted_decided": 0.0,
                "match_mode": match_mode}

    ser = pd.Series(outcomes)
    w = pd.Series(weights)
    wins = int((ser == "Win").sum())
    losses = int((ser == "Loss").sum())
    no_hit = int((ser == "No hit").sum())
    ambiguous = int((ser == "Ambiguous").sum())
    decided = wins + losses
    raw_prob = wins / decided * 100 if decided else 0.0

    win_w = float(w[ser == "Win"].sum())
    loss_w = float(w[ser == "Loss"].sum())
    weighted_decided = win_w + loss_w
    weighted_prob = win_w / weighted_decided * 100 if weighted_decided else 0.0

    return {"direction": direction, "matches": len(outcomes), "wins": wins, "losses": losses,
            "no_hit": no_hit, "ambiguous": ambiguous, "decided": decided,
            "probability": raw_prob, "weighted_probability": weighted_prob,
            "weighted_decided": weighted_decided, "match_mode": match_mode}

def sample_confidence_label(decided):
    if decided >= 250:
        return "High sample"
    if decided >= 100:
        return "Good sample"
    if decided >= 50:
        return "Moderate sample"
    if decided >= 20:
        return "Low sample"
    return "Very low sample"

def grade_from_score(score, probability, decided):
    if decided < 20:
        return "D - Very low sample"
    if score >= 80 and probability >= 68:
        return "A - Strong"
    if score >= 60 and probability >= 62:
        return "B - Good"
    if score >= 40 and probability >= 55:
        return "C - Watch"
    return "D - Weak"

def score_side(direction, res, latest, mtf, structure, support, resistance, price, tp, ctx):
    reasons = []
    p = res.get("weighted_probability", res["probability"])
    d = res["decided"]

    if p >= 72:
        prob_score = 50; reasons.append("Excellent weighted probability")
    elif p >= 68:
        prob_score = 44; reasons.append("Very strong weighted probability")
    elif p >= 64:
        prob_score = 38; reasons.append("Strong weighted probability")
    elif p >= 60:
        prob_score = 30; reasons.append("Acceptable weighted probability")
    elif p >= 55:
        prob_score = 20; reasons.append("Mild weighted edge")
    elif p >= 52:
        prob_score = 10; reasons.append("Small weighted edge")
    else:
        prob_score = 0; reasons.append("Probability too weak")

    if d >= 250:
        sample_score = 20; reasons.append("High sample size")
    elif d >= 100:
        sample_score = 16; reasons.append("Good sample size")
    elif d >= 50:
        sample_score = 12; reasons.append("Moderate sample size")
    elif d >= 20:
        sample_score = 6; reasons.append("Low but usable sample size")
    else:
        sample_score = 0; reasons.append("Very low sample size")

    trend_score = 0
    if (direction == "LONG" and mtf == "Bullish") or (direction == "SHORT" and mtf == "Bearish"):
        trend_score += 7; reasons.append("MTF aligned")
    if (direction == "LONG" and "Bullish" in structure) or (direction == "SHORT" and "Bearish" in structure):
        trend_score += 5; reasons.append("Swing structure aligned")
    if latest["adx14"] >= 22:
        trend_score += 3; reasons.append("Trend strength acceptable")

    confirm_score = 0
    if (direction == "LONG" and price > latest["vwap"]) or (direction == "SHORT" and price < latest["vwap"]):
        confirm_score += 4; reasons.append("VWAP aligned")
    if direction == "LONG" and ctx.get("bos") == "Bullish BOS":
        confirm_score += 4; reasons.append("Bullish BOS")
    if direction == "SHORT" and ctx.get("bos") == "Bearish BOS":
        confirm_score += 4; reasons.append("Bearish BOS")
    if direction == "LONG" and ctx.get("choch") == "Bullish CHOCH":
        confirm_score += 4; reasons.append("Bullish CHOCH")
    if direction == "SHORT" and ctx.get("choch") == "Bearish CHOCH":
        confirm_score += 4; reasons.append("Bearish CHOCH")
    if direction == "LONG" and ctx.get("sweep") == "Bullish sweep":
        confirm_score += 5; reasons.append("Bullish liquidity sweep")
    if direction == "SHORT" and ctx.get("sweep") == "Bearish sweep":
        confirm_score += 5; reasons.append("Bearish liquidity sweep")
    if ctx.get("volume_spike") == "Yes":
        confirm_score += 2; reasons.append("Volume expansion present")
    if direction == "LONG" and ctx.get("volume_zone") == "Near HV support":
        confirm_score += 4; reasons.append("Near high-volume support")
    if direction == "SHORT" and ctx.get("volume_zone") == "Near HV resistance":
        confirm_score += 4; reasons.append("Near high-volume resistance")
    confirm_score = min(15, confirm_score)

    raw_score = prob_score + sample_score + trend_score + confirm_score

    if direction == "LONG" and (resistance - price) < tp:
        raw_score -= 20; reasons.append("Resistance too close for fixed 300 TP")
    if direction == "SHORT" and (price - support) < tp:
        raw_score -= 20; reasons.append("Support too close for fixed 300 TP")
    if direction == "LONG" and ctx.get("sweep") == "Bearish sweep":
        raw_score -= 10; reasons.append("Opposite bearish sweep")
    if direction == "SHORT" and ctx.get("sweep") == "Bullish sweep":
        raw_score -= 10; reasons.append("Opposite bullish sweep")

    final_score = max(0, min(100, int(round(raw_score))))
    return final_score, reasons, grade_from_score(final_score, p, d), sample_confidence_label(d)

def decide(lr, sr, ls, ss, min_prob, min_samples, min_gap, min_score):
    lp = lr.get("weighted_probability", lr["probability"]) if lr["decided"] >= min_samples else 0
    sp = sr.get("weighted_probability", sr["probability"]) if sr["decided"] >= min_samples else 0
    if lr["decided"] < min_samples and sr["decided"] < min_samples:
        return "NO TRADE", "Low sample size"
    if lp >= min_prob and ls >= min_score and (lp-sp) >= min_gap:
        return "LONG", "Long weighted probability and score passed"
    if sp >= min_prob and ss >= min_score and (sp-lp) >= min_gap:
        return "SHORT", "Short weighted probability and score passed"
    return "NO TRADE", "No high-quality edge"

st.title("How Much Possible Which Side")
st.caption("created by Sachin Raosaheb Bhagat")

with st.sidebar:
    st.header("Automatic settings")
    years = 1
    st.caption("CoinDCX history: 1 year")
    refresh = st.selectbox("Auto-refresh seconds", [0, 60, 120, 300], index=1)
    force = st.checkbox("Force rebuild CoinDCX database", False)

    st.header("Backtest reference")
    tp_points = st.number_input("Backtest target points", value=300, min_value=100, max_value=800, step=50)
    sl_points = st.number_input("Backtest stop-loss points", value=400, min_value=50, step=50)
    horizon = st.slider("Max 15m candles to hold", 4, 96, 32)

    st.header("Matching")
    price_tol = st.slider("Similar price zone ±%", 0.25, 8.0, 3.5, 0.25)
    session_match = st.checkbox("India session matching", True)

    st.header("Signal strictness")
    min_samples = st.slider("Minimum decided samples", 20, 300, 50)
    min_prob = st.slider("Minimum probability %", 55, 80, 60)
    min_gap = st.slider("Minimum Long/Short gap %", 5, 35, 8)
    min_score = st.slider("Minimum quality score", 30, 90, 45)

if refresh and not load_cache().empty and not force:
    st_autorefresh(interval=refresh * 1000, key="refresh")

st.subheader("CoinDCX automatic database")
df15_raw = clean_datetime_column(build_or_update_database(years, force))

if df15_raw.empty or len(df15_raw) < 500:
    st.error("Not enough CoinDCX candle data to calculate signal.")
    st.stop()

latest_dt = pd.to_datetime(df15_raw["datetime"].max())
age_min = (pd.Timestamp.now(tz=latest_dt.tz) - latest_dt).total_seconds() / 60
if age_min > 180:
    st.error(f"Data is stale: latest candle is {age_min:.0f} minutes old. No signal shown for safety.")
    st.stop()

df15 = add_indicators(df15_raw)
if df15.empty:
    st.error("Not enough clean candle data after indicator calculation. Datetime rows were cleaned safely, but remaining data is insufficient.")
    st.stop()

try:
    price, price_src = fetch_live_price()
    if np.isnan(price):
        raise ValueError("No live price")
except Exception:
    price = float(df15["close"].iloc[-1])
    price_src = "CoinDCX latest candle close"

dfs = {
    "15m": df15_raw,
    "1h": resample_ohlcv(df15_raw, "1h"),
    "4h": resample_ohlcv(df15_raw, "4h"),
    "1d": resample_ohlcv(df15_raw, "1D"),
}

latest = df15.iloc[-1]
mtf, votes = mtf_bias(dfs)
structure, support, resistance, level_mode = structure_levels(df15, price)
ctx = advanced_market_context(df15, price)

long_r = analyze(df15, price, "LONG", tp_points, sl_points, horizon, price_tol, session_match)
short_r = analyze(df15, price, "SHORT", tp_points, sl_points, horizon, price_tol, session_match)
lscore, lreasons, lgrade, lsample = score_side("LONG", long_r, latest, mtf, structure, support, resistance, price, tp_points, ctx)
sscore, sreasons, sgrade, ssample = score_side("SHORT", short_r, latest, mtf, structure, support, resistance, price, tp_points, ctx)
signal, reason = decide(long_r, short_r, lscore, sscore, min_prob, min_samples, min_gap, min_score)

raw_long_prob = float(long_r.get("weighted_probability", long_r["probability"]))
raw_short_prob = float(short_r.get("weighted_probability", short_r["probability"]))
prob_total = raw_long_prob + raw_short_prob
if prob_total > 0:
    norm_long_prob = raw_long_prob / prob_total * 100
    norm_short_prob = raw_short_prob / prob_total * 100
else:
    norm_long_prob = 0.0
    norm_short_prob = 0.0

edge_gap_display = abs(norm_long_prob - norm_short_prob)
best_side_display = "LONG" if norm_long_prob > norm_short_prob else "SHORT" if norm_short_prob > norm_long_prob else "NEUTRAL"

def round_down_25(x):
    return int(max(0, (int(x) // 25) * 25))

def round_up_25(x):
    return int(max(25, ((int(x) + 24) // 25) * 25))

atr_now = float(latest["atr14"]) if pd.notna(latest["atr14"]) else 150.0
long_room_pts = max(0.0, float(resistance - price))
short_room_pts = max(0.0, float(price - support))

long_possible_pts = round_down_25(min(long_room_pts * 0.80, atr_now * 1.60))
short_possible_pts = round_down_25(min(short_room_pts * 0.80, atr_now * 1.60))

def suggested_sl(tp, atr):
    if tp <= 0:
        return 0
    return round_up_25(min(max(atr * 0.75, tp * 0.55), tp * 0.90))

long_sl_pts = suggested_sl(long_possible_pts, atr_now)
short_sl_pts = suggested_sl(short_possible_pts, atr_now)

direction_signal = "NO TRADE"
possible_tp_pts = 0
possible_sl_pts = 0
direction_reason = "No high-quality mathematical edge"

if best_side_display == "LONG":
    if norm_long_prob >= 55 and edge_gap_display >= 8 and long_possible_pts >= 100 and lscore >= 20:
        direction_signal = "LONG"
        possible_tp_pts = long_possible_pts
        possible_sl_pts = long_sl_pts
        direction_reason = "Long side has better probability and enough room to resistance"
    elif long_possible_pts < 100:
        direction_reason = "Long side has insufficient room before resistance"
elif best_side_display == "SHORT":
    if norm_short_prob >= 55 and edge_gap_display >= 8 and short_possible_pts >= 100 and sscore >= 20:
        direction_signal = "SHORT"
        possible_tp_pts = short_possible_pts
        possible_sl_pts = short_sl_pts
        direction_reason = "Short side has better probability and enough room to support"
    elif short_possible_pts < 100:
        direction_reason = "Short side has insufficient room before support"

if direction_signal == "LONG":
    entry_area = price
    tp_price = price + possible_tp_pts
    sl_price = price - possible_sl_pts
elif direction_signal == "SHORT":
    entry_area = price
    tp_price = price - possible_tp_pts
    sl_price = price + possible_sl_pts
else:
    entry_area = price
    tp_price = np.nan
    sl_price = np.nan

raw_long_prob = float(long_r.get("weighted_probability", long_r["probability"]))
raw_short_prob = float(short_r.get("weighted_probability", short_r["probability"]))
prob_total = raw_long_prob + raw_short_prob

if prob_total > 0:
    norm_long_prob = raw_long_prob / prob_total * 100
    norm_short_prob = raw_short_prob / prob_total * 100
else:
    norm_long_prob = 0.0
    norm_short_prob = 0.0

edge_gap_display = abs(norm_long_prob - norm_short_prob)

def edge_label(gap):
    if gap >= 20:
        return "Strong directional edge"
    if gap >= 12:
        return "Good directional edge"
    if gap >= 8:
        return "Mild directional edge"
    return "No clear edge"

edge_text_display = edge_label(edge_gap_display)
best_side_display = "LONG" if norm_long_prob > norm_short_prob else "SHORT" if norm_short_prob > norm_long_prob else "NEUTRAL"

c1, c2, c3, c4 = st.columns(4)
c1.metric("BTC price", f"{price:,.2f}", price_src)
c2.metric("MTF bias", mtf)
c3.metric("Structure", structure)
c4.metric("Session", latest["india_session"])

c5, c6, c7, c8 = st.columns(4)
c5.metric("RSI", f"{latest['rsi14']:.1f}")
c6.metric("ATR", f"{latest['atr14']:.1f}")
c7.metric("ADX", f"{latest['adx14']:.1f}")
c8.metric("VWAP", f"{latest['vwap']:,.0f}")

c9, c10, c11 = st.columns(3)
c9.metric("Support", f"{support:,.0f}")
c10.metric("Resistance", f"{resistance:,.0f}")
c11.metric("Level mode", level_mode)

room_long = resistance - price
room_short = price - support
f1, f2, f3, f4 = st.columns(4)
f1.metric("Long room to resistance", f"{room_long:,.0f} pts")
f2.metric("Short room to support", f"{room_short:,.0f} pts")
f3.metric("Required TP", f"{tp_points} pts")
f4.metric("SL setting", f"{sl_points} pts")

m1, m2, m3, m4 = st.columns(4)
m1.metric("BOS", ctx.get("bos", "None"))
m2.metric("CHOCH", ctx.get("choch", "None"))
m3.metric("Liquidity sweep", ctx.get("sweep", "None"))
m4.metric("Volume zone", ctx.get("volume_zone", "Neutral"))

st.subheader("How much possible which side")

d1, d2, d3, d4 = st.columns(4)
d1.metric("Trade direction", direction_signal)
d2.metric("Best side", best_side_display)
d3.metric("Long probability", f"{norm_long_prob:.1f}%")
d4.metric("Short probability", f"{norm_short_prob:.1f}%")

p1, p2, p3, p4 = st.columns(4)
p1.metric("Possible TP points", f"{possible_tp_pts} pts" if possible_tp_pts else "NO")
p2.metric("Suggested SL points", f"{possible_sl_pts} pts" if possible_sl_pts else "NO")
p3.metric("Edge gap", f"{edge_gap_display:.1f}%")
p4.metric("Reason", direction_reason)

r1, r2, r3, r4 = st.columns(4)
r1.metric("Long room to resistance", f"{long_room_pts:,.0f} pts")
r2.metric("Short room to support", f"{short_room_pts:,.0f} pts")
r3.metric("ATR 14", f"{atr_now:.1f}")
r4.metric("Match mode", long_r.get("match_mode", "NA") + " / " + short_r.get("match_mode", "NA"))

if direction_signal == "LONG":
    st.success(f"LONG trade: Entry near {entry_area:,.2f} | TP {tp_price:,.2f} (+{possible_tp_pts}) | SL {sl_price:,.2f} (-{possible_sl_pts})")
elif direction_signal == "SHORT":
    st.error(f"SHORT trade: Entry near {entry_area:,.2f} | TP {tp_price:,.2f} (-{possible_tp_pts}) | SL {sl_price:,.2f} (+{possible_sl_pts})")
else:
    st.info("NO TRADE: Direction and possible points are not strong enough mathematically.")

st.subheader("Final signal")
s1, s2, s3, s4 = st.columns(4)
s1.metric("Signal", signal)
s2.metric("Long normalized probability", f"{norm_long_prob:.1f}%")
s3.metric("Short normalized probability", f"{norm_short_prob:.1f}%")
s4.metric("Reason", reason)

r1, r2 = st.columns(2)
r1.metric("Raw long probability", f"{raw_long_prob:.1f}%")
r2.metric("Raw short probability", f"{raw_short_prob:.1f}%")

e1, e2, e3 = st.columns(3)
e1.metric("Normalized edge gap", f"{edge_gap_display:.1f}%")
e2.metric("Edge status", edge_text_display)
e3.metric("Best side", best_side_display)

mm1, mm2 = st.columns(2)
mm1.metric("Long match mode", long_r.get("match_mode", "NA"))
mm2.metric("Short match mode", short_r.get("match_mode", "NA"))

q1, q2, q3, q4 = st.columns(4)
q1.metric("Long quality score", f"{lscore}/100")
q2.metric("Short quality score", f"{sscore}/100")
q3.metric("Long grade", lgrade)
q4.metric("Short grade", sgrade)

g1, g2 = st.columns(2)
g1.metric("Long sample confidence", lsample)
g2.metric("Short sample confidence", ssample)

if signal == "LONG":
    st.success(f"LONG: Entry area {price:,.2f} | TP {price + tp_points:,.2f} | SL {price - sl_points:,.2f}")
elif signal == "SHORT":
    st.error(f"SHORT: Entry area {price:,.2f} | TP {price - tp_points:,.2f} | SL {price + sl_points:,.2f}")
else:
    st.info("NO TRADE: The app did not find enough CoinDCX-based edge for a 300-point trade.")

st.subheader("Evidence")
st.dataframe(pd.DataFrame([long_r, short_r]), use_container_width=True)
st.caption("Weighted probability gives more importance to recent CoinDCX market behavior.")

r1, r2 = st.columns(2)
with r1:
    st.write("Long reasons")
    st.write(lreasons if lreasons else ["No strong long factors"])
with r2:
    st.write("Short reasons")
    st.write(sreasons if sreasons else ["No strong short factors"])

st.write("Multi-timeframe votes:", dict(votes))
st.write(f"CoinDCX data: 15m={len(df15_raw):,}, 1h={len(dfs['1h']):,}, 4h={len(dfs['4h']):,}, 1d={len(dfs['1d']):,}")
st.write(f"Data range: {df15_raw['datetime'].min()} to {df15_raw['datetime'].max()}")

st.subheader("Latest chart")
st.line_chart(df15.tail(300).set_index("datetime")[["close", "ema20", "ema50", "ema200", "vwap"]])

st.warning("Decision-support only. BTC futures are risky. This app rejects weak trades and cannot guarantee profit or accuracy.")