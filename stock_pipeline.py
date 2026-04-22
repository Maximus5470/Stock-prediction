# ============================================================
# stock_pipeline.py  —  Complete rewrite
# Indian Stock Market ML Pipeline
# Tiers: SHORT (1-30d) | MEDIUM (31-365d) | LONG (366d+)
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, classification_report
)
import joblib
import os


# ============================================================
# TIER CONFIGURATION
# ============================================================

def get_tier(days: int) -> str:
    if days <= 30:  return "short"
    if days <= 365: return "medium"
    return "long"

TIER_CONFIG = {
    "short": {
        "label":        "Short-term  (1–30 days)",
        "train_period": "5y",
        "data_period":  "2y",
        "horizon":      7,
        "buy_thresh":   0.02,
        "sell_thresh":  -0.02,
    },
    "medium": {
        "label":        "Medium-term (1–12 months)",
        "train_period": "10y",
        "data_period":  "3y",
        "horizon":      60,
        "buy_thresh":   0.05,
        "sell_thresh":  -0.05,
    },
    "long": {
        "label":        "Long-term   (1–5 years)",
        "train_period": "15y",
        "data_period":  "5y",
        "horizon":      365,
        # ±10% (not ±15%) so BUY/SELL classes have enough samples
        "buy_thresh":   0.10,
        "sell_thresh":  -0.10,
    },
}


# ============================================================
# SECTION 1: DATA COLLECTION
# ============================================================

def _fix_df(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns and strip timezone from index."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def get_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download OHLCV data. Always fetches up to today."""
    print(f"  📥 {ticker}...")
    days_map = {
        "1y": 365, "2y": 730,  "3y": 1095,
        "5y": 1825,"10y":3650, "15y":5475
    }
    end   = datetime.now()
    start = end - timedelta(days=days_map.get(period, 1825))
    tkr   = yf.Ticker(ticker)
    df    = tkr.history(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d", auto_adjust=True,
    )
    df = _fix_df(df)
    df = df[["Open","High","Low","Close","Volume"]].copy()
    df.dropna(inplace=True)
    print(f"     ✅ {len(df)} rows | Latest: {df.index[-1].date()} | ₹{df['Close'].iloc[-1]:.2f}")
    return df


def get_latest_price(ticker: str) -> tuple:
    """Fetch freshest available price. Returns (price, date)."""
    tkr  = yf.Ticker(ticker)
    hist = tkr.history(period="5d", interval="1d", auto_adjust=True)
    hist = _fix_df(hist)
    if hist.empty:
        raise ValueError(f"Could not fetch latest price for {ticker}")
    return float(hist["Close"].iloc[-1]), hist.index[-1].date()


def get_market_context(period: str = "5y") -> pd.DataFrame:
    """Pull India macro/market context data."""
    print("📥 Downloading market context...")
    symbols = {
        "nifty":    "^NSEI",
        "sensex":   "^BSESN",
        "indiavix": "^INDIAVIX",
        "usdinr":   "USDINR=X",
        "crude":    "CL=F",
    }
    ctx = pd.DataFrame()
    for name, sym in symbols.items():
        try:
            raw = yf.download(sym, period=period, interval="1d",
                              auto_adjust=True, progress=False)
            raw = _fix_df(raw)
            s   = raw["Close"].squeeze()
            if hasattr(s.index, "tz") and s.index.tz is not None:
                s.index = s.index.tz_localize(None)
            s.name = name
            ctx = pd.concat([ctx, s], axis=1)
            print(f"   ✅ {name}")
        except Exception as e:
            print(f"   ⚠️  {name}: {e}")
    ctx.dropna(how="all", inplace=True)
    return ctx


# ============================================================
# SECTION 2: FEATURE ENGINEERING
# ============================================================

def add_price_features(df):
    c = df["Close"].squeeze()
    o = df["Open"].squeeze()
    h = df["High"].squeeze()
    l = df["Low"].squeeze()
    df["daily_return"] = c.pct_change()
    df["log_return"]   = np.log(c / c.shift(1))
    df["price_range"]  = h - l
    df["gap"]          = o - c.shift(1)
    df["candle_body"]  = c - o
    df["upper_wick"]   = h - c.clip(lower=o)
    df["lower_wick"]   = o.clip(upper=c) - l
    return df


def add_moving_averages(df):
    c = df["Close"].squeeze()
    for p in [10, 20, 50, 200]:
        df[f"sma_{p}"]         = c.rolling(p).mean()
        df[f"price_vs_sma{p}"] = c / df[f"sma_{p}"]
    for p in [9, 21, 55]:
        df[f"ema_{p}"] = c.ewm(span=p, adjust=False).mean()
    df["sma20_gt_sma50"]  = (df["sma_20"] > df["sma_50"]).astype(int)
    df["sma50_gt_sma200"] = (df["sma_50"] > df["sma_200"]).astype(int)
    return df


def add_momentum(df):
    c = df["Close"].squeeze()
    h = df["High"].squeeze()
    l = df["Low"].squeeze()
    delta       = c.diff()
    gain        = delta.clip(lower=0).rolling(14).mean()
    loss        = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"]   = 100 - 100 / (1 + gain / (loss + 1e-9))
    e12         = c.ewm(span=12, adjust=False).mean()
    e26         = c.ewm(span=26, adjust=False).mean()
    df["macd"]      = e12 - e26
    df["macd_sig"]  = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]
    df["macd_cross"]= (df["macd"] > df["macd_sig"]).astype(int)
    df["roc_10"]    = c.pct_change(10) * 100
    lo14 = l.rolling(14).min()
    hi14 = h.rolling(14).max()
    df["stoch_k"]   = 100 * (c - lo14) / (hi14 - lo14 + 1e-9)
    df["stoch_d"]   = df["stoch_k"].rolling(3).mean()
    df["williams_r"]= -100 * (hi14 - c) / (hi14 - lo14 + 1e-9)
    return df


def add_volatility(df):
    c = df["Close"].squeeze()
    h = df["High"].squeeze()
    l = df["Low"].squeeze()
    s20         = c.rolling(20).mean()
    std20       = c.rolling(20).std()
    df["bb_up"] = s20 + 2 * std20
    df["bb_dn"] = s20 - 2 * std20
    df["bb_wid"]= (df["bb_up"] - df["bb_dn"]) / s20
    df["bb_pos"]= (c - df["bb_dn"]) / (df["bb_up"] - df["bb_dn"] + 1e-9)
    pc          = c.shift(1)
    tr          = pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["atr_pct"]= df["atr14"] / c
    df["vol10"] = c.pct_change().rolling(10).std()
    df["vol20"] = c.pct_change().rolling(20).std()
    df["hl_pct"]= (h - l) / c
    return df


def add_volume_features(df):
    c = df["Close"].squeeze()
    v = df["Volume"].squeeze()
    df["vol_ma20"]  = v.rolling(20).mean()
    df["vol_ratio"] = v / (df["vol_ma20"] + 1e-9)
    df["vol_spike"] = (df["vol_ratio"] > 2.0).astype(int)
    df["obv"]       = (np.sign(c.diff()) * v).cumsum()
    df["obv_ma10"]  = df["obv"].rolling(10).mean()
    df["obv_trend"] = (df["obv"] > df["obv_ma10"]).astype(int)
    df["vwap"]      = (c*v).rolling(20).sum() / (v.rolling(20).sum() + 1e-9)
    df["vs_vwap"]   = c / df["vwap"]
    return df


def add_context_features(df, ctx):
    r = ctx.pct_change()
    r.columns = [f"{c}_ret" for c in r.columns]
    df = df.join(r, how="left")
    if "indiavix" in ctx.columns:
        df = df.join(ctx[["indiavix"]].rename(columns={"indiavix":"vix"}),how="left")
    if "usdinr" in ctx.columns:
        df = df.join(ctx[["usdinr"]], how="left")
    return df


def add_calendar(df):
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df["dow"]      = df.index.dayofweek
    df["month"]    = df.index.month
    df["quarter"]  = df.index.quarter
    df["mo_end"]   = df.index.is_month_end.astype(int)
    df["mo_start"] = df.index.is_month_start.astype(int)
    df["qtr_end"]  = df.index.is_quarter_end.astype(int)
    df["is_thu"]   = (df.index.dayofweek == 3).astype(int)
    df["is_feb"]   = (df.index.month == 2).astype(int)
    return df


def add_lags(df, tier):
    lags = {
        "short":  ([1,2,3,5,10], [1,2,3]),
        "medium": ([5,10,20,40,60], [5,10,20]),
        "long":   ([20,60,120,250], [20,60]),
    }
    rl, il = lags[tier]
    for lag in rl:
        df[f"ret_lag{lag}"] = df["daily_return"].shift(lag)
        df[f"vr_lag{lag}"]  = df["vol_ratio"].shift(lag)
    for lag in il:
        df[f"rsi_lag{lag}"] = df["rsi"].shift(lag)
        df[f"mh_lag{lag}"]  = df["macd_hist"].shift(lag)
        df[f"atr_lag{lag}"] = df["atr_pct"].shift(lag)
    return df


def add_medium_features(df):
    c = df["Close"].squeeze()
    for p in [100, 150]:
        df[f"sma_{p}"]         = c.rolling(p).mean()
        df[f"price_vs_sma{p}"] = c / df[f"sma_{p}"]
    df["ret_1m"]     = c.pct_change(20)
    df["ret_3m"]     = c.pct_change(60)
    df["ret_6m"]     = c.pct_change(120)
    df["vol_regime"] = (df["vol20"] > df["vol20"].rolling(60).mean()).astype(int)
    h = df["High"].squeeze()
    l = df["Low"].squeeze()
    dmu = h.diff().clip(lower=0)
    dmd = (-l.diff()).clip(lower=0)
    df["trend_str"] = (dmu.rolling(14).mean() - dmd.rolling(14).mean()).abs()
    return df


def add_long_features(df):
    c = df["Close"].squeeze()
    for p in [250, 500]:
        df[f"sma_{p}"]         = c.rolling(p).mean()
        df[f"price_vs_sma{p}"] = c / df[f"sma_{p}"]
    df["ret_6m"]       = c.pct_change(120)
    df["ret_1y"]       = c.pct_change(250)
    df["ret_2y"]       = c.pct_change(500)
    df["vol_1y"]       = c.pct_change().rolling(250).std()
    df["vol_rat_long"] = df["vol20"] / (df["vol_1y"] + 1e-9)
    df["pos_52w_hi"]   = c / c.rolling(250).max()
    df["pos_52w_lo"]   = c / c.rolling(250).min()
    return df


def add_targets(df, tier):
    cfg = TIER_CONFIG[tier]
    h   = cfg["horizon"]
    c   = df["Close"].squeeze()
    hi  = df["High"].squeeze()
    lo  = df["Low"].squeeze()

    df["future_high"]   = hi.shift(-1).rolling(h).max().shift(-(h-1))
    df["future_low"]    = lo.shift(-1).rolling(h).min().shift(-(h-1))
    df["future_return"] = c.shift(-h) / c - 1

    fv = df["daily_return"].shift(-1).rolling(h).std().shift(-(h-1))
    df["risk_level"] = pd.cut(
        fv, bins=[0, 0.010, 0.025, 1.0],
        labels=["LOW","MEDIUM","HIGH"]
    )

    df["signal"] = "HOLD"
    df.loc[df["future_return"] >  cfg["buy_thresh"],  "signal"] = "BUY"
    df.loc[df["future_return"] <  cfg["sell_thresh"], "signal"] = "SELL"
    return df


def build_feature_set(ticker, ctx, tier):
    cfg = TIER_CONFIG[tier]
    df  = get_stock_data(ticker, period=cfg["train_period"])
    df  = add_price_features(df)
    df  = add_moving_averages(df)
    df  = add_momentum(df)
    df  = add_volatility(df)
    df  = add_volume_features(df)
    df  = add_context_features(df, ctx)
    df  = add_calendar(df)
    df  = add_lags(df, tier)
    if tier in ("medium","long"):
        df = add_medium_features(df)
    if tier == "long":
        df = add_long_features(df)
    df  = add_targets(df, tier)
    df.dropna(inplace=True)
    return df


EXCLUDE = [
    "Open","High","Low","Close","Volume",
    "future_high","future_low","future_return",
    "risk_level","signal"
]

def get_feature_columns(df):
    return [c for c in df.columns if c not in EXCLUDE]


# ============================================================
# SECTION 3: MODEL TRAINING
# ============================================================

def _train_regressor(X, y, label):
    tscv = TimeSeriesSplit(n_splits=5)
    maes, mdls = [], []
    for fold, (tr, val) in enumerate(tscv.split(X)):
        m = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbosity=0
        )
        m.fit(X.iloc[tr], y.iloc[tr],
              eval_set=[(X.iloc[val], y.iloc[val])], verbose=False)
        mae = mean_absolute_error(y.iloc[val], m.predict(X.iloc[val]))
        maes.append(mae); mdls.append(m)
        print(f"   Fold {fold+1} | {label} | MAE: ₹{mae:.2f}")
    print(f"   ✅ Avg MAE: ₹{np.mean(maes):.2f}")
    return {"model": mdls[-1], "avg_mae": float(np.mean(maes))}


def _safe_classification_report(y_true, y_pred, le):
    """
    Print classification_report using ONLY the classes that
    actually appear in y_true or y_pred for this fold.

    ROOT CAUSE of the ValueError:
      le.classes_ always has ALL classes (e.g. LOW/MEDIUM/HIGH).
      But in the LONG tier, rare classes like HIGH volatility or SELL
      may have zero samples in the most-recent validation fold
      (recent market data tends to be calm / bullish).
      Passing le.classes_ when only 2 classes are present causes:
        "Number of classes, 2, does not match size of target_names, 3"

    FIX: compute present_ints from actual data, not from le.classes_.
    """
    # Which integer-encoded labels actually appear in this fold?
    present_ints  = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    # Map integer labels → original string class names
    present_names = [le.classes_[i] for i in present_ints]

    # Warn if any class is completely absent
    missing = [c for c in le.classes_ if c not in present_names]
    if missing:
        print(f"   ℹ️  Note: {missing} absent from last val fold "
              f"(normal for rare classes in LONG tier)")

    print(classification_report(
        y_true,
        y_pred,
        labels      = present_ints,    # ← ONLY labels present in this fold
        target_names= present_names,   # ← exactly matching names
        zero_division=0
    ))


def _train_classifier(X, y_raw, label):
    """
    Train XGBoost classifier with walk-forward validation.
    Uses _safe_classification_report to handle rare/missing classes.
    """
    le = LabelEncoder()
    # Fit on full data so integer encoding is consistent across folds
    y  = le.fit_transform(y_raw.astype(str))
    n  = len(le.classes_)

    # Show distribution so you can spot dangerous imbalance
    dist = {le.classes_[i]: int((y == i).sum()) for i in range(n)}
    print(f"   Classes [{label}]: {dist}")
    for cls, cnt in dist.items():
        pct = cnt / len(y) * 100
        if pct < 5:
            print(f"   ⚠️  '{cls}' only {cnt} samples ({pct:.1f}%) "
                  f"— may vanish from some validation folds")

    tscv = TimeSeriesSplit(n_splits=5)
    scores, mdls = [], []

    for fold, (tr, val) in enumerate(tscv.split(X)):
        y_tr  = y[tr]
        y_val = y[val]

        # XGBoost needs at least 2 classes in the training split
        if len(np.unique(y_tr)) < 2:
            print(f"   Fold {fold+1} | {label} | ⚠️  Skipped (only 1 class in train)")
            continue

        m = xgb.XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, n_jobs=-1, verbosity=0
        )
        m.fit(X.iloc[tr], y_tr,
              eval_set=[(X.iloc[val], y_val)], verbose=False)

        preds = m.predict(X.iloc[val])
        acc   = accuracy_score(y_val, preds)
        scores.append(acc); mdls.append(m)
        print(f"   Fold {fold+1} | {label} | Acc: {acc*100:.1f}%")

    if not mdls:
        raise ValueError(
            f"All folds skipped for [{label}] — only 1 class in entire dataset.\n"
            f"Distribution: {dist}\n"
            f"Fix: lower buy_thresh/sell_thresh in TIER_CONFIG for the long tier."
        )

    best    = mdls[-1]
    avg_acc = float(np.mean(scores))
    print(f"   ✅ Avg Accuracy: {avg_acc*100:.1f}%")

    # Print report using only classes present in the last validation fold
    val_idx = list(tscv.split(X))[-1][1]
    _safe_classification_report(y[val_idx], best.predict(X.iloc[val_idx]), le)

    return {"model": best, "encoder": le, "avg_accuracy": avg_acc}


def train_all_models(df, tier):
    """Train all 4 models for a given tier."""
    fc = get_feature_columns(df)
    X  = df[fc]

    print(f"\n🏋️  [{tier.upper()}] Training future HIGH regressor...")
    high_m = _train_regressor(X, df["future_high"], "future_high")

    print(f"\n🏋️  [{tier.upper()}] Training future LOW regressor...")
    low_m  = _train_regressor(X, df["future_low"],  "future_low")

    print(f"\n🏋️  [{tier.upper()}] Training RISK LEVEL classifier...")
    risk_m = _train_classifier(X, df["risk_level"], "risk_level")

    print(f"\n🏋️  [{tier.upper()}] Training BUY/HOLD/SELL classifier...")
    sig_m  = _train_classifier(X, df["signal"], "signal")

    return {
        "future_high": high_m,
        "future_low":  low_m,
        "risk":        risk_m,
        "signal":      sig_m,
    }


def get_feature_importance(models, feature_cols, top_n=15):
    imp = pd.Series(
        models["future_high"]["model"].feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False).head(top_n)
    print(f"\n🔍 Top {top_n} Features:")
    for feat, score in imp.items():
        bar = "█" * int(score * 200)
        print(f"   {feat:<32} {bar} {score:.4f}")
    return imp


# ============================================================
# SECTION 4: SAVE & LOAD MODELS
# ============================================================

def save_models(tier, models, path="./models"):
    os.makedirs(path, exist_ok=True)
    p = f"{path}/{tier}"
    joblib.dump(models["future_high"]["model"],   f"{p}_high_model.pkl")
    joblib.dump(models["future_high"]["avg_mae"], f"{p}_high_mae.pkl")
    joblib.dump(models["future_low"]["model"],    f"{p}_low_model.pkl")
    joblib.dump(models["future_low"]["avg_mae"],  f"{p}_low_mae.pkl")
    joblib.dump(models["risk"]["model"],          f"{p}_risk_model.pkl")
    joblib.dump(models["risk"]["encoder"],        f"{p}_risk_enc.pkl")
    joblib.dump(models["signal"]["model"],        f"{p}_signal_model.pkl")
    joblib.dump(models["signal"]["encoder"],      f"{p}_signal_enc.pkl")
    print(f"   💾 [{tier}] models saved → {path}/")


def load_models(tier, path="./models"):
    p = f"{path}/{tier}"
    if not os.path.exists(f"{p}_high_model.pkl"):
        raise FileNotFoundError(
            f"No models found for tier '{tier}'. Run train.py first."
        )
    return {
        "future_high": {
            "model":   joblib.load(f"{p}_high_model.pkl"),
            "avg_mae": joblib.load(f"{p}_high_mae.pkl"),
        },
        "future_low": {
            "model":   joblib.load(f"{p}_low_model.pkl"),
            "avg_mae": joblib.load(f"{p}_low_mae.pkl"),
        },
        "risk": {
            "model":   joblib.load(f"{p}_risk_model.pkl"),
            "encoder": joblib.load(f"{p}_risk_enc.pkl"),
        },
        "signal": {
            "model":   joblib.load(f"{p}_signal_model.pkl"),
            "encoder": joblib.load(f"{p}_signal_enc.pkl"),
        },
    }


# ============================================================
# SECTION 5: PREDICTION
# ============================================================

def _dur_label(days):
    if days < 7:   return f"{days} day(s)"
    if days < 30:  return f"{days} days (~{days//7} week{'s' if days//7>1 else ''})"
    if days < 365: return f"{days} days (~{round(days/30)} months)"
    return f"{days} days (~{round(days/365,1)} year{'s' if days>=730 else ''})"


def predict(ticker, days, ctx, model_path="./models"):
    """
    Predict safe price range + risk + BUY/HOLD/SELL for any ticker and duration.
    Automatically routes to the correct tier. Loads saved models (no retraining).
    """
    tier   = get_tier(days)
    cfg    = TIER_CONFIG[tier]
    models = load_models(tier, path=model_path)

    print(f"\n🔮 Predicting {ticker} | {_dur_label(days)} | Tier: {tier.upper()}")

    # Build features using shorter data window (faster)
    df = get_stock_data(ticker, period=cfg["data_period"])
    df = add_price_features(df)
    df = add_moving_averages(df)
    df = add_momentum(df)
    df = add_volatility(df)
    df = add_volume_features(df)
    df = add_context_features(df, ctx)
    df = add_calendar(df)
    df = add_lags(df, tier)
    if tier in ("medium","long"):
        df = add_medium_features(df)
    if tier == "long":
        df = add_long_features(df)
    df.dropna(inplace=True)

    # Align to trained feature names
    trained = list(models["future_high"]["model"].feature_names_in_)
    for col in trained:
        if col not in df.columns:
            df[col] = 0.0
    X = df[trained].iloc[[-1]]

    # Freshest price
    curr_price, curr_date = get_latest_price(ticker)

    # Predictions
    pred_high = float(models["future_high"]["model"].predict(X)[0])
    pred_low  = float(models["future_low"]["model"].predict(X)[0])
    mae_h     = models["future_high"]["avg_mae"]
    mae_l     = models["future_low"]["avg_mae"]

    risk_enc  = models["risk"]["model"].predict(X)[0]
    risk_lbl  = models["risk"]["encoder"].inverse_transform([risk_enc])[0]
    risk_conf = max(models["risk"]["model"].predict_proba(X)[0]) * 100

    sig_enc   = models["signal"]["model"].predict(X)[0]
    sig_lbl   = models["signal"]["encoder"].inverse_transform([sig_enc])[0]
    sig_conf  = max(models["signal"]["model"].predict_proba(X)[0]) * 100

    se  = {"BUY":"🟢","HOLD":"🟡","SELL":"🔴"}.get(sig_lbl,  "⚪")
    re  = {"LOW":"🟢","MEDIUM":"🟡","HIGH":"🔴"}.get(risk_lbl,"⚪")

    rsi       = float(df["rsi"].iloc[-1])
    macd_hist = float(df["macd_hist"].iloc[-1])
    bb_pos    = float(df["bb_pos"].iloc[-1])
    atr       = float(df["atr14"].iloc[-1])
    vol_ratio = float(df["vol_ratio"].iloc[-1])

    print(f"""
╔══════════════════════════════════════════════════════════╗
║              📊 STOCK ANALYSIS REPORT                    ║
╠══════════════════════════════════════════════════════════╣
║  Ticker        : {ticker:<39}║
║  Date          : {str(curr_date):<39}║
║  Current Price : ₹{curr_price:<38.2f}║
║  Horizon       : {_dur_label(days):<39}║
║  Tier          : {cfg['label']:<39}║
╠══════════════════════════════════════════════════════════╣
║  🎯 Predicted Safe Price Range                           ║
║                                                          ║
║  Upper Bound   : ₹{pred_high:<38.2f}║
║  Lower Bound   : ₹{pred_low:<38.2f}║
║  Est. Error    : ±₹{mae_h:.2f} / ±₹{mae_l:.2f}                  ║
╠══════════════════════════════════════════════════════════╣
║  {se}  Signal      : {sig_lbl:<27}  ({sig_conf:.0f}% conf) ║
║  {re}  Risk Level  : {risk_lbl:<27}  ({risk_conf:.0f}% conf) ║
╠══════════════════════════════════════════════════════════╣
║  🔍 Key Indicators                                       ║
║  RSI           : {rsi:<39.1f}║
║  MACD Hist     : {macd_hist:<39.4f}║
║  Bollinger Pos : {bb_pos:<39.2f}║
║  ATR (14)      : ₹{atr:<38.2f}║
║  Volume Ratio  : {vol_ratio:<39.2f}║
╚══════════════════════════════════════════════════════════╝""")

    notes = []
    if rsi < 35:         notes.append("⚠️  RSI oversold — potential bounce zone")
    elif rsi > 65:       notes.append("⚠️  RSI overbought — potential pullback")
    if macd_hist > 0:    notes.append("✅ MACD bullish momentum")
    else:                notes.append("🔻 MACD bearish momentum")
    if bb_pos < 0.2:     notes.append("✅ Near Bollinger lower band — potential support")
    elif bb_pos > 0.8:   notes.append("⚠️  Near Bollinger upper band — potential resistance")
    if vol_ratio > 1.5:  notes.append("📢 Above-average volume — strong move")
    if risk_lbl=="HIGH": notes.append("🚨 HIGH RISK — consider smaller position size")

    if notes:
        print("\n📌 Signal Notes:")
        for n in notes: print(f"   {n}")

    return {
        "ticker": ticker, "days": days, "tier": tier,
        "current_price": curr_price,
        "pred_high": pred_high, "pred_low": pred_low,
        "signal": sig_lbl,   "signal_conf": sig_conf,
        "risk":   risk_lbl,  "risk_conf":   risk_conf,
    }