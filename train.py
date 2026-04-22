# ============================================================
# train.py  —  RUN THIS ONCE (or weekly) TO TRAIN ALL TIERS
#
# Imports from stock_pipeline.py (must be in the same folder).
#
# Trains 3 separate model sets:
#   SHORT  — 7-day  horizon, technical indicators
#   MEDIUM — 60-day horizon, trend + momentum
#   LONG   — 365-day horizon, long-term patterns
#
# Each tier trains 4 models and saves to ./models/:
#   future_high  → upper bound of safe range
#   future_low   → lower bound of safe range
#   risk_level   → LOW / MEDIUM / HIGH
#   signal       → BUY / HOLD / SELL
#
# Runtime: ~10–20 minutes for all 3 tiers combined.
# After this, use predict.py for instant predictions.
# ============================================================

import importlib
import stock_pipeline
importlib.reload(stock_pipeline)          # ensures latest code is loaded

from stock_pipeline import (
    get_market_context,
    build_feature_set,
    get_feature_columns,
    train_all_models,
    get_feature_importance,
    save_models,
    TIER_CONFIG,
)
import pandas as pd


# ── Tickers to train on ────────────────────────────────────────
# Diverse across sectors for general pattern learning.
# More stocks = better generalisation.
TRAIN_TICKERS = [
    # Banking & Finance
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS",
    "AXISBANK.NS", "KOTAKBANK.NS", "BAJFINANCE.NS",
    # IT
    "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
    # Energy & Industrials
    "RELIANCE.NS", "ONGC.NS", "BPCL.NS",
    "LT.NS", "NTPC.NS", "POWERGRID.NS",
    # FMCG & Consumption
    "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS",
    "ASIANPAINT.NS", "TITAN.NS",
    # Auto & Pharma
    "MARUTI.NS", "M&M.NS",
    "SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS",
    # Metals (cyclical — helps train SELL signals)
    "TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS",
]

# ── Which tiers to train ───────────────────────────────────────
# ["short", "medium", "long"] trains all three.
# You can run just one, e.g. ["long"] to retrain only the long tier.
TRAIN_TIERS = ["short", "medium", "long"]


# ── Step 1: Download market context once ──────────────────────
print("=" * 60)
print("STEP 1 — Downloading market context (Nifty, VIX, INR, Crude)")
print("=" * 60)
context = get_market_context(period="15y")


# ── Step 2: Train each tier ────────────────────────────────────
for tier in TRAIN_TIERS:
    cfg = TIER_CONFIG[tier]
    print(f"\n{'='*60}")
    print(f"TIER : {cfg['label']}")
    print(f"{'='*60}")

    print(f"\n📦 Building [{tier}] training dataset...")
    all_data = []
    for ticker in TRAIN_TICKERS:
        try:
            df = build_feature_set(ticker, context, tier=tier)
            all_data.append(df)
        except Exception as e:
            print(f"   ⚠️  Skipping {ticker}: {e}")

    if not all_data:
        print(f"❌ No data collected for [{tier}] — skipping.")
        continue

    combined = pd.concat(all_data).dropna()
    print(f"\n✅ [{tier}] dataset: {len(combined):,} rows "
          f"across {len(all_data)} stocks")

    # Train all 4 models for this tier
    models = train_all_models(combined, tier=tier)

    # Show top features
    feature_cols = get_feature_columns(combined)
    get_feature_importance(models, feature_cols, top_n=10)

    # Save to ./models/
    save_models(tier, models)


print("\n" + "=" * 60)
print("🎉 Training complete!")
print("   Run predict.py to get instant predictions.")
print("=" * 60)