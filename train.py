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


TRAIN_TICKERS = [
 
    # ── LARGE CAP: Banking & Finance (keep existing + add) ──────
    "HDFCBANK.NS",      # Largest private bank — stable, low vol
    "ICICIBANK.NS",     # Strong growth bank
    "SBIN.NS",          # PSU bank — different behaviour to private
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "BAJFINANCE.NS",    # NBFC — high beta, volatile
    "INDUSINDBK.NS",    # Mid-large, more volatile than HDFC/ICICI
    "BANDHANBNK.NS",    # Micro-finance bank — high volatility ← NEW
    "PNB.NS",           # PSU bank — weak fundamentals, high swings ← NEW
    "BANKBARODA.NS",    # Another PSU with different cycle ← NEW
 
    # ── LARGE CAP: IT ───────────────────────────────────────────
    "TCS.NS",
    "INFY.NS",
    "WIPRO.NS",
    "HCLTECH.NS",
    "TECHM.NS",
    "LTIM.NS",          # LTIMindtree — strong mid-large IT ← NEW
    "MPHASIS.NS",       # Mid-cap IT, more volatile ← NEW
    "PERSISTENT.NS",    # High-growth mid IT ← NEW
    "COFORGE.NS",       # Aggressive growth, volatile ← NEW
 
    # ── LARGE CAP: Energy & Industrials ─────────────────────────
    "RELIANCE.NS",
    "ONGC.NS",
    "BPCL.NS",
    "LT.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "ADANIPOWER.NS",    # Adani — very high volatility ← NEW
    "ADANIENT.NS",      # Adani flagship — extreme swings ← NEW
    "TATAPOWER.NS",     # Power sector diversity ← NEW
    "SUZLON.NS",        # Renewable energy — penny-to-multi-bagger ← NEW
 
    # ── LARGE CAP: FMCG & Consumption ───────────────────────────
    "HINDUNILVR.NS",
    "ITC.NS",
    "NESTLEIND.NS",
    "ASIANPAINT.NS",
    "TITAN.NS",
    "DMART.NS",         # Retail — steady compounder ← NEW
    "TRENT.NS",         # Tata retail — high growth mid ← NEW
    "ETERNAL.NS",        # New-age, highly volatile ← NEW
    "NYKAA.NS",         # New listing, volatile ← NEW
 
    # ── LARGE CAP: Auto ─────────────────────────────────────────
    "MARUTI.NS",
    "M&M.NS",
    "TMCV.NS",    # High volatility, JLR exposure ← NEW
    "BAJAJ-AUTO.NS",    # Premium auto — different cycle ← NEW
    "HEROMOTOCO.NS",    # Two-wheeler — defensive auto ← NEW
    "EICHERMOT.NS",     # Royal Enfield — premium niche ← NEW
 
    # ── LARGE CAP: Pharma & Healthcare ──────────────────────────
    "SUNPHARMA.NS",
    "CIPLA.NS",
    "DRREDDY.NS",
    "DIVISLAB.NS",      # API manufacturer — export cycle ← NEW
    "APOLLOHOSP.NS",    # Healthcare services — defensive ← NEW
    "MAXHEALTH.NS",     # Hospital chain ← NEW
 
    # ── LARGE CAP: Metals & Mining (cyclical — SELL signals) ────
    "TATASTEEL.NS",
    "HINDALCO.NS",
    "JSWSTEEL.NS",
    "COALINDIA.NS",     # PSU commodity — dividend + cycle ← NEW
    "VEDL.NS",          # Vedanta — high vol, commodity ← NEW
    "NATIONALUM.NS",    # Aluminium — deep cyclical ← NEW
 
    # ── MID CAP: High Volatility (critical for SELL signals) ────
    "YESBANK.NS",       # Near-collapse then recovery — extreme ← NEW
    "IDEA.NS",          # Telecom distress stock — drawdowns ← NEW
    "IRCTC.NS",         # Govt monopoly + travel + volatile ← NEW
    "INDHOTEL.NS",      # Hotels — COVID crash & recovery ← NEW
    "JUBLFOOD.NS",      # QSR — post-COVID volatile recovery ← NEW
    "PAGEIND.NS",       # Premium apparel — niche compounder ← NEW
 
    # ── MID CAP: Finance & Insurance ────────────────────────────
    "IDFCFIRSTB.NS",    # Turnaround story — volatile ← NEW
    "FEDERALBNK.NS",    # South Indian bank — different behaviour ← NEW
    "LICI.NS",          # LIC — huge PSU insurer ← NEW
    "HDFCLIFE.NS",      # Private insurer — steady ← NEW
    "SBILIFE.NS",       # PSU insurer — different risk profile ← NEW
 
    # ── MID CAP: Infrastructure & Real Estate ───────────────────
    "ADANIPORTS.NS",    # Port + logistics
    "DLF.NS",           # Real estate — highly cyclical ← NEW
    "LODHA.NS",         # Real estate developer ← NEW
    "IRB.NS",           # Road infra — steady toll income ← NEW
 
    # ── SECTOR INDICES (macro pattern learning) ─────────────────
    "^NSEBANK",         # Bank Nifty — sector-wide banking moves ← NEW
    "^CNXIT",           # Nifty IT index ← NEW
    "^CNXPHARMA",       # Pharma index ← NEW
    "^CNXFMCG",         # FMCG index ← NEW
]

TRAIN_TIERS = ["short", "medium", "long"]

print("=" * 60)
print("STEP 1 — Downloading market context (Nifty, VIX, INR, Crude)")
print("=" * 60)
context = get_market_context(period="15y")

for tier in TRAIN_TIERS:
    cfg = TIER_CONFIG[tier]
    print(f"\n{'='*60}")
    print(f"TIER : {cfg['label']}")
    print(f"{'='*60}")

    print(f"\n[BUILD] Building [{tier}] training dataset...")
    all_data = []
    for ticker in TRAIN_TICKERS:
        try:
            df = build_feature_set(ticker, context, tier=tier)
            all_data.append(df)
        except Exception as e:
            print(f"   [SKIP] Skipping {ticker}: {e}")

    if not all_data:
        print(f"[ERROR] No data collected for [{tier}] - skipping.")
        continue

    combined = pd.concat(all_data).dropna()
    print(f"\n[OK] [{tier}] dataset: {len(combined):,} rows "
          f"across {len(all_data)} stocks")

    models = train_all_models(combined, tier=tier)
    feature_cols = get_feature_columns(combined)
    get_feature_importance(models, feature_cols, top_n=10)
    save_models(tier, models)


print("\n" + "=" * 60)
print("[SUCCESS] Training complete!")
print("   Run predict.py to get instant predictions.")
print("=" * 60)