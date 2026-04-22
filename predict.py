# ============================================================
# predict.py  —  RUN THIS ANYTIME FOR INSTANT PREDICTIONS
#
# Prerequisites: run train.py at least once first.
# No retraining — loads saved models and predicts in ~5 seconds.
#
# The user can enter ANY duration:
#   5       → 5 days   → SHORT tier
#   30      → 1 month  → SHORT tier
#   90      → 3 months → MEDIUM tier
#   180     → 6 months → MEDIUM tier
#   365     → 1 year   → MEDIUM tier
#   730     → 2 years  → LONG tier
#   1825    → 5 years  → LONG tier
# ============================================================

from stock_pipeline import get_market_context, predict, get_tier, TIER_CONFIG


def get_user_inputs() -> tuple:
    """Interactive prompt to get ticker and duration from user."""

    print("\n" + "=" * 55)
    print("  📈 Indian Stock Market Predictor")
    print("=" * 55)

    # ── Ticker ──────────────────────────────────────────────
    print("\nEnter NSE ticker (e.g. RELIANCE, TCS, HDFCBANK)")
    print("The '.NS' suffix will be added automatically.")
    raw = input("  Ticker: ").strip().upper()
    ticker = raw if raw.endswith(".NS") or raw.endswith(".BO") else raw + ".NS"

    # ── Duration ─────────────────────────────────────────────
    print("\nHow far ahead do you want to predict?")
    print("  Examples:")
    print("    7        →  1 week        (SHORT tier)")
    print("    30       →  1 month       (SHORT tier)")
    print("    90       →  3 months      (MEDIUM tier)")
    print("    180      →  6 months      (MEDIUM tier)")
    print("    365      →  1 year        (MEDIUM tier)")
    print("    730      →  2 years       (LONG tier)")
    print("    1825     →  5 years       (LONG tier)")

    while True:
        try:
            days = int(input("\n  Number of days: ").strip())
            if days < 1:
                print("  ⚠️  Please enter a positive number.")
                continue
            if days > 1825:
                print("  ⚠️  Maximum supported horizon is 1825 days (5 years).")
                continue
            break
        except ValueError:
            print("  ⚠️  Please enter a whole number.")

    # Show which tier will be used
    tier = get_tier(days)
    cfg  = TIER_CONFIG[tier]
    print(f"\n  ✅ Using: {cfg['label']}")

    return ticker, days


def run_batch(predictions: list, context):
    """Run multiple predictions without re-asking for context."""
    for ticker, days in predictions:
        try:
            predict(ticker, days, context)
        except FileNotFoundError as e:
            print(f"\n❌ {e}")
        except Exception as e:
            print(f"\n⚠️  Could not predict {ticker}: {e}")


# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":

    # Fetch market context once (shared across all predictions)
    print("\n📥 Fetching market context data...")
    context = get_market_context(period="2y")

    while True:
        try:
            ticker, days = get_user_inputs()
            predict(ticker, days, context)
        except FileNotFoundError as e:
            print(f"\n❌ {e}")
        except Exception as e:
            print(f"\n⚠️  Error: {e}")

        print("\n" + "-" * 55)
        again = input("  Predict another stock? (y/n): ").strip().lower()
        if again != "y":
            print("\n👋 Goodbye!\n")
            break
