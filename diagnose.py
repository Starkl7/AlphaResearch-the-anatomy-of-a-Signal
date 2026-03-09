"""
diagnose_backtest.py
--------------------
Run this BEFORE fixing anything. It will tell you exactly why trades aren't firing.

Usage:
    python diagnose_backtest.py

Edit PAIR1, PAIR2, INTERVAL, FILE_PATH below to match a pair that showed 0 trades
and one that showed some trades, so you can compare.
"""

import os
import numpy as np
import pandas as pd
from collections import deque
from sklearn.linear_model import LinearRegression

# ── CONFIG ─────────────────────────────────────────────────────────────────────
FILE_PATH   = '/Volumes/SEAGATE/FX_data/FX_histdata/'
TEST_START  = '2013-01-01'
TEST_END    = '2023-12-31'

# Run diagnostics on one zero-trade pair and one pair that did trade
PAIRS_TO_CHECK = [
    ('EURUSD', 'AUDJPY', '30T'),   # zero trades
    ('EURCHF', 'USDJPY', '1D'),    # had trades (37% return — suspicious)
    ('EURJPY', 'EURCHF', '30T'),   # had some paper trades
]

VALID_HALF_LIVES = {
    '30T': (3, 480),
    '1H':  (3, 240),
    '3H':  (3, 120),
    '6H':  (3,  80),
    '1D':  (3,  60),
}

# ── HELPERS ────────────────────────────────────────────────────────────────────
def load_close(pair, interval):
    path = os.path.join(FILE_PATH, f"{pair}/", f"{pair}_resampled_{interval}_returns.parquet")
    return pd.read_parquet(path, columns=['Close'])['Close']

def get_beta(s1, s2):
    from sklearn.linear_model import LinearRegression
    m = LinearRegression()
    m.fit(s1.values.reshape(-1, 1), s2.values)
    return m.coef_[0]

def compute_half_life(spread):
    lag = spread.shift(1).dropna()
    ret = (spread - spread.shift(1)).dropna()
    common = lag.index.intersection(ret.index)
    beta = get_beta(lag.loc[common], ret.loc[common])
    return -np.log(2) / beta if beta < 0 else np.nan

def get_rolling_calib(s1, s2, train_end, window_years=1, offset_months=3):
    """Reproduce get_rolling_half_life logic inline to avoid import issues."""
    common = s1.index.intersection(s2.index)
    s1, s2 = s1.loc[common], s2.loc[common]
    s1 = s1.loc[:train_end]
    s2 = s2.loc[:train_end]

    window_start_min = s1.index.min() + pd.DateOffset(months=offset_months)
    month_starts = pd.date_range(start=window_start_min, end=train_end, freq='MS')

    records = []
    for md in month_starts:
        ws = md - pd.DateOffset(years=window_years)
        we = md - pd.DateOffset(days=1)
        x, y = s1.loc[ws:we], s2.loc[ws:we]
        if len(x) < 30:
            records.append({'date': md, 'hedge_ratio': np.nan, 'half_life': np.nan})
            continue
        try:
            hr = get_beta(x, y)
            sp = y - hr * x
            hl = compute_half_life(sp)
            records.append({'date': md, 'hedge_ratio': hr, 'half_life': hl})
        except Exception:
            records.append({'date': md, 'hedge_ratio': np.nan, 'half_life': np.nan})

    return pd.DataFrame(records).set_index('date')


# ── MAIN DIAGNOSTIC ────────────────────────────────────────────────────────────
def diagnose(pair1, pair2, interval):
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"DIAGNOSTIC: {pair1}/{pair2} @ {interval}")
    print(sep)

    # 1. Load data
    s1 = load_close(pair1, interval)
    s2 = load_close(pair2, interval)
    common = s1.index.intersection(s2.index)
    s1, s2 = s1.loc[common], s2.loc[common]
    print(f"Data range: {s1.index.min().date()} → {s1.index.max().date()}")
    print(f"Total bars available: {len(s1)}")

    # 2. Calibration
    print(f"\n--- CALIBRATION ---")
    try:
        calib = get_rolling_calib(s1, s2, train_end=TEST_END)
    except Exception as e:
        print(f"  FAILED to compute calibration: {e}")
        return

    calib_in_test = calib.loc[TEST_START:TEST_END]
    print(f"Total calibration rows (all history to {TEST_END}): {len(calib)}")
    print(f"Calibration rows in test window [{TEST_START} : {TEST_END}]: {len(calib_in_test)}")
    print(f"Non-NaN hedge ratios: {calib_in_test['hedge_ratio'].notna().sum()}")
    print(f"Non-NaN half-lives:   {calib_in_test['half_life'].notna().sum()}")

    valid_hl = calib_in_test['half_life'].dropna()
    lo, hi = VALID_HALF_LIVES.get(interval, (3, 60))
    in_bounds  = valid_hl[(valid_hl >= lo) & (valid_hl <= hi)]
    out_bounds = valid_hl[(valid_hl < lo)  | (valid_hl > hi)]
    nan_count  = calib_in_test['half_life'].isna().sum()

    print(f"\nHalf-life bounds for {interval}: [{lo}, {hi}]")
    print(f"  In-bounds  : {len(in_bounds)}  ({100*len(in_bounds)/max(len(calib_in_test),1):.1f}%)")
    print(f"  Out-of-bounds: {len(out_bounds)}  ({100*len(out_bounds)/max(len(calib_in_test),1):.1f}%)  ← Lock 2 triggered here")
    print(f"  NaN        : {nan_count}")

    if len(valid_hl) > 0:
        print(f"\nHalf-life stats (non-NaN):")
        print(f"  min={valid_hl.min():.2f}  median={valid_hl.median():.2f}  max={valid_hl.max():.2f}")
        print(f"\nSample of calibration rows:")
        print(calib_in_test[['hedge_ratio','half_life']].head(10).to_string())

    # 3. Simulate spread and z-score for test period
    print(f"\n--- SPREAD & Z-SCORE (test period) ---")
    test_mask = (s1.index >= pd.Timestamp(TEST_START)) & (s1.index <= pd.Timestamp(TEST_END))
    s1t, s2t = s1[test_mask], s2[test_mask]

    # Use first available hedge ratio from calibration
    first_hr_row = calib_in_test.dropna().iloc[0] if len(calib_in_test.dropna()) > 0 else None
    if first_hr_row is None:
        print("  No valid calibration — cannot compute spread.")
        return

    hr = first_hr_row['hedge_ratio']
    print(f"Using hedge ratio: {hr:.6f} (from {first_hr_row.name.date()})")

    spread = s2t - hr * s1t
    spread_rolling_mean = spread.rolling(252, min_periods=30).mean()
    spread_rolling_std  = spread.rolling(252, min_periods=30).std()
    z = (spread - spread_rolling_mean) / spread_rolling_std

    print(f"Spread stats:  min={spread.min():.6f}  mean={spread.mean():.6f}  max={spread.max():.6f}")
    print(f"Z-score stats: min={z.min():.2f}  mean={z.mean():.2f}  max={z.max():.2f}")

    # Entry thresholds
    entry_z_map = {'30T': 2.5, '1H': 2.2, '3H': 2.0, '6H': 1.8, '1D': 1.5}
    entry_z = entry_z_map.get(interval, 2.0)
    long_signals  = (z < -entry_z).sum()
    short_signals = (z >  entry_z).sum()
    print(f"\nEntry threshold: ±{entry_z}")
    print(f"  Bars where z < -{entry_z} (long entry):  {long_signals}  ({100*long_signals/len(z):.2f}%)")
    print(f"  Bars where z >  {entry_z} (short entry): {short_signals}  ({100*short_signals/len(z):.2f}%)")
    print(f"  Total potential entry bars: {long_signals + short_signals}")

    if long_signals + short_signals == 0:
        print("\n  *** Z-SCORE NEVER BREACHES ENTRY THRESHOLD ***")
        print("  Likely causes:")
        print("  a) Spread is nearly constant (non-cointegrated pair)")
        print("  b) Entry threshold too high for this interval/pair")
        print("  c) Rolling window too short → noisy z-score")

    # 4. Log-price vs raw-price check
    print(f"\n--- LOG vs RAW PRICE CHECK ---")
    log_s1 = np.log(s1t)
    log_s2 = np.log(s2t)
    hr_log = get_beta(log_s1, log_s2)
    spread_log = log_s2 - hr_log * log_s1
    z_log_std  = spread_log.rolling(252, min_periods=30).std()
    z_log      = (spread_log - spread_log.rolling(252, min_periods=30).mean()) / z_log_std

    long_log  = (z_log < -entry_z).sum()
    short_log = (z_log >  entry_z).sum()
    print(f"Log-price hedge ratio: {hr_log:.6f}")
    print(f"Log-price z-score: min={z_log.min():.2f}  max={z_log.max():.2f}")
    print(f"  Entry signals (log): long={long_log}, short={short_log}")
    print(f"  Entry signals (raw): long={long_signals}, short={short_signals}")
    if (long_log + short_log) > (long_signals + short_signals):
        print("  *** Log prices generate more signals — consider switching ***")

    print(f"\n{'─'*60}")


if __name__ == '__main__':
    for p1, p2, iv in PAIRS_TO_CHECK:
        try:
            diagnose(p1, p2, iv)
        except Exception as e:
            print(f"\nFATAL ERROR for {p1}/{p2}@{iv}: {e}")
            import traceback
            traceback.print_exc()

    print("\n\nDIAGNOSTIC COMPLETE")