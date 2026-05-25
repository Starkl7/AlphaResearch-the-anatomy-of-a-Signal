"""
Diagnose why a pair has zero trades or anomalous returns.
Edit PAIRS_TO_CHECK below, then run from project root:
    python scripts/diagnose.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from collections import deque

from src.analysis.halflife import get_rolling_half_life
from src.analysis.stats import get_beta_lr
from src.data.loader import load_resampled_returns

FILE_PATH   = '/Volumes/SEAGATE/FX_data/FX_histdata/'
TEST_START  = '2013-01-01'
TEST_END    = '2023-12-31'

PAIRS_TO_CHECK = [
    ('EURUSD', 'AUDJPY', '30T'),   # zero trades
    ('EURCHF', 'USDJPY', '1D'),    # had trades (37% return — suspicious)
    ('EURJPY', 'EURCHF', '30T'),   # had some paper trades
]

VALID_HALF_LIVES = {
    '30T': (3, 480), '1H': (3, 240), '3H': (3, 120),
    '6H':  (3,  80), '1D': (3,  60),
}


def compute_half_life(spread):
    lag = spread.shift(1).dropna()
    ret = (spread - spread.shift(1)).dropna()
    common = lag.index.intersection(ret.index)
    beta = get_beta_lr(lag.loc[common], ret.loc[common])
    return -np.log(2) / beta if beta < 0 else np.nan


def diagnose(pair1, pair2, interval):
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"DIAGNOSTIC: {pair1}/{pair2} @ {interval}")
    print(sep)

    s1_df = load_resampled_returns(pair1, interval, FILE_PATH)
    s2_df = load_resampled_returns(pair2, interval, FILE_PATH)
    s1 = s1_df['Close']
    s2 = s2_df['Close']
    common = s1.index.intersection(s2.index)
    s1, s2 = s1.loc[common], s2.loc[common]
    print(f"Data range: {s1.index.min().date()} → {s1.index.max().date()}")
    print(f"Total bars available: {len(s1)}")

    print(f"\n--- CALIBRATION ---")
    try:
        calib = get_rolling_half_life(pair1, pair2, interval, FILE_PATH, train_end=TEST_END)
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

    print(f"\nHalf-life bounds for {interval}: [{lo}, {hi}]")
    print(f"  In-bounds    : {len(in_bounds)}  ({100*len(in_bounds)/max(len(calib_in_test),1):.1f}%)")
    print(f"  Out-of-bounds: {len(out_bounds)}  ({100*len(out_bounds)/max(len(calib_in_test),1):.1f}%)")
    print(f"  NaN          : {calib_in_test['half_life'].isna().sum()}")

    if len(valid_hl) > 0:
        print(f"\nHalf-life stats: min={valid_hl.min():.2f}  median={valid_hl.median():.2f}  max={valid_hl.max():.2f}")
        print(calib_in_test[['hedge_ratio', 'half_life']].head(10).to_string())

    print(f"\n--- SPREAD & Z-SCORE (test period) ---")
    test_mask = (s1.index >= pd.Timestamp(TEST_START)) & (s1.index <= pd.Timestamp(TEST_END))
    s1t, s2t = s1[test_mask], s2[test_mask]

    first_hr_row = calib_in_test.dropna().iloc[0] if len(calib_in_test.dropna()) > 0 else None
    if first_hr_row is None:
        print("  No valid calibration — cannot compute spread.")
        return

    hr = first_hr_row['hedge_ratio']
    print(f"Using hedge ratio: {hr:.6f} (from {first_hr_row.name.date()})")

    spread = s2t - hr * s1t
    z_mean = spread.rolling(252, min_periods=30).mean()
    z_std  = spread.rolling(252, min_periods=30).std()
    z      = (spread - z_mean) / z_std

    print(f"Spread:  min={spread.min():.6f}  mean={spread.mean():.6f}  max={spread.max():.6f}")
    print(f"Z-score: min={z.min():.2f}  mean={z.mean():.2f}  max={z.max():.2f}")

    entry_z = 2.5
    print(f"\nEntry threshold: ±{entry_z}")
    print(f"  Long entries (z < -{entry_z}):  {(z < -entry_z).sum()}")
    print(f"  Short entries (z > {entry_z}): {(z > entry_z).sum()}")

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
