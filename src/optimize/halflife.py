"""
halflife_crisis_analysis.py
---------------------------
For each interval and each valid combo, computes rolling half-lives across
3 periods (pre/during/post crisis) using data up to 2012-12-31.
Outputs one PNG per interval showing 3 side-by-side histograms with
10th/90th percentile vertical lines.

Usage:
    from halflife_crisis_analysis import run_halflife_analysis

    valid_combos = [('EURUSD', 'EURCHF'), ('GBPUSD', 'USDJPY'), ...]  # list of (pair1, pair2) tuples
    run_halflife_analysis(valid_combos, file_path='/Volumes/SEAGATE/FX_data/FX_histdata/')
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.linear_model import LinearRegression
from itertools import product

# ── CONSTANTS ──────────────────────────────────────────────────────────────────

INTERVALS = ['1T', '5T', '10T', '15T', '30T', '1H', '3H', '6H', '1D']

PERIODS = {
    'Pre-Crisis':    ('2002-01-01', '2006-12-31'),
    'Crisis':        ('2007-01-01', '2009-12-31'),
    'Post-Crisis':   ('2010-01-01', '2012-12-31'),
}

PERIOD_COLORS = {
    'Pre-Crisis':  '#2196F3',   # blue
    'Crisis':      '#F44336',   # red
    'Post-Crisis': '#4CAF50',   # green
}

# Minimum bars required in a calibration window to compute hedge ratio
MIN_BARS = 30

# Rolling calibration window (months of data per calibration point)
CALIB_WINDOW_MONTHS = 12


# ── DATA HELPERS ───────────────────────────────────────────────────────────────

def _load_close(pair: str, interval: str, file_path: str) -> pd.Series:
    path = os.path.join(file_path, pair, f"{pair}_resampled_{interval}_returns.parquet")
    return pd.read_parquet(path, columns=['Close'])['Close']


def _get_beta(s1: pd.Series, s2: pd.Series) -> float:
    m = LinearRegression()
    m.fit(s1.values.reshape(-1, 1), s2.values)
    return float(m.coef_[0])


def _compute_half_life(spread: pd.Series) -> float:
    lag = spread.shift(1).dropna()
    ret = (spread - spread.shift(1)).dropna()
    idx = lag.index.intersection(ret.index)
    if len(idx) < MIN_BARS:
        return np.nan
    beta = _get_beta(lag.loc[idx], ret.loc[idx])
    return -np.log(2) / beta if beta < 0 else np.nan


def _align(s1: pd.Series, s2: pd.Series,
           start: str, end: str) -> tuple:
    """Intersect indices and slice to [start, end]."""
    common = s1.index.intersection(s2.index)
    s1, s2 = s1.loc[common], s2.loc[common]
    return s1.loc[start:end], s2.loc[start:end]


# ── CORE: rolling half-lives for one combo × one period ───────────────────────

def _rolling_half_lives(s1: pd.Series, s2: pd.Series,
                        period_start: str, period_end: str) -> list:
    """
    Monthly rolling calibration within [period_start, period_end].
    Each month's window = [month - CALIB_WINDOW_MONTHS, month - 1 day].
    Returns list of half-life values (NaN excluded).
    """
    s1p, s2p = _align(s1, s2, period_start, period_end)
    if len(s1p) < MIN_BARS:
        return []

    month_starts = pd.date_range(
        start=pd.Timestamp(period_start) + pd.DateOffset(months=CALIB_WINDOW_MONTHS),
        end=period_end,
        freq='MS'
    )

    half_lives = []
    for md in month_starts:
        ws = md - pd.DateOffset(months=CALIB_WINDOW_MONTHS)
        we = md - pd.DateOffset(days=1)
        x = s1p.loc[ws:we]
        y = s2p.loc[ws:we]
        if len(x) < MIN_BARS:
            continue
        try:
            hr  = _get_beta(np.log(x), np.log(y))
            sp  = np.log(y) - hr * np.log(x)
            hl  = _compute_half_life(sp)
            if not np.isnan(hl):
                half_lives.append(hl)
        except Exception:
            continue

    return half_lives


# ── MAIN ───────────────────────────────────────────────────────────────────────

def run_halflife_analysis(
    valid_combos: list,
    file_path: str = '/Volumes/SEAGATE/FX_data/FX_histdata/',
    output_dir: str = './temp/halflife_plots',
    interval: str = None,
    verbose: bool = True,
):
    """
    Parameters
    ----------
    valid_combos : list of (pair1, pair2) tuples
    file_path    : root directory of FX parquet data
    output_dir   : where to save PNGs
    interval     : specific interval to run; defaults to all
    verbose      : print progress
    """
    os.makedirs(output_dir, exist_ok=True)
    if interval is None:
        raise ValueError("interval parameter cannot be None. Provide an interval or use the default.")

    # for interval in intervals:
    if verbose:
        print(f"\n{'='*60}")
        print(f"Interval: {interval}  |  Combos: {len(valid_combos)}")
        print('='*60)

    # Collect half-lives per period across all combos
    period_half_lives = {p: [] for p in PERIODS}

    for idx, (pair1, pair2) in enumerate(valid_combos):
        if verbose:
            print(f"  [{idx+1}/{len(valid_combos)}] {pair1}/{pair2}", end='  ')

        # Load once per combo
        try:
            s1 = _load_close(pair1, interval, file_path)
            s2 = _load_close(pair2, interval, file_path)
        except FileNotFoundError as e:
            if verbose:
                print(f"SKIP (file not found: {e})")
            continue

        for period_name, (pstart, pend) in PERIODS.items():
            hls = _rolling_half_lives(s1, s2, pstart, pend)
            period_half_lives[period_name].extend(hls)
            if verbose:
                print(f"{period_name}: {len(hls)} pts", end='  ')
        if verbose:
            print()

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    fig.suptitle(
        f'Half-Life Distribution — {interval}\n'
        f'(all valid combos, data up to 2012-12-31)',
        fontsize=13, fontweight='bold', y=1.02
    )

    for ax, (period_name, (pstart, pend)) in zip(axes, PERIODS.items()):
        data = np.array(period_half_lives[period_name])
        color = PERIOD_COLORS[period_name]

        if len(data) < 2:
            ax.text(0.5, 0.5, 'Insufficient data',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{period_name}\n({pstart[:4]}–{pend[:4]})')
            continue

        # Clip extreme outliers for readability (plot up to 99th pct)
        clip_hi = np.percentile(data, 99)
        data_clipped = data[data <= clip_hi]

        p10 = np.percentile(data, 10)
        p60 = np.percentile(data, 60)
        p70 = np.percentile(data, 70)
        p80 = np.percentile(data, 80)
        p90 = np.percentile(data, 90)

        n_bins = min(80, max(20, len(data_clipped) // 10))
        ax.hist(data_clipped, bins=n_bins, color=color, alpha=0.75,
                edgecolor='white', linewidth=0.4)

        # Percentile lines
        ax.axvline(p10, color='black', linestyle='--', linewidth=1.5,
                    label=f'P10 = {p10:.1f}')
        ax.axvline(p60, color='black', linestyle='-.', linewidth=1.5,
                    label=f'P60 = {p60:.1f}')
        ax.axvline(p70, color='black', linestyle=':',  linewidth=1.5,
                    label=f'P70 = {p70:.1f}')
        ax.axvline(p80, color='black', linestyle=':',  linewidth=1.5,
                    label=f'P80 = {p80:.1f}')
        ax.axvline(p90, color='black', linestyle=':',  linewidth=1.5,
                    label=f'P90 = {p90:.1f}')

        ax.set_title(f'{period_name}\n({pstart[:4]}–{pend[:4]})',
                        fontsize=11, fontweight='bold')
        ax.set_xlabel('Half-Life (bars)', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f'{x:,.0f}'))

        # Annotation: n, median
        ax.text(0.97, 0.95,
                f'n = {len(data):,}\nmedian = {np.median(data):.1f}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            alpha=0.7))

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'halflife_{interval}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    if verbose:
        print(f"  Saved → {out_path}")
        for pname, vals in period_half_lives.items():
            if vals:
                arr = np.array(vals)
                print(f"    {pname}: n={len(arr)}, "
                        f"P10={np.percentile(arr,10):.1f}, "
                        f"median={np.median(arr):.1f}, "
                        f"P50={np.percentile(arr,50):.1f},"
                        f"P60={np.percentile(arr,60):.1f},"
                        f"P70={np.percentile(arr,70):.1f},"
                        f"P80={np.percentile(arr,80):.1f},"
                        f"P90={np.percentile(arr,90):.1f}"
                )

    print(f"\nDone. Plots saved to: {output_dir}")


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Example: load valid_combos from a CSV with columns pair1, pair2
    import sys

    if len(sys.argv) < 2:
        print("Usage: python halflife_crisis_analysis.py <valid_combos_path> [file_path] [output_dir]")
        print("  valid_combos_path: CSV with columns 'pair1','pair2'")
        sys.exit(1)

    combos_path = sys.argv[1]
    fp          = sys.argv[2] if len(sys.argv) > 2 else '/Volumes/SEAGATE/FX_data/FX_histdata/'
    out         = sys.argv[3] if len(sys.argv) > 3 else './halflife_plots'

    combos_df = pd.read_csv(combos_path)
    combos    = list(zip(combos_df['pair1'], combos_df['pair2']))
    print(f"Loaded {len(combos)} combos from {combos_path}")

    run_halflife_analysis(combos, file_path=fp, output_dir=out)