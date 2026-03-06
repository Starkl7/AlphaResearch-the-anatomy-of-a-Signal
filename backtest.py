import os
import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass
from pprint import pprint
import helper_functions as fx


# --------------------------------------------------------------------------- #
# Per-interval valid half-life bounds (bars)
# --------------------------------------------------------------------------- #
VALID_HALF_LIVES = {
    '1T':  (5, 4320),
    '5T':  (5, 1440),
    '10T': (5,  720),
    '15T': (5,  480),
    '30T': (3,  480),
    '1H':  (3,  240),
    '3H':  (3,  120),
    '6H':  (3,   80),
    '1D':  (3,   60),
}


# --------------------------------------------------------------------------- #
# Tunable parameter stubs (static now, dynamic-ready later)
# --------------------------------------------------------------------------- #
def get_entry_z(ps=None):
    return 2.5

def get_tp_z(ps=None):
    return 0.5

def get_sl_z(ps=None):
    return 4.0

def get_time_limit(half_life):
    return int(half_life * 2)

def get_position_size(portfolio_value, ps=None):
    """0.1% of capital per trade — replace body for dynamic sizing."""
    return portfolio_value * 0.001


# --------------------------------------------------------------------------- #
# State object
# --------------------------------------------------------------------------- #
@dataclass
class PairState:
    # Calibration
    hedge_ratio:      float = np.nan
    half_life:        float = np.nan
    prev_hedge_ratio: float = np.nan
    prev_half_life:   float = np.nan

    # Lock flags
    lock1: bool = False   # 5 consecutive SL → paper
    lock2: bool = False   # half-life out of bounds → monitoring
    lock3: bool = False   # hedge ratio change > 30% → monitoring
    lock4: bool = False   # half-life change > 50% → monitoring
    lock5: bool = False   # drawdown > 15% → paper

    # Lock 1 counters
    consec_sl:     int = 0
    l1_paper_wins: int = 0

    # Lock 5 counters
    peak_value:    float = 0.0
    l5_paper_wins: int = 0

    # Locks 2/3/4 recovery counters (3 clean months to unlock each)
    l2_months_ok: int = 0
    l3_months_ok: int = 0
    l4_months_ok: int = 0

    # Open trade
    in_position:  bool  = False
    direction:    int   = 0        # +1 long spread, -1 short spread
    entry_bar:    int   = 0
    entry_spread: float = 0.0
    is_paper:     bool  = False

    # Portfolio
    portfolio_value: float = 100_000.0


# --------------------------------------------------------------------------- #
# State helpers
# --------------------------------------------------------------------------- #
def _state(ps: PairState) -> str:
    blocking   = ps.lock1 or ps.lock5
    monitoring = ps.lock2 or ps.lock3 or ps.lock4
    if blocking and monitoring: return 'SUSPENDED'
    if blocking:                return 'PAPER'
    if monitoring:              return 'MONITORING'
    return 'LIVE'

def _can_trade_real(ps: PairState) -> bool:
    return _state(ps) == 'LIVE'

def _can_trade_paper(ps: PairState) -> bool:
    return _state(ps) in ('PAPER', 'SUSPENDED')

def _no_trade(ps: PairState) -> bool:
    return _state(ps) == 'MONITORING'


# --------------------------------------------------------------------------- #
# Monthly recalibration — updates hedge_ratio, half_life and Locks 2/3/4
# --------------------------------------------------------------------------- #
def _recalibrate(ps: PairState, month_calib_row, interval: str):
    new_hr = month_calib_row['hedge_ratio']
    new_hl = month_calib_row['half_life']

    ps.prev_hedge_ratio = ps.hedge_ratio
    ps.prev_half_life   = ps.half_life
    ps.hedge_ratio      = new_hr
    ps.half_life        = new_hl

    lo, hi = VALID_HALF_LIVES.get(interval, (3, 60))

    # Lock 2 — half-life out of bounds
    hl_valid = (not np.isnan(new_hl)) and (lo <= new_hl <= hi)
    if not hl_valid:
        ps.lock2 = True
        ps.l2_months_ok = 0
    elif ps.lock2:
        ps.l2_months_ok += 1
        if ps.l2_months_ok >= 3:
            ps.lock2 = False
            ps.l2_months_ok = 0

    # Lock 3 — hedge ratio change > 30%
    if not np.isnan(ps.prev_hedge_ratio) and ps.prev_hedge_ratio != 0:
        hr_chg = abs((new_hr - ps.prev_hedge_ratio) / ps.prev_hedge_ratio)
        if hr_chg > 0.30:
            ps.lock3 = True
            ps.l3_months_ok = 0
        elif ps.lock3:
            ps.l3_months_ok += 1
            if ps.l3_months_ok >= 3:
                ps.lock3 = False
                ps.l3_months_ok = 0

    # Lock 4 — half-life change > 50%
    if not np.isnan(ps.prev_half_life) and ps.prev_half_life != 0:
        hl_chg = abs((new_hl - ps.prev_half_life) / ps.prev_half_life)
        if hl_chg > 0.50:
            ps.lock4 = True
            ps.l4_months_ok = 0
        elif ps.lock4:
            ps.l4_months_ok += 1
            if ps.l4_months_ok >= 3:
                ps.lock4 = False
                ps.l4_months_ok = 0


# --------------------------------------------------------------------------- #
# Exit logic
# --------------------------------------------------------------------------- #
def _check_exit(ps: PairState, z: float, bars_held: int):
    if abs(z) < get_tp_z(ps):                      return 'TP'
    if abs(z) > get_sl_z(ps):                      return 'SL'
    if bars_held >= get_time_limit(ps.half_life):   return 'TIME'
    return None


def _on_exit(ps: PairState, exit_type: str, dollar_pnl: float):
    is_win = exit_type in ('TP', 'TIME')

    if ps.is_paper:
        # Paper wins/losses count towards unlocking Lock 1 and Lock 5 independently
        if is_win:
            if ps.lock1: ps.l1_paper_wins += 1
            if ps.lock5: ps.l5_paper_wins += 1
        else:
            if ps.lock1: ps.l1_paper_wins = 0
            if ps.lock5: ps.l5_paper_wins = 0

        # 5 consecutive paper wins unlocks each blocking lock
        if ps.lock1 and ps.l1_paper_wins >= 5:
            ps.lock1 = False
            ps.l1_paper_wins = 0
        if ps.lock5 and ps.l5_paper_wins >= 5:
            ps.lock5 = False
            ps.l5_paper_wins = 0

    else:
        # Real trade: Lock 1 consecutive SL tracker
        # TP resets counter; SL increments; TIME leaves unchanged
        if exit_type == 'SL':
            ps.consec_sl += 1
            if ps.consec_sl >= 5:
                ps.lock1 = True
                ps.consec_sl = 0
                ps.l1_paper_wins = 0
        elif exit_type == 'TP':
            ps.consec_sl = 0

        # Real trade: update portfolio and check Lock 5 drawdown
        ps.portfolio_value += dollar_pnl
        ps.peak_value = max(ps.peak_value, ps.portfolio_value)
        if ps.peak_value > 0:
            drawdown = (ps.peak_value - ps.portfolio_value) / ps.peak_value
            if drawdown > 0.15:
                ps.lock5 = True
                ps.l5_paper_wins = 0


# --------------------------------------------------------------------------- #
# Record helpers
# --------------------------------------------------------------------------- #
def _drawdown(ps: PairState) -> float:
    if ps.peak_value > 0:
        return (ps.peak_value - ps.portfolio_value) / ps.peak_value
    return 0.0


def _empty_record(dt, p1, p2, ps: PairState) -> dict:
    return {
        'datetime':        dt,
        'price1':          p1,
        'price2':          p2,
        'spread':          np.nan,
        'z_score':         np.nan,
        'hedge_ratio':     ps.hedge_ratio,
        'half_life':       ps.half_life,
        'signal':          0,
        'is_paper':        False,
        'state':           _state(ps),
        'portfolio_value': ps.portfolio_value,
        'drawdown':        _drawdown(ps),
        'lock1':           ps.lock1,
        'lock2':           ps.lock2,
        'lock3':           ps.lock3,
        'lock4':           ps.lock4,
        'lock5':           ps.lock5,
    }


def _make_record(dt, p1, p2, spread, z, ps: PairState) -> dict:
    return {
        'datetime':        dt,
        'price1':          p1,
        'price2':          p2,
        'spread':          spread,
        'z_score':         z,
        'hedge_ratio':     ps.hedge_ratio,
        'half_life':       ps.half_life,
        'signal':          ps.direction if ps.in_position else 0,
        'is_paper':        ps.is_paper if ps.in_position else False,
        'state':           _state(ps),
        'portfolio_value': ps.portfolio_value,
        'drawdown':        _drawdown(ps),
        'lock1':           ps.lock1,
        'lock2':           ps.lock2,
        'lock3':           ps.lock3,
        'lock4':           ps.lock4,
        'lock5':           ps.lock5,
    }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _compute_metrics(results: pd.DataFrame, ps: PairState,
                     n_real: int, n_paper: int,
                     real_wins: int, real_losses: int) -> dict:
    total_ret = (ps.portfolio_value - 100_000) / 100_000
    max_dd    = results['drawdown'].max()
    win_rate  = real_wins / n_real if n_real > 0 else np.nan

    return {
        'final_portfolio_value': round(ps.portfolio_value, 2),
        'total_return_pct':      round(total_ret * 100, 2),
        'max_drawdown_pct':      round(max_dd * 100, 2),
        'n_real_trades':         n_real,
        'n_paper_trades':        n_paper,
        'real_win_rate':         round(win_rate, 3) if not np.isnan(win_rate) else np.nan,
        'real_wins':             real_wins,
        'real_losses':           real_losses,
        'n_bars':                len(results),
    }


# --------------------------------------------------------------------------- #
# Main backtest function
# --------------------------------------------------------------------------- #
def run_pair_interval_backtest(
    pair1: str,
    pair2: str,
    interval: str,
    file_path: str,
    test_start: str = '2013-01-01',
    test_end:   str = '2017-12-31',
    portfolio_value: float = 100_000.0,
) -> tuple:

    # 1. Load Close prices (DatetimeIndex comes along automatically)
    def _load(pair):
        path = os.path.join(file_path, f"{pair}/", f"{pair}_resampled_{interval}_returns.parquet")
        return pd.read_parquet(path, columns=['Close'])

    df1 = _load(pair1)
    df2 = _load(pair2)

    # 2. Pre-compute monthly calibrations across full history up to test_end.
    #    Each month's window = [month - 1yr, month], so no look-ahead bias.
    print(f"  Computing rolling calibrations...")
    calib = fx.get_rolling_half_life(
        pair1, pair2, interval, file_path,
        feature='Close', train_end=test_end
    )

    # 3. Align to common index
    common_idx = df1.index.intersection(df2.index)
    s1_all = df1.loc[common_idx, 'Close']
    s2_all = df2.loc[common_idx, 'Close']

    # 4. Pre-seed z-score buffer with 1 year of pre-test data so trading
    #    can begin from bar 1 of the test period without warmup loss.
    spread_buffer: deque = deque()   # elements: (pd.Timestamp, float)

    pre_calib = calib.loc[:test_start].dropna()
    if not pre_calib.empty:
        seed_hr = float(pre_calib.iloc[-1]['hedge_ratio'])
        preseed_start = pd.Timestamp(test_start) - pd.DateOffset(years=1)
        preseed_mask  = (s1_all.index >= preseed_start) & (s1_all.index < pd.Timestamp(test_start))
        for dt_ps in s1_all[preseed_mask].index:
            sp = float(s2_all.loc[dt_ps]) - seed_hr * float(s1_all.loc[dt_ps])
            spread_buffer.append((dt_ps, sp))

    # 5. Test period slices
    test_mask = (
        (s1_all.index >= pd.Timestamp(test_start)) &
        (s1_all.index <= pd.Timestamp(test_end))
    )
    s1_test = s1_all[test_mask]
    s2_test = s2_all[test_mask]

    # 6. Initialise state
    ps = PairState(portfolio_value=portfolio_value, peak_value=portfolio_value)
    records       = []
    current_month = None
    bars_held     = 0

    # Trade counters for metrics
    n_real      = 0
    n_paper     = 0
    real_wins   = 0
    real_losses = 0

    # 7. Bar-by-bar loop
    for t, dt in enumerate(s1_test.index):
        p1 = float(s1_test.iloc[t])
        p2 = float(s2_test.iloc[t])

        # Monthly recalibration at the start of every new calendar month
        month_key = dt.to_period('M')
        if month_key != current_month:
            current_month = month_key
            month_start   = month_key.to_timestamp()
            if month_start in calib.index:
                row = calib.loc[month_start]
                if not row.isna().any():
                    _recalibrate(ps, row, interval)

        # Skip bars until first calibration is available
        if np.isnan(ps.hedge_ratio):
            records.append(_empty_record(dt, p1, p2, ps))
            continue

        # Compute spread and calendar 1-year rolling z-score
        spread = p2 - ps.hedge_ratio * p1
        spread_buffer.append((dt, spread))
        cutoff = dt - pd.DateOffset(years=1)
        while spread_buffer and spread_buffer[0][0] < cutoff:
            spread_buffer.popleft()

        buf = np.array([v for _, v in spread_buffer])
        if len(buf) < 30:
            records.append(_empty_record(dt, p1, p2, ps))
            continue
        z = (spread - buf.mean()) / buf.std()

        # Exit open position
        if ps.in_position:
            bars_held += 1
            exit_type  = _check_exit(ps, z, bars_held)
            if exit_type:
                trade_pnl  = ps.direction * (spread - ps.entry_spread)
                pos_size   = get_position_size(ps.portfolio_value, ps)
                dollar_pnl = trade_pnl * pos_size if not ps.is_paper else 0.0

                was_paper = ps.is_paper
                _on_exit(ps, exit_type, dollar_pnl)

                if was_paper:
                    n_paper += 1
                else:
                    n_real += 1
                    if exit_type in ('TP', 'TIME'):
                        real_wins += 1
                    else:
                        real_losses += 1

                ps.in_position = False
                ps.direction   = 0
                bars_held      = 0

        # Enter new position (MONITORING state → no entry of any kind)
        if not ps.in_position and not _no_trade(ps):
            entry_z = get_entry_z(ps)
            if z < -entry_z:
                ps.direction    = 1
                ps.entry_spread = spread
                ps.entry_bar    = t
                ps.in_position  = True
                bars_held       = 0
                ps.is_paper     = not _can_trade_real(ps)
            elif z > entry_z:
                ps.direction    = -1
                ps.entry_spread = spread
                ps.entry_bar    = t
                ps.in_position  = True
                bars_held       = 0
                ps.is_paper     = not _can_trade_real(ps)

        records.append(_make_record(dt, p1, p2, spread, z, ps))

    results = pd.DataFrame(records).set_index('datetime')
    metrics = _compute_metrics(results, ps, n_real, n_paper, real_wins, real_losses)
    return results, metrics


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    file_path = '/Volumes/SEAGATE/FX_data/FX_histdata/'
    pair1, pair2 = 'AUDJPY', 'EURAUD'

    output_dir = './backtest_results'
    os.makedirs(output_dir, exist_ok=True)

    for interval in ['1H', '3H', '6H', '1D']:
        print(f"\n{'='*60}")
        print(f"Running backtest: {pair1}/{pair2} @ {interval}")
        print('='*60)
        try:
            results, metrics = run_pair_interval_backtest(
                pair1, pair2, interval, file_path
            )
            pprint(metrics)
            out_path = os.path.join(output_dir, f'{pair1}_{pair2}_{interval}.csv')
            results.to_csv(out_path)
            print(f"Results saved to: {out_path}")
        except Exception as e:
            print(f"Error: {e}")
