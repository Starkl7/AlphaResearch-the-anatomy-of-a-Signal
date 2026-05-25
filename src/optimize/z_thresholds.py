"""
Z-Threshold Bayesian Optimizer
================================
- Optuna-based Bayesian optimization of entry_z, tp_z, sl_z, time_mult
- Per-interval optimization across all pairs
- Objective: Sharpe ratio (annualised)
- Hard constraints: tp_z < entry_z < sl_z
- Early pruning: kill trials with bad intermediate Sharpe
- Train: start → 2014-12-31
- Test:  2015-01-01 → 2023-12-31
- Bayesian only for 1D, 6H, 3H (slow intervals)
- Coarser grid for 1H, 30T
"""

import os
import warnings
import numpy as np
import pandas as pd
import optuna
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from src.analysis.halflife import get_rolling_half_life

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Constants ─────────────────────────────────────────────────────────────────

VALID_HALF_LIVES = {
    '1T':  (5, 18000), '5T':  (5, 4000), '10T': (5, 2500),
    '15T': (5,   850), '30T': (3,  480), '1H':  (3,  450),
    '3H':  (3,   175), '6H':  (3,   95), '1D':  (3,   35),
}

TRANSACTION_COST_PCT = 0.00005   # 0.005% per leg

BARS_PER_YEAR = {
    '1T': 525_600, '5T': 105_120, '10T': 52_560, '15T': 35_040,
    '30T': 17_520, '1H': 8_760,   '3H':  2_920,  '6H':  1_460, '1D': 252,
}

# Intervals to use Bayesian optimization vs coarse grid
BAYESIAN_INTERVALS = {'1D', '6H', '3H'}
GRID_INTERVALS     = {'1H', '30T'}

# Pruning: if Sharpe after PRUNE_AFTER_YEARS is below this, kill trial
PRUNE_SHARPE_THRESHOLD = -1.0
PRUNE_AFTER_YEARS      = 5

MIN_TRADES = 30   # minimum real trades for a valid trial

# ── Position sizing ───────────────────────────────────────────────────────────

MIN_POSITION_PCT = 0.05   # 5% floor
MAX_POSITION_PCT = 0.15   # 15% cap
VOL_TARGET_PCT   = 0.10   # 10% average target

def _get_position_size(
    portfolio_value: float,
    spread_vol:      float,
    all_spread_vols: Dict,
) -> float:
    """Inverse-vol sizing clamped to [5%, 15%] of portfolio_value."""
    min_n = portfolio_value * MIN_POSITION_PCT
    max_n = portfolio_value * MAX_POSITION_PCT

    if spread_vol <= 0 or not all_spread_vols:
        return np.clip(portfolio_value * VOL_TARGET_PCT, min_n, max_n)

    inv_vols = {k: 1.0 / v for k, v in all_spread_vols.items() if v > 0}
    if not inv_vols:
        return np.clip(portfolio_value * VOL_TARGET_PCT, min_n, max_n)

    total_inv_vol = sum(inv_vols.values())
    weight_i      = (1.0 / spread_vol) / total_inv_vol
    notional      = portfolio_value * VOL_TARGET_PCT * weight_i * len(inv_vols)
    return float(np.clip(notional, min_n, max_n))

# ── Pair state (minimal — no Lock5) ──────────────────────────────────────────

@dataclass
class PairState:
    hedge_ratio:       float  = np.nan
    half_life:         float  = np.nan
    prev_hedge_ratio:  float  = np.nan
    prev_half_life:    float  = np.nan
    lock1:             bool   = False
    lock2:             bool   = False
    lock3:             bool   = False
    lock4:             bool   = False
    consec_losses:     int    = 0
    l1_paper_wins:     int    = 0
    l2_months_ok:      int    = 0
    l3_months_ok:      int    = 0
    l4_months_ok:      int    = 0
    in_position:       bool   = False
    direction:         int    = 0
    entry_spread:      float  = 0.0
    entry_price1:      float  = 0.0
    entry_price2:      float  = 0.0
    entry_z_score:     float  = 0.0
    is_paper:          bool   = False
    entry_time:        object = None
    allocated_notional: float = 0.0
    bars_held:         int    = 0


@dataclass
class PortfolioState:
    portfolio_value:  float = 100_000.0
    peak_value:       float = 100_000.0
    deployed_capital: float = 0.0

    def available_capital(self) -> float:
        return self.portfolio_value - self.deployed_capital

    def drawdown(self) -> float:
        if self.peak_value > 0:
            return (self.peak_value - self.portfolio_value) / self.peak_value
        return 0.0

    def update_peak(self):
        self.peak_value = max(self.peak_value, self.portfolio_value)


# ── State helpers ─────────────────────────────────────────────────────────────

def _state(ps: PairState) -> str:
    blocking   = ps.lock1
    monitoring = ps.lock2 or ps.lock3 or ps.lock4
    if blocking and monitoring: return 'SUSPENDED'
    if blocking:                return 'PAPER'
    if monitoring:              return 'MONITORING'
    return 'LIVE'

def _can_trade_real(ps): return _state(ps) == 'LIVE'
def _no_trade(ps):       return _state(ps) == 'MONITORING'


def _recalibrate(ps: PairState, row, interval: str):
    new_hr = float(row['hedge_ratio'])
    new_hl = float(row['half_life'])
    ps.prev_hedge_ratio = ps.hedge_ratio
    ps.prev_half_life   = ps.half_life
    ps.hedge_ratio      = new_hr
    ps.half_life        = new_hl
    lo, hi = VALID_HALF_LIVES.get(interval, (3, 60))

    hl_valid = (not np.isnan(new_hl)) and (lo <= new_hl <= hi)
    if not hl_valid:
        ps.lock2 = True; ps.l2_months_ok = 0
    elif ps.lock2:
        ps.l2_months_ok += 1
        if ps.l2_months_ok >= 3: ps.lock2 = False; ps.l2_months_ok = 0

    if not np.isnan(ps.prev_hedge_ratio) and ps.prev_hedge_ratio != 0:
        hr_chg = abs((new_hr - ps.prev_hedge_ratio) / ps.prev_hedge_ratio)
        if hr_chg > 1000:
            ps.lock3 = True; ps.l3_months_ok = 0
        elif ps.lock3:
            ps.l3_months_ok += 1
            if ps.l3_months_ok >= 3: ps.lock3 = False; ps.l3_months_ok = 0

    if not np.isnan(ps.prev_half_life) and ps.prev_half_life != 0:
        hl_chg = abs((new_hl - ps.prev_half_life) / ps.prev_half_life)
        if hl_chg > 1_000_000:
            ps.lock4 = True; ps.l4_months_ok = 0
        elif ps.lock4:
            ps.l4_months_ok += 1
            if ps.l4_months_ok >= 3: ps.lock4 = False; ps.l4_months_ok = 0


# ── Core backtest (stripped for speed — no trade records during optimization) ─

def _run_backtest_fast(
    price_data:    Dict,
    calibrations:  Dict,
    valid_pairs:   List[Tuple[str, str]],
    interval:      str,
    test_start:    str,
    test_end:      str,
    entry_z:       float,
    tp_z:          float,
    sl_z:          float,
    time_mult:     float,
    initial_capital: float = 100_000.0,
    prune_date:    Optional[str] = None,
) -> dict:
    """
    Stripped-down universe backtest returning Sharpe + trade count.
    Uses inverse-vol position sizing. No trade record storage for speed.
    """
    port = PortfolioState(
        portfolio_value  = initial_capital,
        peak_value       = initial_capital,
        deployed_capital = 0.0,
    )

    pair_states:    Dict[str, PairState] = {f"{p1}/{p2}": PairState() for p1, p2 in valid_pairs}
    spread_buffers: Dict[str, deque]     = {}
    current_months: Dict[str, object]   = {f"{p1}/{p2}": None for p1, p2 in valid_pairs}

    # Pre-seed spread buffers
    for p1, p2 in valid_pairs:
        key = f"{p1}/{p2}"
        spread_buffers[key] = deque()
        calib     = calibrations[key]
        pre_calib = calib.loc[:test_start].dropna()
        if not pre_calib.empty:
            seed_hr   = float(pre_calib.iloc[-1]['hedge_ratio'])
            s1, s2    = price_data[key]
            preseed_start = pd.Timestamp(test_start) - pd.DateOffset(years=1)
            mask = (s1.index >= preseed_start) & (s1.index < pd.Timestamp(test_start))
            for dt_ps in s1[mask].index:
                sp = np.log(float(s2.loc[dt_ps])) - seed_hr * np.log(float(s1.loc[dt_ps]))
                spread_buffers[key].append((dt_ps, sp))

    # Build time index
    all_indices = []
    for p1, p2 in valid_pairs:
        s1, _ = price_data[f"{p1}/{p2}"]
        mask  = (s1.index >= pd.Timestamp(test_start)) & (s1.index <= pd.Timestamp(test_end))
        all_indices.append(s1[mask].index)
    if not all_indices:
        return {'sharpe': -999.0, 'n_real': 0}

    time_index = all_indices[0]
    for idx in all_indices[1:]:
        time_index = time_index.union(idx)
    time_index = time_index.sort_values()

    n_real       = 0
    bar_returns  = []
    prev_value   = initial_capital
    prune_sharpe = None
    prune_ts     = pd.Timestamp(prune_date) if prune_date else None

    for dt in time_index:
        entry_signals = []

        for p1, p2 in valid_pairs:
            key = f"{p1}/{p2}"
            ps  = pair_states[key]
            s1, s2 = price_data[key]

            if dt not in s1.index:
                continue

            p1v = float(s1.loc[dt])
            p2v = float(s2.loc[dt])

            # Recalibrate
            month_key = dt.to_period('M')
            if month_key != current_months[key]:
                current_months[key] = month_key
                calib       = calibrations[key]
                month_calib = calib[calib.index.to_period('M') == month_key]
                if len(month_calib) > 0:
                    row = month_calib.iloc[0]
                    if not np.isnan(float(row['hedge_ratio'])):
                        _recalibrate(ps, row, interval)

            if np.isnan(ps.hedge_ratio):
                continue

            # Spread & z
            spread = np.log(p2v) - ps.hedge_ratio * np.log(p1v)
            buf    = spread_buffers[key]
            buf.append((dt, spread))
            cutoff = dt - pd.DateOffset(years=1)
            while buf and buf[0][0] < cutoff:
                buf.popleft()
            arr = np.array([v for _, v in buf])
            if len(arr) < 30:
                continue
            z = (spread - arr.mean()) / arr.std()

            # Exit
            if ps.in_position:
                ps.bars_held += 1
                time_limit = int(ps.half_life * time_mult) if not np.isnan(ps.half_life) else 1000
                exit_type  = None
                if abs(z) < tp_z:                    exit_type = 'TP'
                elif abs(z) > sl_z:                  exit_type = 'SL'
                elif ps.bars_held >= time_limit:     exit_type = 'TIME'

                if exit_type:
                    notional   = ps.allocated_notional
                    gross_leg1 = abs(ps.hedge_ratio) * ps.entry_price1
                    gross_leg2 = ps.entry_price2
                    N          = notional / (gross_leg1 + gross_leg2)
                    pnl_pu     = ps.direction * ((p2v - ps.entry_price2) - ps.hedge_ratio * (p1v - ps.entry_price1))
                    dollar_gross = N * pnl_pu
                    txn_cost     = TRANSACTION_COST_PCT * 2 * notional
                    dollar_pnl   = (dollar_gross - txn_cost) if not ps.is_paper else 0.0

                    # Release capital
                    if not ps.is_paper:
                        port.deployed_capital = max(0.0, port.deployed_capital - notional)
                        port.portfolio_value += dollar_pnl
                        port.update_peak()
                        n_real += 1

                        # Lock 1
                        if dollar_pnl >= 0:
                            ps.consec_losses = 0
                        else:
                            ps.consec_losses += 1
                            if ps.consec_losses >= 2:
                                ps.lock1 = True
                                ps.consec_losses = 0
                                ps.l1_paper_wins = 0
                    else:
                        if dollar_pnl >= 0 and ps.lock1:
                            ps.l1_paper_wins += 1
                        elif ps.lock1:
                            ps.l1_paper_wins = 0
                        if ps.lock1 and ps.l1_paper_wins >= 2:
                            ps.lock1 = False
                            ps.l1_paper_wins = 0

                    ps.in_position = False
                    ps.direction   = 0
                    ps.bars_held   = 0

            # Collect entry
            if not ps.in_position and not _no_trade(ps):
                if z < -entry_z and abs(z) < sl_z:
                    entry_signals.append((key, 1,  z, spread, p1v, p2v, arr.std()))
                elif z > entry_z and abs(z) < sl_z:
                    entry_signals.append((key, -1, z, spread, p1v, p2v, arr.std()))

        # Build universe vol snapshot for inverse-vol sizing
        all_spread_vols = {}
        for p1, p2 in valid_pairs:
            k   = f"{p1}/{p2}"
            buf = spread_buffers[k]
            if len(buf) >= 30:
                all_spread_vols[k] = np.std([v for _, v in buf])

        # Process entries
        for key, direction, z, spread, p1v, p2v, buf_std in entry_signals:
            ps       = pair_states[key]
            notional = _get_position_size(
                portfolio_value = port.portfolio_value,
                spread_vol      = buf_std,
                all_spread_vols = all_spread_vols,
            )
            if port.available_capital() < notional:
                continue
            ps.direction          = direction
            ps.entry_spread       = spread
            ps.entry_price1       = p1v
            ps.entry_price2       = p2v
            ps.entry_z_score      = z
            ps.in_position        = True
            ps.allocated_notional = notional
            ps.is_paper           = not _can_trade_real(ps)
            ps.bars_held          = 0
            if not ps.is_paper:
                port.deployed_capital += notional

        bar_returns.append(port.portfolio_value / prev_value - 1)
        prev_value = port.portfolio_value

        # Intermediate pruning check
        if prune_ts and dt >= prune_ts and prune_sharpe is None:
            prune_sharpe = _sharpe_from_returns(bar_returns, interval)

    # Final metrics
    if not bar_returns:
        return {'sharpe': -999.0, 'n_real': 0, 'prune_sharpe': prune_sharpe}

    sharpe = _sharpe_from_returns(bar_returns, interval)

    return {
        'sharpe':        sharpe,
        'n_real':        n_real,
        'prune_sharpe':  prune_sharpe,
    }


def _sharpe_from_returns(bar_returns: list, interval: str) -> float:
    rets = np.array(bar_returns)
    if len(rets) < 2 or rets.std() == 0:
        return -999.0
    bpy    = BARS_PER_YEAR.get(interval, 252)
    sharpe = rets.mean() / rets.std() * np.sqrt(bpy)
    return float(sharpe)


# ── Data loader (called once per interval, reused across all trials) ──────────

def load_interval_data(
    pairs:     List[Tuple[str, str]],
    interval:  str,
    file_path: str,
    train_end: str,
) -> Tuple[Dict, Dict, List]:
    """Load prices and calibrations once — reused across all Optuna trials."""
    print(f"  Loading data for {interval}...")

    price_data:   Dict = {}
    calibrations: Dict = {}
    valid_pairs         = []

    for p1, p2 in pairs:
        key = f"{p1}/{p2}"
        try:
            path1 = os.path.join(file_path, p1, f"{p1}_resampled_{interval}_returns.parquet")
            path2 = os.path.join(file_path, p2, f"{p2}_resampled_{interval}_returns.parquet")
            s1 = pd.read_parquet(path1, columns=['Close'])['Close']
            s2 = pd.read_parquet(path2, columns=['Close'])['Close']
            common = s1.index.intersection(s2.index)
            price_data[key] = (s1.loc[common], s2.loc[common])

            calib = get_rolling_half_life(
                p1, p2, interval, file_path,
                feature='Close', train_end=train_end,
                window_years=1, offset_months=1
            )
            calibrations[key] = calib
            valid_pairs.append((p1, p2))
        except Exception as e:
            print(f"    WARNING: {key} failed: {e}")

    print(f"  Loaded {len(valid_pairs)}/{len(pairs)} pairs")
    return price_data, calibrations, valid_pairs


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_objective(
    price_data:    Dict,
    calibrations:  Dict,
    valid_pairs:   List,
    interval:      str,
    train_start:   str,
    train_end:     str,
    initial_capital: float,
):
    prune_date = (
        pd.Timestamp(train_start) + pd.DateOffset(years=PRUNE_AFTER_YEARS)
    ).strftime('%Y-%m-%d')

    def objective(trial):
        # Sample parameters
        entry_z   = trial.suggest_float('entry_z',   1.5, 3.5, step=0.1)
        tp_z      = trial.suggest_float('tp_z',      0.0, 1.5, step=0.1)
        sl_z      = trial.suggest_float('sl_z',      3.0, 6.0, step=0.1)
        time_mult = trial.suggest_float('time_mult', 2.0, 6.0, step=0.5)

        # Hard constraints
        if tp_z >= entry_z:   return -999.0
        if sl_z <= entry_z:   return -999.0

        result = _run_backtest_fast(
            price_data      = price_data,
            calibrations    = calibrations,
            valid_pairs     = valid_pairs,
            interval        = interval,
            test_start      = train_start,
            test_end        = train_end,
            entry_z         = entry_z,
            tp_z            = tp_z,
            sl_z            = sl_z,
            time_mult       = time_mult,
            initial_capital = initial_capital,
            prune_date      = prune_date,
        )

        # Minimum trades filter
        if result['n_real'] < MIN_TRADES:
            return -999.0

        # Intermediate pruning
        prune_sharpe = result.get('prune_sharpe')
        if prune_sharpe is not None and prune_sharpe < PRUNE_SHARPE_THRESHOLD:
            raise optuna.exceptions.TrialPruned()

        return result['sharpe']

    return objective


# ── Coarse grid search for 1H and 30T ────────────────────────────────────────

def run_grid_search(
    price_data:    Dict,
    calibrations:  Dict,
    valid_pairs:   List,
    interval:      str,
    train_start:   str,
    train_end:     str,
    initial_capital: float,
) -> dict:
    """Coarse grid for shorter intervals."""
    grid = {
        'entry_z':   [1.8, 2.0, 2.2, 2.5, 2.8],
        'tp_z':      [0.3, 0.5, 0.8, 1.0],
        'sl_z':      [3.5, 4.0, 4.5, 5.0],
        'time_mult': [3.0, 4.0, 5.0],
    }

    best_sharpe = -999.0
    best_params = {}
    n_evaluated = 0

    total = (len(grid['entry_z']) * len(grid['tp_z']) *
             len(grid['sl_z'])    * len(grid['time_mult']))
    print(f"  Grid search: {total} combinations for {interval}")

    for entry_z in grid['entry_z']:
        for tp_z in grid['tp_z']:
            for sl_z in grid['sl_z']:
                for time_mult in grid['time_mult']:
                    if tp_z >= entry_z: continue
                    if sl_z <= entry_z: continue

                    result = _run_backtest_fast(
                        price_data      = price_data,
                        calibrations    = calibrations,
                        valid_pairs     = valid_pairs,
                        interval        = interval,
                        test_start      = train_start,
                        test_end        = train_end,
                        entry_z         = entry_z,
                        tp_z            = tp_z,
                        sl_z            = sl_z,
                        time_mult       = time_mult,
                        initial_capital = initial_capital,
                    )
                    n_evaluated += 1

                    if result['n_real'] >= MIN_TRADES and result['sharpe'] > best_sharpe:
                        best_sharpe = result['sharpe']
                        best_params = {
                            'entry_z': entry_z, 'tp_z': tp_z,
                            'sl_z': sl_z, 'time_mult': time_mult,
                        }

    print(f"  Grid: {n_evaluated} valid evals, best Sharpe={best_sharpe:.4f}")
    return {'params': best_params, 'sharpe': best_sharpe}


# ── Main optimizer ────────────────────────────────────────────────────────────

def optimize_z_thresholds(
    pairs:           List[Tuple[str, str]],
    intervals:       List[str],
    file_path:       str,
    train_start:     str   = '2001-01-01',
    train_end:       str   = '2014-12-31',
    test_start:      str   = '2015-01-01',
    test_end:        str   = '2023-12-31',
    initial_capital: float = 100_000.0,
    n_trials:        int   = 50,
    output_dir:      str   = './optimization_results',
) -> Dict[str, dict]:
    """
    Run z-threshold optimization per interval.

    Returns:
        results: dict keyed by interval with keys:
            'train_params'  — best params on training set
            'train_sharpe'  — Sharpe on training set
            'test_sharpe'   — Sharpe on test set (OOS)
            'test_result'   — full test metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    all_results: Dict[str, dict] = {}

    for interval in intervals:
        print(f"\n{'='*60}")
        print(f"Optimizing: {interval}")
        print('='*60)

        # Load data once for this interval
        price_data, calibrations, valid_pairs = load_interval_data(
            pairs, interval, file_path, train_end
        )
        if not valid_pairs:
            print(f"  No valid pairs for {interval}, skipping.")
            continue

        # ── Optimize ─────────────────────────────────────────────────────────
        if interval in BAYESIAN_INTERVALS:
            print(f"  Running Bayesian optimization ({n_trials} trials)...")
            sampler = optuna.samplers.TPESampler(seed=42)
            pruner  = optuna.pruners.MedianPruner(n_startup_trials=10)
            study   = optuna.create_study(
                direction = 'maximize',
                sampler   = sampler,
                pruner    = pruner,
            )
            objective = make_objective(
                price_data      = price_data,
                calibrations    = calibrations,
                valid_pairs     = valid_pairs,
                interval        = interval,
                train_start     = train_start,
                train_end       = train_end,
                initial_capital = initial_capital,
            )
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

            best_params  = study.best_params
            train_sharpe = study.best_value
            print(f"  Best train Sharpe: {train_sharpe:.4f}")
            print(f"  Best params: {best_params}")

            # Top-5 trials summary
            trials_df = study.trials_dataframe()
            valid_trials = trials_df[trials_df['value'] > -999].nlargest(5, 'value')
            print(f"\n  Top 5 trials:")
            print(valid_trials[['number', 'value',
                                 'params_entry_z', 'params_tp_z',
                                 'params_sl_z', 'params_time_mult']].to_string(index=False))

        else:  # Grid search for 1H, 30T
            grid_result  = run_grid_search(
                price_data, calibrations, valid_pairs,
                interval, train_start, train_end, initial_capital
            )
            best_params  = grid_result['params']
            train_sharpe = grid_result['sharpe']

        if not best_params:
            print(f"  WARNING: No valid params found for {interval}")
            continue

        # ── OOS test ──────────────────────────────────────────────────────────
        print(f"\n  Running OOS test ({test_start} → {test_end})...")

        # Reload calibrations with full train_end = test_end for OOS
        _, calib_oos, valid_pairs_oos = load_interval_data(
            pairs, interval, file_path, test_end
        )

        test_result = _run_backtest_fast(
            price_data      = price_data,
            calibrations    = calib_oos,
            valid_pairs     = valid_pairs_oos,
            interval        = interval,
            test_start      = test_start,
            test_end        = test_end,
            entry_z         = best_params['entry_z'],
            tp_z            = best_params['tp_z'],
            sl_z            = best_params['sl_z'],
            time_mult       = best_params['time_mult'],
            initial_capital = initial_capital,
        )

        test_sharpe = test_result['sharpe']
        degradation = (train_sharpe - test_sharpe) / abs(train_sharpe) * 100 if train_sharpe != 0 else np.nan

        print(f"  OOS Sharpe:   {test_sharpe:.4f}")
        print(f"  Train Sharpe: {train_sharpe:.4f}")
        print(f"  Degradation:  {degradation:.1f}%")

        interval_result = {
            'interval':      interval,
            'train_params':  best_params,
            'train_sharpe':  train_sharpe,
            'test_sharpe':   test_sharpe,
            'test_n_real':   test_result['n_real'],
            'degradation_pct': round(degradation, 2) if not np.isnan(degradation) else np.nan,
        }
        all_results[interval] = interval_result

        # Save per-interval result
        result_path = os.path.join(output_dir, f'opt_result_{interval}.csv')
        pd.DataFrame([interval_result]).to_csv(result_path, index=False)
        print(f"  Saved: {result_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("OPTIMIZATION SUMMARY")
    print('='*60)
    summary_rows = []
    for interval, res in all_results.items():
        p = res['train_params']
        summary_rows.append({
            'interval':    interval,
            'entry_z':     p.get('entry_z'),
            'tp_z':        p.get('tp_z'),
            'sl_z':        p.get('sl_z'),
            'time_mult':   p.get('time_mult'),
            'train_sharpe': round(res['train_sharpe'], 4),
            'test_sharpe':  round(res['test_sharpe'],  4),
            'degradation%': res.get('degradation_pct'),
            'oos_trades':   res['test_n_real'],
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    summary_path = os.path.join(output_dir, 'optimization_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved: {summary_path}")

    return all_results


# ── Entry point ───────────────────────────────────────────────────────────────

# if __name__ == '__main__':

#     FILE_PATH = '/Volumes/SEAGATE/FX_data/FX_histdata/'

#     PAIRS = [
#         ('USDCHF', 'EURCHF'), ('GBPUSD', 'EURCHF'), ('AUDJPY', 'EURCHF'),
#         ('EURAUD', 'EURCHF'), ('EURJPY', 'EURCHF'),  ('AUDUSD', 'EURCHF'),
#         ('CADJPY', 'EURCHF'), ('EURUSD', 'EURCHF'),  ('GBPJPY', 'EURCHF'),
#         ('NZDUSD', 'EURCHF'), ('USDCAD', 'EURCHF'),  ('USDJPY', 'EURCHF'),
#     ]

#     # Start with slower intervals first (Bayesian)
#     # then grid-search intervals
#     INTERVALS = ['1D', '6H', '3H', '1H', '30T']

#     results = optimize_z_thresholds(
#         pairs           = PAIRS,
#         intervals       = INTERVALS,
#         file_path       = FILE_PATH,
#         train_start     = '2001-01-01',
#         train_end       = '2014-12-31',
#         test_start      = '2015-01-01',
#         test_end        = '2023-12-31',
#         initial_capital = 100_000.0,
#         n_trials        = 50,
#         output_dir      = './optimization_results',
#     )