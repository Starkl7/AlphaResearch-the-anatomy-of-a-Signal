import os
import numpy as np
import pandas as pd
import traceback
from collections import deque
from dataclasses import dataclass, field
from pprint import pprint
from src.analysis.halflife import get_rolling_half_life
from statsmodels.tsa.stattools import adfuller

# ── Kalman Filter for dynamic hedge ratio ────────────────────────────────────

class KalmanHedge:
    """
    1-D Kalman filter estimating the hedge ratio beta in:
        log(P2) = beta * log(P1) + alpha + noise

    State vector: [beta, alpha]  (2x1)
    Observation:  log(P2)        (scalar)
    Regressor:    [log(P1), 1]   (1x2)

    Parameters
    ----------
    delta : float
        Process noise scaling. Higher = faster adaptation, noisier estimate.
        Typical range: 1e-5 (slow) to 1e-3 (fast). Start with 1e-4.
    obs_noise : float
        Observation noise variance R. Auto-estimated if None.
    """

    def __init__(self, delta: float = 1e-4, obs_noise: float = None):
        self.delta = delta
        self.obs_noise = obs_noise

        # State: [beta, alpha]
        self._theta = np.zeros(2)           # state estimate
        self._P = np.eye(2) * 1.0           # state covariance
        self._R = obs_noise if obs_noise is not None else None
        self._e_var = 0.0                   # running innovation variance for R estimation
        self._n = 0                         # observations seen

        # Process noise
        self._Q = delta / (1 - delta) * np.eye(2)

    def update(self, p1: float, p2: float):
        """
        Feed one bar. Returns (beta, alpha, spread, innovation).
        spread = log(p2) - beta*log(p1) - alpha
        """
        x1 = np.log(p1)
        x2 = np.log(p2)
        F = np.array([x1, 1.0])            # observation matrix row

        # Predict
        P_pred = self._P + self._Q

        # Innovation
        y_hat = F @ self._theta
        e = x2 - y_hat

        # Adaptive R: use running variance of innovations (first 30 bars use prior)
        self._n += 1
        if self._R is None:
            self._e_var += (e ** 2 - self._e_var) / self._n
            R = max(self._e_var, 1e-8)
        else:
            R = self._R

        # Kalman gain
        S = F @ P_pred @ F + R
        K = P_pred @ F / S

        # Update
        self._theta = self._theta + K * e
        self._P = (np.eye(2) - np.outer(K, F)) @ P_pred

        beta  = self._theta[0]
        alpha = self._theta[1]
        spread = x2 - beta * x1 - alpha

        return beta, alpha, spread, e

    @property
    def alpha(self):
        return self._theta[1]


def rolling_eg_pval(spread_buffer: deque, min_bars: int = 120) -> float:
    arr = np.array([v for _, v in spread_buffer])
    if len(arr) < min_bars:
        return 1.0
    try:
        adf_res = adfuller(arr, maxlag=1, autolag=None, regression='c')
        return float(adf_res[1])
    except Exception:
        return 1.0

# ── Constants ─────────────────────────────────────────────────────────────────

VALID_HALF_LIVES = {
    '1T':  (5, 18000),
    '5T':  (5,  4000),
    '10T': (3,  2500),
    '15T': (3,   850),
    '30T': (3,   480),
    '1H':  (2,   450),
    '3H':  (2,   175),
    '6H':  (2,    95),
    '1D':  (2,    35),
}

TRANSACTION_COST_PCT = 0.00005      # 0.5 pips * 2 round-trip
MIN_POSITION_PCT     = 0.01
MAX_POSITION_PCT     = 0.05
VOL_TARGET_PCT       = 0.02

# Kalman tuning
KALMAN_DELTA         = {
    # '1T':  ,
    # '5T':  ,
    # '10T': ,
    # '15T': ,
    # '30T': ,
    '1H':  5e-5,
    '3H':  5e-4,
    '6H':  5e-3,
    '1D':  5e-2,
}       # optimal
KALMAN_WARMUP_BARS   = {
    '1T': 1000,
    '5T': 800,
    '10T': 600,
    '15T': 400,
    '30T': 200,
    '1H':  120,
    '3H':  80,
    '6H':  60,
    '1D':  30,
}          # heuristically

MIN_SPREAD_BARS      = {
    '1T': 43200,
    '5T':  8640,
    '10T': 4320,
    '15T': 2880,
    '30T': 1440,
    '1H':  720,
    '3H':  240,
    '6H':  120,
    '1D':  30,
}           # min spread history for z-score (1 month equiv)

EG_LOOKBACK_BARS = {            # trailing window for ADF test by interval
    '1T': 43200,
    '5T':  8640,
    '10T': 4320,
    '15T': 2880,
    '30T': 1440,
    '1H':  720,
    '3H':  240,
    '6H':  120,
    '1D':  30,
}

# Only recompute EG every N bars, cache result
EG_RECOMPUTE_EVERY = {
    '1D': 5,    # weekly
    '6H': 20,   # ~5 days
    '3H': 40,   # ~5 days
    '1H': 120,  # ~5 days
    '30T': 240,
    '15T': 480,
    '10T': 720,
    '5T': 1440,
    '1T': 7200,
}

# EG_PVALUE_THRESHOLD = 0.05      # cointegration gate
EG_PVALUE_THRESHOLD_BY_INTERVAL = {
    '1D': 0.05,
    '6H': 0.05,
    '3H': 0.05,
    '1H': 0.05,
    '30T': 0.05,
    '15T': 0.05,
    '10T': 0.05,
    '5T': 0.05,
    '1T': 0.05
}

def get_entry_z(interval=None):
    # return {'6H': 2.0, '1D': 2.5, '3H': 2.0}.get(interval, 2.5)
    return 2.5

def get_tp_z(interval=None):
    # return {'6H': 1.0, '1D': 0.5, '3H': 0.5}.get(interval, 0.5)
    return 0.5

def get_sl_z(interval=None):
    # return {'6H': 4.0, '1D': 4.0, '3H': 4.0}.get(interval, 4.0)
    return 4.5

def get_time_limit(half_life, interval=None):
    return int(half_life * 5)  # time limit = 5 half-lives, can be tuned


def get_position_size(portfolio_value, spread_vol, all_spread_vols):
    min_n = portfolio_value * MIN_POSITION_PCT
    max_n = portfolio_value * MAX_POSITION_PCT
    if spread_vol is None or spread_vol <= 0 or not all_spread_vols:
        return np.clip(portfolio_value * VOL_TARGET_PCT, min_n, max_n)
    inv_vols = {k: 1.0 / v for k, v in all_spread_vols.items() if v > 0}
    if not inv_vols:
        return np.clip(portfolio_value * VOL_TARGET_PCT, min_n, max_n)
    total_inv = sum(inv_vols.values())
    w = (1.0 / spread_vol) / total_inv
    notional = portfolio_value * VOL_TARGET_PCT * w * len(inv_vols)
    return float(np.clip(notional, min_n, max_n))


# ── State objects ─────────────────────────────────────────────────────────────

@dataclass
class PairState:
    # Kalman filter (created once, runs continuously)
    kalman: KalmanHedge
    kalman_bars: int = 0            # bars fed to Kalman (warmup counter)

    # Current Kalman estimates (updated every bar)
    hedge_ratio: float = np.nan
    kalman_alpha: float = np.nan

    # Half-life from rolling calibration (used for time-limit only)
    half_life: float = np.nan
    prev_half_life: float = np.nan

    # Lock flags
    lock1: bool = False             # 2 consecutive SL → paper
    lock2: bool = False             # half-life out of bounds
    lock3: bool = False             # hedge ratio change > 50%
    lock4: bool = False             # half-life change > 50%

    # Lock 1 counters
    consec_losses: int = 0
    l1_paper_wins: int = 0

    # Locks 2/3/4 recovery counters
    l2_months_ok: int = 0
    l3_months_ok: int = 0
    l4_months_ok: int = 0

    # Hedge ratio stability tracking (for Lock 3)
    prev_hedge_ratio: float = np.nan

    # Open trade
    in_position:       bool   = False
    direction:         int    = 0
    entry_bar:         int    = 0
    entry_spread:      float  = 0.0
    entry_price1:      float  = 0.0
    units1:            float  = 0.0
    units2:            float  = 0.0
    entry_price2:      float  = 0.0
    entry_z_score:     float  = 0.0
    entry_spread_std:  float  = 1.0
    entry_hedge_ratio: float  = np.nan   # hedge ratio locked at entry
    entry_kalman_alpha: float = np.nan
    is_paper:          bool   = False
    entry_time:        object = None
    allocated_notional: float = 0.0
    last_eg_pval:      float  = 1.0   # for monitoring cointegration at entry
    eg_pval_last_bar:   int  = -9999


@dataclass
class PortfolioState:
    initial_value:   float = 1_000_000.0
    portfolio_value: float = 1_000_000.0
    peak_value:      float = 1_000_000.0
    deployed_capital: float = 0.0

    def available_capital(self):
        return self.portfolio_value - self.deployed_capital

    def drawdown(self):
        if self.peak_value > 0:
            return (self.peak_value - self.portfolio_value) / self.peak_value
        return 0.0

    def update_peak(self):
        self.peak_value = max(self.peak_value, self.portfolio_value)


# ── State helpers ─────────────────────────────────────────────────────────────

def _state(ps):
    blocking   = ps.lock1
    monitoring = ps.lock2 or ps.lock3 or ps.lock4
    if blocking and monitoring: return 'SUSPENDED'
    if blocking:                return 'PAPER'
    if monitoring:              return 'MONITORING'
    return 'LIVE'

def _can_trade_real(ps):  return _state(ps) == 'LIVE'
def _no_trade(ps):        return _state(ps) == 'MONITORING'


# ── Monthly recalibration (half-life only — hedge ratio now from Kalman) ──────

def _recalibrate(ps: PairState, month_calib_row, interval: str):
    """
    Only updates half_life and stability locks.
    Hedge ratio is now driven by Kalman filter, not OLS.
    Lock 3 checks Kalman beta stability instead of OLS hedge ratio.
    """
    new_hl = float(month_calib_row['half_life'])
    new_hr = ps.kalman.beta   # use live Kalman beta for lock3 check

    ps.prev_half_life   = ps.half_life
    ps.prev_hedge_ratio = ps.hedge_ratio
    ps.half_life        = new_hl

    lo, hi = VALID_HALF_LIVES.get(interval, (3, 60))

    # Lock 2 — half-life out of bounds
    hl_valid = (not np.isnan(new_hl)) and (lo <= new_hl <= hi)
    if not hl_valid:
        ps.lock2 = True
        ps.l2_months_ok = 0
    elif ps.lock2:
        ps.l2_months_ok += 1
        if ps.l2_months_ok >= 2:
            ps.lock2 = False
            ps.l2_months_ok = 0

    # Lock 3 — Kalman beta change > 50% month-over-month
    if not np.isnan(ps.prev_hedge_ratio) and ps.prev_hedge_ratio != 0:
        hr_chg = abs((new_hr - ps.prev_hedge_ratio) / ps.prev_hedge_ratio)
        if hr_chg > 0.50:
            ps.lock3 = True
            ps.l3_months_ok = 0
        elif ps.lock3:
            ps.l3_months_ok += 1
            if ps.l3_months_ok >= 2:
                ps.lock3 = False
                ps.l3_months_ok = 0

    # Lock 4 — half-life change > 50% month-over-month
    if not np.isnan(ps.prev_half_life) and ps.prev_half_life != 0:
        hl_chg = abs((new_hl - ps.prev_half_life) / ps.prev_half_life)
        if hl_chg > 0.50:
            ps.lock4 = True
            ps.l4_months_ok = 0
        elif ps.lock4:
            ps.l4_months_ok += 1
            if ps.l4_months_ok >= 2:
                ps.lock4 = False
                ps.l4_months_ok = 0


# ── Exit logic ────────────────────────────────────────────────────────────────

def _check_exit(ps: PairState, z: float, bars_held: int, interval: str, spread_std: float = None, spread_change: float = None):
    """
    Check exit conditions with directional mean-reversion logic.
    
    Directional Take-Profit:
    - If entered LONG (entry_z < -2.5), exit when spread moved UP sufficiently
    - If entered SHORT (entry_z > 2.5), exit when spread moved DOWN sufficiently

    """
    tp_threshold = get_tp_z(interval)
    sl_threshold = get_sl_z(interval)
    
    # Normal regime - use directional z-score exits
    if ps.entry_z_score < 0:
        # Long spread
        if z >= -tp_threshold:
            return 'TP'
        if z < -sl_threshold:
            return 'SL'
    else:
        # Short spread
        if z <= tp_threshold:
            return 'TP'
        if z > sl_threshold:
            return 'SL'
    
    if not np.isnan(ps.half_life) and bars_held >= get_time_limit(ps.half_life, interval):
        return 'TIME'
    
    return None

def _check_entry(z: float, interval: str, spread_std: float = None):
    entry_threshold = get_entry_z(interval)
    outer_threshold = get_sl_z(interval)
    if spread_std is not None and spread_std < 1e-5:
        return None
    if z <= -entry_threshold and z > - outer_threshold:
        return 'LONG'
    elif z >= entry_threshold and z < outer_threshold:
        return 'SHORT'
    return None

def _on_exit(ps: PairState, port: PortfolioState, dollar_pnl: float):
    is_profitable = (dollar_pnl >= 0)

    if ps.is_paper:
        if is_profitable:
            if ps.lock1: ps.l1_paper_wins += 1
        else:
            if ps.lock1: ps.l1_paper_wins = 0
        if ps.lock1 and ps.l1_paper_wins >= 2:
            ps.lock1 = False
            ps.l1_paper_wins = 0
    else:
        if is_profitable:
            ps.consec_losses = 0
        else:
            ps.consec_losses += 1
            if ps.consec_losses >= 2:
                ps.lock1 = True
                ps.consec_losses = 0
                ps.l1_paper_wins = 0

        port.deployed_capital = max(0.0, port.deployed_capital - ps.allocated_notional)
        port.portfolio_value += dollar_pnl
        port.update_peak()


def _calc_pnl(ps: PairState, p1: float, p2: float, notional: float):
    """
    PnL using hedge ratio locked at entry (entry_hedge_ratio).
    
    For a spread trade where spread = log(p2) - hr*log(p1):
    - direction = 1: LONG spread (long p2, short hr*p1)
    - direction = -1: SHORT spread (short p2, long hr*p1)
    """
    # hr = ps.entry_hedge_ratio   # locked at entry

    # Position sizing based on notional
    gross_leg1 = ps.units1 * ps.entry_price1
    gross_leg2 = ps.units2 * ps.entry_price2
    # by gross exposure
    # N = notional / (abs(gross_leg1) + abs(gross_leg2))
    # beta_N = ps.direction * (-hr) * N

    # pnl_per_unit does not make sense because pair1 and pair2 have different number of units
    
    
    # Sanity check: verify P&L sign matches spread direction
    # Calculate log spread change
    # log_spread_exit = np.log(p2) - hr * np.log(p1)
    # log_spread_entry = np.log(ps.entry_price2) - hr * np.log(ps.entry_price1)
    # log_spread_change = log_spread_exit - log_spread_entry
    
    # # P&L sign should match: direction * log_spread_change
    # expected_positive = (ps.direction * log_spread_change > 0)
    # actual_positive = (pnl_per_unit > 0)
    
    # # If signs don't match, hedge ratio is likely invalid - correct P&L sign to match spread
    # # This is critical during volatile periods when hedge ratios become unreliable
    # if expected_positive != actual_positive and abs(log_spread_change) > 1e-6:
    #     # Force P&L sign to match spread direction
    #     pnl_per_unit = abs(pnl_per_unit) if expected_positive else -abs(pnl_per_unit)
    
    gross_dollar_pnl = ps.units1 * (p1 - ps.entry_price1) + ps.units2 * (p2 - ps.entry_price2)
    txn_cost = TRANSACTION_COST_PCT * 2 * notional

    return gross_dollar_pnl, txn_cost, gross_leg1, gross_leg2


# ── Main backtest ─────────────────────────────────────────────────────────────

def run_universe_backtest(
    pairs:           list,
    interval:        str,
    file_path:       str,
    test_start:      str  = '2013-01-01',
    test_end:        str  = '2018-12-31',
    initial_capital: float = 1_000_000.0,
    max_position_pct: float = 0.10,
    verbosity:       int  = 1,
    kalman_delta:      float = None,
) -> tuple:
    
    if kalman_delta is None:
        delta = KALMAN_DELTA.get(interval, 1e-6)
    else:
        delta = kalman_delta

    # ── 1. Load prices ────────────────────────────────────────────────────────
    def _load(pair):
        path = os.path.join(file_path, f"{pair}/", f"{pair}_resampled_{interval}_returns.parquet")
        return pd.read_parquet(path, columns=['Close'])

    if verbosity >= 1:
        print(f"Universe backtest: {len(pairs)} @ {interval}")
        print(f"Period: {test_start} - {test_end}")

    price_data  = {}
    valid_pairs = []
    for p1, p2 in pairs:
        try:
            s1 = _load(p1)['Close']
            s2 = _load(p2)['Close']
            common = s1.index.intersection(s2.index)
            price_data[f"{p1}/{p2}"] = (s1.loc[common], s2.loc[common])
            valid_pairs.append((p1, p2))
        except Exception as e:
            if verbosity >= 1:
                print(f"WARNING: Could not load {p1}/{p2}: {e}")

    if verbosity >= 1:
        print(f"Loaded {len(valid_pairs)}/{len(pairs)} successfully")

    # ── 2. Calibrations (half-life for time-limit + lock logic only) ──────────
    calibrations  = {}
    spread_buffers = {}     # rolling spread history for z-score
    pair_states   = {}

    # Get interval-specific delta for Kalman filter
    if kalman_delta is None:
        delta = KALMAN_DELTA.get(interval, 1e-4)  # fallback to 1e-4 if interval not in dict
    else:
        delta = kalman_delta

    for p1, p2 in valid_pairs:
        key = f"{p1}/{p2}"
        try:
            calib = get_rolling_half_life(
                p1, p2, interval, file_path,
                feature='Close', train_end=test_end,
                window_years=1, offset_months=1
            )
            calibrations[key]   = calib
            pair_states[key]    = PairState(kalman = KalmanHedge(delta=delta))
            spread_buffers[key] = deque()

            # Pre-warm Kalman on pre-test data so filter is stabilised at test_start
            s1, s2 = price_data[key]
            preseed_start = pd.Timestamp(test_start) - pd.DateOffset(years=2)
            preseed_mask  = (s1.index >= preseed_start) & (s1.index < pd.Timestamp(test_start))
            kf = pair_states[key].kalman
            for dt_ps in s1[preseed_mask].index:
                try:
                    v1 = float(s1.loc[dt_ps])
                    v2 = float(s2.loc[dt_ps])
                    beta, alpha, spread, _ = kf.update(v1, v2)
                    pair_states[key].kalman_bars += 1
                    spread_buffers[key].append((dt_ps, spread))
                except Exception:
                    pass

            pair_states[key].hedge_ratio  = kf.beta
            pair_states[key].kalman_alpha = kf.alpha

        except Exception as e:
            if verbosity >= 1:
                print(f"WARNING: Calibration failed for {key}: {e}")
            valid_pairs.remove((p1, p2))

    # ── 3. Align to common time index ─────────────────────────────────────────
    all_indices = []
    for p1, p2 in valid_pairs:
        key = f"{p1}/{p2}"
        s1, _ = price_data[key]
        mask = (s1.index >= pd.Timestamp(test_start)) & (s1.index <= pd.Timestamp(test_end))
        all_indices.append(s1[mask].index)

    if not all_indices:
        raise ValueError("No valid pairs to backtest!")

    time_index = all_indices[0]
    for idx in all_indices[1:]:
        time_index = time_index.union(idx)
    time_index = time_index.sort_values()

    # ── 4. Initialise portfolio & accumulators ────────────────────────────────
    port = PortfolioState(
        initial_value=initial_capital,
        portfolio_value=initial_capital,
        peak_value=initial_capital,
    )

    n_real = n_paper = real_wins = real_losses = 0
    equity_records = []
    trade_records  = []
    current_months = {f"{p1}/{p2}": None for p1, p2 in valid_pairs}
    bars_held      = {f"{p1}/{p2}": 0    for p1, p2 in valid_pairs}

    # ── 5. Bar-by-bar loop ────────────────────────────────────────────────────
    for dt in time_index:

        entry_signals   = []
        all_spread_vols = {}

        for p1, p2 in valid_pairs:
            key = f"{p1}/{p2}"
            ps  = pair_states[key]
            s1, s2 = price_data[key]

            if dt not in s1.index:
                continue

            p1v = float(s1.loc[dt])
            p2v = float(s2.loc[dt])

            # ── Kalman update (every bar, always) ────────────────────────
            try:
                beta, alpha, kalman_spread, _ = ps.kalman.update(p1v, p2v)
            except Exception:
                continue
            ps.kalman_bars    += 1
            ps.hedge_ratio     = beta
            ps.kalman_alpha    = alpha

            # ── Monthly recalibration (half-life + stability locks) ───────
            month_key = dt.to_period('M')
            if month_key != current_months[key]:
                current_months[key] = month_key
                calib = calibrations[key]
                month_calib = calib[calib.index.to_period('M') == month_key]
                if len(month_calib) > 0:
                    row = month_calib.iloc[0]
                    if not np.isnan(float(row['hedge_ratio'])):
                        _recalibrate(ps, row, interval)

            if ps.kalman_bars < KALMAN_WARMUP_BARS.get(interval, 60):
                continue
            if np.isnan(ps.hedge_ratio):
                continue

            # ── Rolling spread buffer & z-score ──────────────────────────
            spread_buffers[key].append((dt, kalman_spread))
            cutoff = dt - pd.DateOffset(years=1)
            buf = spread_buffers[key]
            while buf and buf[0][0] < cutoff:
                buf.popleft()

            arr = np.array([v for _, v in buf])
            spread_std = arr.std()
            all_spread_vols[key] = spread_std

            if len(arr) < MIN_SPREAD_BARS.get(interval, 252):
                continue
            if spread_std < 1e-10:
                continue

            # Apply safety floor to prevent division by near-zero during volatile periods
            spread_std_safe = max(spread_std, 1e-6)
            z = (kalman_spread - arr.mean()) / spread_std_safe
            # Don't cap for entry decisions - we want to detect true extreme moves

            # ── Exit open position ────────────────────────────────────────
            if ps.in_position:
                bars_held[key] += 1

                # Calculate spread using ENTRY hedge ratio AND alpha (consistent with entry)
                current_spread_entry_hr = (
                    np.log(p2v)
                    - ps.entry_hedge_ratio * np.log(p1v)
                    - ps.entry_kalman_alpha  # Include alpha for consistency
                )
                
                # Spread change (alpha cancels out anyway)
                spread_change = current_spread_entry_hr - ps.entry_spread
                
                # Calculate z based on spread change from entry
                # Use max of entry_spread_std and current spread_std_safe for stability
                spread_std_for_exit = max(ps.entry_spread_std, spread_std_safe)
                z_exit_raw = spread_change / spread_std_for_exit
                
                # Also calculate z using current Kalman for context
                z_current_kalman = z
                
                # Use the spread change-based z for exit decision
                # Pass spread_change for absolute threshold logic during instability
                exit_type = _check_exit(
                    ps, z_exit_raw, bars_held[key], interval, 
                    spread_std=spread_std, spread_change=spread_change
                )
                if exit_type:
                    # Cap z for logging purposes only
                    z_exit_logged = np.clip(z_exit_raw, -50, 50)
                    
                    notional = ps.allocated_notional
                    dollar_gross, txn_cost, gross_leg1, gross_leg2 = _calc_pnl(ps, p1v, p2v, notional)
                    dollar_pnl = (dollar_gross - txn_cost) if not ps.is_paper else 0.0

                    pnl_leg2 = gross_leg2
                    pnl_leg1 = gross_leg1

                    trade_rec = {
                        'pair':                   key,
                        'entry_time':             ps.entry_time,
                        'exit_time':              dt,
                        'interval':               interval,
                        'entry_price1':           ps.entry_price1,
                        'entry_price2':           ps.entry_price2,
                        'exit_price1':            p1v,
                        'exit_price2':            p2v,
                        'entry_spread':           ps.entry_spread,
                        'exit_spread':            current_spread_entry_hr,  # Use consistent hedge ratio
                        'spread_change':          spread_change,            # Consistent spread change
                        'entry_z':                ps.entry_z_score,
                        'exit_z':                 z_exit_logged,
                        'signal':                 ps.direction,
                        'hedge_ratio_at_entry':   ps.entry_hedge_ratio,
                        'hedge_ratio_current':    ps.hedge_ratio,
                        'P1_units':               ps.units1,
                        'PnL_leg1':               pnl_leg1,
                        'P2_units':               ps.units2,
                        'PnL_leg2':               pnl_leg2,
                        'dollar_pnl_gross':       dollar_gross,
                        'txn_cost':               txn_cost,
                        'dollar_pnl_net':         dollar_pnl,
                        'half_life':              ps.half_life,
                        'bars_held':              bars_held[key],
                        'notional':               notional,
                        'eg_pval_at_entry':       ps.last_eg_pval,
                        'exit_type':              exit_type,
                        'spread_std':             spread_std,  # Track volatility at exit
                        'is_paper':               ps.is_paper,
                        'state':                  _state(ps),
                        'portfolio_value_before': port.portfolio_value,
                    }

                    was_paper = ps.is_paper
                    _on_exit(ps, port, dollar_pnl)
                    trade_rec['portfolio_value_after'] = port.portfolio_value
                    trade_records.append(trade_rec)

                    if was_paper:
                        n_paper += 1
                    else:
                        n_real += 1
                        if dollar_pnl >= 0: real_wins   += 1
                        else:               real_losses += 1

                    ps.in_position = False
                    ps.direction   = 0
                    bars_held[key] = 0

            # ── Collect entry signal ──────────────────────────────────────
            if not ps.in_position and not _no_trade(ps):

                # ── Cointegration gate ────────────────────────────────────────
                if interval != '1D':  # skip EG for daily, too few bars to be meaningful
                    recompute_every = EG_RECOMPUTE_EVERY.get(interval, 20)
                    bar_idx = equity_records.__len__()  # current bar count as proxy

                    if (bar_idx - ps.eg_pval_last_bar) >= recompute_every:
                        lookback = EG_LOOKBACK_BARS.get(interval, 500)
                        recent_buf = deque(list(spread_buffers[key])[-lookback:])
                        ps.last_eg_pval = rolling_eg_pval(recent_buf, min_bars=lookback // 2)
                        ps.eg_pval_last_bar = bar_idx

                    if ps.last_eg_pval > EG_PVALUE_THRESHOLD_BY_INTERVAL.get(interval, 0.05):
                        continue
                # ─────────────────────────────────────────────────────────────
                sig = _check_entry(z, interval, spread_std=spread_std)
                if sig is not None:
                    if sig == 'LONG':
                        entry_signals.append((key, 1,  z, kalman_spread, p1v, p2v, spread_std))
                    elif sig == 'SHORT':
                        entry_signals.append((key, -1, z, kalman_spread, p1v, p2v, spread_std))

        # ── Process entry signals ─────────────────────────────────────────────
        for key, direction, z, spread, p1v, p2v, buf_std in entry_signals:
            ps       = pair_states[key]
            notional = get_position_size(
                portfolio_value=port.portfolio_value,
                spread_vol=buf_std,
                all_spread_vols=all_spread_vols,
            )
            if port.available_capital() < notional:
                continue

            hr_raw = ps.hedge_ratio * (p2v / p1v)
            denom = abs(hr_raw) * p1v + p2v
            if denom <= 0:
                continue
            u2_abs = notional / denom
            u1_abs = abs(hr_raw) * u2_abs
            if direction == 1:
                units2 = u2_abs
                units1 = -np.sign(hr_raw) * u1_abs
            else:
                units2 = -u2_abs
                units1 = np.sign(hr_raw) * u1_abs

            ps.entry_time         = dt
            ps.direction          = direction
            ps.entry_spread       = spread
            ps.entry_price1       = p1v
            ps.units1             = units1
            ps.units2             = units2
            ps.entry_price2       = p2v
            ps.entry_z_score      = z
            ps.entry_spread_std   = max(buf_std, 1e-8)
            ps.entry_bar          = 0
            ps.in_position        = True
            ps.allocated_notional = notional
            ps.entry_hedge_ratio  = ps.hedge_ratio        # lock current Kalman beta
            ps.entry_kalman_alpha = ps.kalman_alpha        # lock current Kalman alpha
            ps.is_paper           = not _can_trade_real(ps)
            bars_held[key]        = 0

            if not ps.is_paper:
                port.deployed_capital += notional

        # ── Equity snapshot ───────────────────────────────────────────────────
        n_open = sum(
            1 for p1, p2 in valid_pairs
            if pair_states[f"{p1}/{p2}"].in_position
            and not pair_states[f"{p1}/{p2}"].is_paper
        )
        equity_records.append({
            'datetime':         dt,
            'portfolio_value':  port.portfolio_value,
            'deployed_capital': port.deployed_capital,
            'drawdown':         port.drawdown(),
            'n_open_trades':    n_open,
        })

    # ── 6. Build output DataFrames ────────────────────────────────────────────
    equity_curve = pd.DataFrame(equity_records).set_index('datetime')
    trades_df    = pd.DataFrame(trade_records)
    if not trades_df.empty:
        trades_df = trades_df.set_index('entry_time')

    metrics = _compute_portfolio_metrics(
        equity_curve, trades_df, port, initial_capital,
        n_real, n_paper, real_wins, real_losses,
        test_start, test_end, interval, delta
    )

    if verbosity >= 1:
        print(f"\n  Portfolio Metrics:")
        pprint(metrics)

    return equity_curve, trades_df, metrics


# ── Portfolio metrics ─────────────────────────────────────────────────────────

def _compute_portfolio_metrics(
    equity_curve, trades_df, port, initial_capital,
    n_real, n_paper, real_wins, real_losses,
    test_start, test_end, interval, delta=None
):
    final_value  = port.portfolio_value
    total_return = (final_value - initial_capital) / initial_capital

    t_start = pd.Timestamp(test_start)
    t_end   = pd.Timestamp(test_end)
    years   = (t_end - t_start).days / 365.25
    cagr    = (final_value / initial_capital) ** (1 / years) - 1 if years > 0 else np.nan

    max_dd = equity_curve['drawdown'].max() if not equity_curve.empty else np.nan

    bars_per_year = {
        '1T': 525_600, '5T': 105_120, '10T': 52_560, '15T': 35_040,
        '30T': 17_520, '1H': 8_760,   '3H': 2_920,   '6H': 1_460, '1D': 252,
    }
    bpy = bars_per_year.get(interval, 252)
    pv  = equity_curve['portfolio_value']
    bar_returns = pv.pct_change().dropna()
    sharpe = (
        (bar_returns.mean() / bar_returns.std()) * np.sqrt(bpy)
        if len(bar_returns) > 1 and bar_returns.std() > 0 else np.nan
    )

    win_rate = avg_bars = turnover = avg_pnl = total_txn = np.nan
    if not trades_df.empty:
        real_trades = trades_df[~trades_df['is_paper']]
        if not real_trades.empty:
            win_rate  = real_wins / n_real if n_real > 0 else np.nan
            avg_bars  = real_trades['bars_held'].mean()
            total_txn = real_trades['txn_cost'].sum()
            avg_pnl   = real_trades['dollar_pnl_net'].mean()
            avg_port  = equity_curve['portfolio_value'].mean()
            total_notional = real_trades['notional'].sum() * 2
            turnover  = (total_notional / avg_port) / years if avg_port > 0 else np.nan

    return {
        'initial_capital':        round(initial_capital, 2),
        'final_portfolio_value':  round(final_value, 2),
        'total_return_pct':       round(total_return * 100, 4),
        'cagr_pct':               round(cagr * 100, 4) if not np.isnan(cagr) else np.nan,
        'max_drawdown_pct':       round(max_dd * 100, 4) if not np.isnan(max_dd) else np.nan,
        'sharpe_ratio':           round(sharpe, 4) if not np.isnan(sharpe) else np.nan,
        'n_real_trades':          n_real,
        'n_paper_trades':         n_paper,
        'real_win_rate':          round(win_rate, 4) if not np.isnan(win_rate) else np.nan,
        'real_wins':              real_wins,
        'real_losses':            real_losses,
        'avg_trade_pnl_net':      round(avg_pnl, 4) if not np.isnan(avg_pnl) else np.nan,
        'avg_bars_held':          round(avg_bars, 2) if not np.isnan(avg_bars) else np.nan,
        'total_txn_costs':        round(total_txn, 2) if not np.isnan(total_txn) else np.nan,
        'portfolio_turnover':     round(turnover, 4) if not np.isnan(turnover) else np.nan,
        'n_pairs':                trades_df['pair'].nunique() if not trades_df.empty else 0,
        'test_years':             round(years, 2),
        'n_bars':                 len(equity_curve),
        'kalman_delta':           delta,
        'kalman_warmup_bars':     KALMAN_WARMUP_BARS.get(interval, 60),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def backtest_universe_main(
    pairs:           list,
    interval:        str,
    test_start:      str,
    test_end:        str,
    file_path:       str   = '/Volumes/SEAGATE/FX_data/FX_histdata/',
    output_dir:      str   = './backtest_results_kalman',
    initial_capital: float = 1_000_000.0,
    max_position_pct: float = 0.10,
    verbosity:       int   = 1,
    kalman_delta:      float = None,
) -> tuple:

    os.makedirs(output_dir, exist_ok=True)

    for ts in [test_start, test_end]:
        if not isinstance(ts, str):
            raise TypeError(f"test_start/test_end must be strings, got {type(ts)}")
    if kalman_delta is None:
        kalman_delta = KALMAN_DELTA.get(interval, 1e-6)

    equity_curve, trades_df, metrics = run_universe_backtest(
        pairs=pairs,
        interval=interval,
        file_path=file_path,
        test_start=test_start,
        test_end=test_end,
        initial_capital=initial_capital,
        max_position_pct=max_position_pct,
        verbosity=verbosity,
        kalman_delta=kalman_delta
    )

    equity_path = os.path.join(output_dir, f'equity_curve_{interval}.csv')
    trades_path = os.path.join(output_dir, f'trades_{interval}.csv')
    equity_curve.to_csv(equity_path)
    if not trades_df.empty:
        trades_df.to_csv(trades_path)

    if verbosity >= 1:
        print(f"\n  Saved: {equity_path}")
        print(f"  Saved: {trades_path}")

    return equity_curve, trades_df, metrics


# ── Example usage ─────────────────────────────────────────────────────────────

# if __name__ == '__main__':
#     PAIRS = [
#         ('USDCHF', 'EURCHF'),
#         ('GBPUSD', 'EURCHF'),
#         ('AUDJPY', 'EURCHF'),
#         ('EURAUD', 'EURCHF'),
#         ('EURJPY', 'EURCHF'),
#     ]
#
#     equity_curve, trades_df, metrics = backtest_universe_main(
#         pairs      = PAIRS,
#         interval   = '6H',
#         test_start = '2015-01-01',
#         test_end   = '2022-12-31',
#         verbosity  = 2,
#     )