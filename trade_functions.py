import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import helper_functions as fx

def load_resampled_returns(pair, interval, file_path):
    df = pd.read_parquet(os.path.join(file_path, f"{pair}/", f"{pair}_resampled_{interval}_returns.parquet"))
    return df

def get_dynamic_hedge_ratio(series1, series2):
    model = LinearRegression()
    model.fit(series1.values.reshape(-1, 1), series2.values)
    beta = model.coef_[0]
    return beta

def get_mean(series, window=10):
    #handle NaNs, min_periods=window-2 only for now!
    return series.rolling(window=window, min_periods=window-2).mean()

def get_std(series, window=10):
    #handle NaNs, min_periods=window-2 only for now!
    return series.rolling(window=window, min_periods=window-2).std()

def get_spread(pair1_s, pair2_s, hedge_ratio):
    return pair2_s - hedge_ratio * pair1_s

# def get_half_life(hedge_ratio):
#     half_life = -np.log(2) / hedge_ratio
#     return half_life

def get_rolling_zscore(spread, half_life):
    window = int(np.ceil(half_life))
    window = max(window, 10)
    rolling_mean = get_mean(spread, window)
    rolling_std = get_std(spread, window)
    res = (spread - rolling_mean) / rolling_std
    return res

def get_portfolio_returns(signals, pair1_returns, pair2_returns, hedge_ratio_baseline):
    res = signals.shift(1)* (pair2_returns - hedge_ratio_baseline*pair1_returns)
    return res

def annualization_factor(index_s):
    freq = fx.get_dynamic_freq(index_s)
    freq_to_periods = {
        'T': 252 * 24 * 60,
        '5T': 252 * 24 * 12,
        '10T': 252 * 24 * 6,
        '15T': 252 * 24 * 4,
        '30T': 252 * 4 * 2,
        'H': 252 * 24,
        '3H': 252 * 8,
        '6H': 252 * 4,
        'D': 252
    }
    if freq in freq_to_periods:
        return np.sqrt(freq_to_periods[freq])
    else:
        raise ValueError(f"Unknown Frequency: {freq}")

# def backtest_pairs(pair1_train, pair2_train, pair1_test, pair2_test, hedge_ratio_baseline, feature='Return'):
#     # Double check that both pairs share same index and same interval
#     signals, z_score = generate_signals_with_exits(pair1_test[feature], pair2_test[feature], hedge_ratio_baseline, get_half_life(hedge_ratio_baseline))

#     pair1_returns = pair1_test[feature]
#     pair2_returns = pair2_test[feature]

#     portfolio_returns = get_portfolio_returns(signals, pair1_returns, pair2_returns, hedge_ratio_baseline)

#     cum_returns = (1 + portfolio_returns).cumprod()
#     sharpe = portfolio_returns.mean() / portfolio_returns.std() * annualization_factor(portfolio_returns.index)
#     max_dd = (cum_returns / cum_returns.cummax() - 1).min()

#     return {
#         'cumulative_returns': cum_returns,
#         'sharpe_ratio': sharpe,
#         'max_drawdown': max_dd,
#         'total_return': cum_returns.iloc[-1] - 1,
#         'signals': signals,
#         'z_score': z_score,
#         'window_size': int(np.ceil(get_half_life(hedge_ratio_baseline)))
#     }

# Risk Management points:
# Max 3-5 pairs open simultaneously
# Stop trading a pair if:
    # Cointegration p-value > 0.05 on rolling test
    # Half-life doubles or becomes negative
    # 3 consecutive stop-losses
# Position size: 2-5% portfolio risk per pair

def generate_signals_with_exits(pair1_s, pair2_s, hedge_ratio_baseline, half_life_baseline, entry_z=2.0, exit_z=0.5, stop_z=3.5):
    spread = get_spread(pair1_s, pair2_s, hedge_ratio_baseline)
    z_scores = get_rolling_zscore(spread, half_life_baseline)

    position = pd.Series(0, index=spread.index)
    # signals = []
    days_held = 0
    max_hold = int(half_life_baseline * 2)
    current_position = 0

    for i in range(1, len(z_scores)):
        z = z_scores.iloc[i]
        if pd.isna(z):
            position.iloc[i] = current_position
            continue

        if current_position == 0:
            if z > entry_z:
                current_position = -1
                days_held = 0
            elif z < -entry_z:
                current_position = 1
                days_held = 0
        else:
            days_held += 1
            if abs(z) < exit_z:
                current_position = 0
                days_held = 0
            elif abs(z) > stop_z:
                current_position = 0
                days_held = 0
            elif days_held >= max_hold:
                current_position = 0
                days_held = 0
        position.iloc[i] = current_position

    signals = pd.DataFrame({
        'spread': spread,
        'z_score': z_scores,
        'signal': position,
        'position': position.shift(1).fillna(0)
    }, index=z_scores.index)

    return signals

def apply_transaction_costs(returns, signals, cost_bps=0.0002):
    trades = signals.diff().abs()
    costs = trades * cost_bps
    net_returns = returns - costs
    return net_returns, costs

def rolling_backtest(pair1_full, pair2_full, feature, recal_window=60, test_start='2011-01-01', test_end='2016-12-31'):
    results = []

    for date in pd.date_range(test_start, test_end, freq='60D'):
        train_end = date
        train_start = date - pd.Timedelta(days=recal_window)

        p1_train = pair1_full.loc[train_start:train_end, feature]
        p2_train = pair2_full.loc[train_start:train_end, feature]

        hedge_ratio = fx.get_beta_lr(p1_train, p2_train)
        # half_life = get_half_life(p1_train, p2_train, hedge_ratio)

        # Calibrate on 60 days, test on next 60 days: NOT REALISTIC
        # Continuous recalibration with a rolling window
        test_end = date + pd.Timedelta(days=60)
        p1_test = pair1_full.loc[date:test_end, feature]
        p2_test = pair2_full.loc[date:test_end, feature]

        # signals = generate_signals_with_exits(p1_test, p2_test, hedge_ratio, half_life)
        # Calculate returns and apply transaction costs

def get_cagr_return(net_returns, total_return):
    start = net_returns.index.min()
    end = net_returns.index.max()
    time_span_years = (end - start).days / 365.25
    if time_span_years > 0:
        annual_return_cagr = (1 + total_return) ** (1 / time_span_years) - 1
    else:
        annual_return_cagr = 0
    return annual_return_cagr

def get_sharpe_ratio(net_returns, risk_free_rate=0.035):
    sqrt_periods = annualization_factor(net_returns.index)
    mean_return = net_returns.mean()
    std_return = net_returns.std()
    if std_return > 0:
        sharpe_ratio = ((mean_return - risk_free_rate) / std_return) * sqrt_periods
    else:
        sharpe_ratio = 0
    return sharpe_ratio

def get_annual_volatility(net_returns):
    sqrt_periods = annualization_factor(net_returns.index)
    annual_volatility = net_returns.std() * sqrt_periods
    return annual_volatility

def calculate_metrics(returns, signals):
    if returns is None:
        raise ValueError("Must calculate returns first!")
    
    net_returns = returns['net_returns'].dropna()
    cumulative_returns = returns['cumulative_returns']

    total_return = cumulative_returns.iloc[-1] - 1
    annual_return_cagr = get_cagr_return(net_returns, total_return)
    sharpe_ratio = get_sharpe_ratio(net_returns)
    annual_volatility = get_annual_volatility(net_returns)

    # Drawdown calculation
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns / running_max) - 1
    max_drawdown = drawdown.min()

    winning_trades = (net_returns > 0).sum()
    total_trades = (net_returns != 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    in_position = (signals['position'] != 0).sum()
    total_time = len(signals)
    percent_time_in_market = in_position / total_time if total_time > 0 else 0

    metrics = {
        'total_return': total_return,
        'annual_return_cagr': annual_return_cagr,
        'sharpe_ratio': sharpe_ratio,
        'annual_volatility': annual_volatility,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'percent_time_in_market': percent_time_in_market,
        'num_trades': total_trades
    }
    return metrics

def print_performance(pair1, pair2, signals, returns, metrics=None):
        """Print formatted performance summary"""
        if metrics is None:
            metrics = calculate_metrics(returns, signals)
        
        print("=" * 70)
        print(f"BACKTEST RESULTS: {pair1} / {pair2}")
        print("=" * 70)
        print(f"Period: {signals.index[0].date()} to {signals.index[-1].date()}")
        print(f"Trading Data length: {len(signals)}")
        print()
        print("RETURNS")
        print("-" * 70)
        print(f"Total Return: {metrics['total_return']*100:>10.2f}%")
        print(f"Annual Return: {metrics['annual_return_cagr']*100:>10.2f}%")
        print(f"Annual Volatility: {metrics['annual_volatility']*100:>10.2f}%")
        print()
        print("RISK METRICS")
        print("-" * 70)
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:>10.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']*100:>10.2f}%")
        print()
        print("TRADING ACTIVITY")
        print("-" * 70)
        print(f"Number of Trades: {metrics['num_trades']:>10.0f}")
        print(f"Win Rate (Daily): {metrics['win_rate']*100:>10.2f}%")
        print(f"Time in Market: {metrics['percent_time_in_market']*100:>10.1f}%")
        print(f"Total TC Cost: {returns['transaction_costs'].sum()*100:>10.4f}%")
        print("=" * 70)
        # print(f"Avg Daily Return: {metrics['avg_daily_return']*100:>10.4f}%")

def plot_results(pair1, pair2, signals, returns, figsize=(14, 12)):
    if returns is None or signals is None:
        raise ValueError("Must run backtest first")
        
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    
    # 1. Equity curve
    axes[0].plot(returns.index, returns['cumulative_returns'], 
                    label='Strategy', linewidth=2)
    axes[0].axhline(1, color='black', linestyle='--', alpha=0.3)
    axes[0].set_title(f'Equity Curve: {pair1}/{pair2}', 
                        fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Cumulative Returns')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 2. Z-score with positions
    axes[1].plot(signals.index, signals['z_score'], 
                    label='Z-Score', color='black', alpha=0.6)
    axes[1].axhline(0, color='gray', linestyle='-', alpha=0.3)
    axes[1].axhline(2, color='red', linestyle='--', alpha=0.5, label='Entry')
    axes[1].axhline(-2, color='red', linestyle='--', alpha=0.5)
    axes[1].axhline(0.5, color='green', linestyle='--', alpha=0.5, label='Exit')
    axes[1].axhline(-0.5, color='green', linestyle='--', alpha=0.5)
    
    # Highlight positions
    long_periods = signals['position'] > 0
    short_periods = signals['position'] < 0
    axes[1].fill_between(signals.index, -5, 5, 
                            where=long_periods, alpha=0.2, color='green', label='Long Spread')
    axes[1].fill_between(signals.index, -5, 5, 
                            where=short_periods, alpha=0.2, color='red', label='Short Spread')
    
    axes[1].set_ylabel('Z-Score')
    axes[1].set_ylim(-5, 5)
    axes[1].legend(loc='upper right')
    axes[1].grid(alpha=0.3)
    
    # 3. Spread
    axes[2].plot(signals.index, signals['spread'], 
                    label='Spread', color='blue')
    spread_mean = signals['spread'].mean()
    axes[2].axhline(spread_mean, color='black', linestyle='--', 
                    alpha=0.5, label='Mean')
    axes[2].set_ylabel('Spread Value')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    # 4. Drawdown
    cum_ret = returns['cumulative_returns']
    running_max = cum_ret.expanding().max()
    drawdown = (cum_ret - running_max) / running_max * 100
    
    axes[3].fill_between(drawdown.index, 0, drawdown, 
                        color='red', alpha=0.3)
    axes[3].plot(drawdown.index, drawdown, color='red', linewidth=1)
    axes[3].set_ylabel('Drawdown (%)')
    axes[3].set_xlabel('Date')
    axes[3].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

def calculate_returns(pair1_s, pair2_s, signals, hedge_ratio):
    if signals is None:
        raise ValueError("Must generate signals first!")
    
    pair1_ret = pair1_s.pct_change().fillna(0)
    pair2_ret = pair2_s.pct_change().fillna(0)
    spread_ret = pair2_ret - hedge_ratio * pair1_ret
    portfolio_ret = signals['position'] * spread_ret

    net_ret, costs = apply_transaction_costs(portfolio_ret, signals['position'], cost_bps=0.000)
    returns = pd.DataFrame({
        'gross_returns': portfolio_ret,
        'transaction_costs': costs,
        'net_returns': net_ret,
        'cumulative_returns': (1 + net_ret).cumprod()
    }, index=signals.index)

    trade_entries = signals['position'].diff().fillna(0)
    trades = trade_entries[trade_entries != 0]

    return returns, trades

def generate_signals_with_exits2(pair1_s, pair2_s, hedge_ratio_baseline, half_life_baseline, 
                                entry_z=2.0, exit_z=0.5, stop_z=3.5, max_consecutive_losses=5):
    spread = get_spread(pair1_s, pair2_s, hedge_ratio_baseline)
    z_scores = get_rolling_zscore(spread, half_life_baseline)

    position = pd.Series(0, index=spread.index)
    days_held = 0
    max_hold = int(half_life_baseline * 2)
    current_position = 0
    
    # Track consecutive losses
    consecutive_losses = 0
    last_trade_pnl = 0
    trading_suspended = False
    entry_spread = None

    for i in range(1, len(z_scores)):
        z = z_scores.iloc[i]
        if pd.isna(z):
            position.iloc[i] = current_position
            continue

        # Check if trading is suspended due to consecutive losses
        if trading_suspended:
            position.iloc[i] = 0
            current_position = 0
            continue

        if current_position == 0:
            # Entry logic
            if z > entry_z:
                current_position = -1
                days_held = 0
                entry_spread = spread.iloc[i]
            elif z < -entry_z:
                current_position = 1
                days_held = 0
                entry_spread = spread.iloc[i]
        else:
            days_held += 1
            
            # Exit logic
            exit_trade = False
            
            # Normal exit
            if abs(z) < exit_z:
                exit_trade = True
                exit_reason = 'normal'
            
            # Stop loss exit
            elif abs(z) > stop_z:
                exit_trade = True
                exit_reason = 'stop_loss'
            
            # Time-based exit
            elif days_held >= max_hold:
                exit_trade = True
                exit_reason = 'time_limit'
            
            if exit_trade:
                # Calculate trade P&L
                exit_spread = spread.iloc[i]
                trade_pnl = current_position * (exit_spread - entry_spread)
                
                # Track consecutive losses
                if exit_reason == 'stop_loss' or trade_pnl < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0
                
                # Suspend trading if 3 consecutive losses
                if consecutive_losses >= max_consecutive_losses:
                    trading_suspended = True
                    print(f"WARNING: Trading suspended at {spread.index[i]} after {consecutive_losses} consecutive losses")
                
                current_position = 0
                days_held = 0
                entry_spread = None
        
        position.iloc[i] = current_position

    signals = pd.DataFrame({
        'spread': spread,
        'z_score': z_scores,
        'signal': position,
        'position': position.shift(1).fillna(0),
        'trading_active': ~trading_suspended
    }, index=z_scores.index)

    return signals


def calculate_position_size(portfolio_value, pair_volatility, risk_per_trade=0.02, max_risk=0.05):
    """
    Calculate position size based on portfolio risk management
    
    Parameters:
    -----------
    portfolio_value : float
        Current portfolio value
    pair_volatility : float
        Expected volatility of the pair spread (annualized std dev)
    risk_per_trade : float
        Target risk per trade as fraction of portfolio (default 2%)
    max_risk : float
        Maximum risk per trade as fraction of portfolio (default 5%)
    
    Returns:
    --------
    position_size : float
        Position size as fraction of portfolio value
    """
    # Calculate position size to risk target percentage
    # Position Size = (Portfolio Value * Risk %) / (Volatility * Stop Distance)
    
    if pair_volatility <= 0:
        return 0
    
    # Simple volatility-based position sizing
    target_risk_dollars = portfolio_value * risk_per_trade
    position_size = target_risk_dollars / (pair_volatility * portfolio_value)
    
    # Cap at maximum risk
    max_position_size = max_risk
    position_size = min(position_size, max_position_size)
    
    return position_size


def calculate_returns_with_position_sizing(pair1_s, pair2_s, signals, hedge_ratio, 
                                           portfolio_value=100000, 
                                           risk_per_trade=0.02,
                                           max_risk=0.05):
    """
    Calculate returns with position sizing based on risk management
    """
    if signals is None:
        raise ValueError("Must generate signals first!")
    
    pair1_ret = pair1_s.pct_change().fillna(0)
    pair2_ret = pair2_s.pct_change().fillna(0)
    spread_ret = pair2_ret - hedge_ratio * pair1_ret
    
    # Calculate rolling volatility for position sizing
    lookback = 20
    spread_volatility = spread_ret.rolling(window=lookback, min_periods=10).std()
    
    # Annualize volatility
    annualization = annualization_factor(spread_ret.index)
    annual_volatility = spread_volatility * annualization
    
    # Calculate position size for each period
    position_sizes = []
    for i, vol in enumerate(annual_volatility):
        if pd.isna(vol) or vol == 0:
            position_sizes.append(risk_per_trade / 0.2)  # Default moderate size
        else:
            size = calculate_position_size(portfolio_value, vol, risk_per_trade, max_risk)
            position_sizes.append(size)
    
    position_size_series = pd.Series(position_sizes, index=signals.index)
    
    # Apply position sizing to returns
    portfolio_ret = signals['position'] * spread_ret * position_size_series

    net_ret, costs = apply_transaction_costs(portfolio_ret, signals['position'], cost_bps=0.0002)
    
    returns = pd.DataFrame({
        'gross_returns': portfolio_ret,
        'transaction_costs': costs,
        'net_returns': net_ret,
        'cumulative_returns': (1 + net_ret).cumprod(),
        'position_size': position_size_series,
        'spread_volatility': annual_volatility
    }, index=signals.index)

    trade_entries = signals['position'].diff().fillna(0)
    trades = trade_entries[trade_entries != 0]

    return returns, trades


def backtest_pairs_main2(pair1, pair2, interval, hedge_ratio_baseline, half_life_baseline, 
                       start, end, file_path, feature='Close',
                       use_position_sizing=False,
                       risk_per_trade=0.02,
                       max_risk=0.05,
                       portfolio_value=100000):
    """
    Main backtesting function with optional position sizing
    
    Parameters:
    -----------
    use_position_sizing : bool
        Whether to apply risk-based position sizing (default False for backward compatibility)
    risk_per_trade : float
        Target risk per trade (2-5% recommended)
    max_risk : float
        Maximum risk per trade cap
    """
    pair1_fulldf = load_resampled_returns(pair1, interval, file_path)
    pair2_fulldf = load_resampled_returns(pair2, interval, file_path)

    pair1_test_fulldf = fx.get_test_data(pair1_fulldf, start, end)
    pair2_test_fulldf = fx.get_test_data(pair2_fulldf, start, end)
    
    pair1_aligned_fulldf, pair2_aligned_fulldf = fx.common_duration(pair1_test_fulldf, pair2_test_fulldf)
    pair1_test_s = pd.Series(pair1_aligned_fulldf.loc[start:end, feature])
    pair2_test_s = pd.Series(pair2_aligned_fulldf.loc[start:end, feature])

    signals = generate_signals_with_exits2(pair1_test_s, pair2_test_s, 
                                         hedge_ratio_baseline, half_life_baseline)
    
    if use_position_sizing:
        returns, trades = calculate_returns_with_position_sizing(
            pair1_test_s, pair2_test_s, signals, hedge_ratio_baseline,
            portfolio_value=portfolio_value,
            risk_per_trade=risk_per_trade,
            max_risk=max_risk
        )
    else:
        returns, trades = calculate_returns(pair1_test_s, pair2_test_s, signals, hedge_ratio_baseline)
    
    print_performance(pair1, pair2, signals, returns)
    fig = plot_results(pair1, pair2, signals, returns)
    
    return signals, returns, trades, fig

def backtest_pairs_main(pair1, pair2, interval, hedge_ratio_baseline, half_life_baseline, start, end, file_path, feature='Close'):
    pair1_fulldf = load_resampled_returns(pair1, interval, file_path)
    pair2_fulldf = load_resampled_returns(pair2, interval, file_path)

    pair1_test_fulldf = fx.get_test_data(pair1_fulldf, start, end)
    pair2_test_fulldf = fx.get_test_data(pair2_fulldf, start, end)
    # Check if indices match
    pair1_aligned_fulldf, pair2_aligned_fulldf = fx.common_duration(pair1_test_fulldf, pair2_test_fulldf)
    pair1_test_s = pd.Series(pair1_aligned_fulldf.loc[start:end, feature])
    pair2_test_s = pd.Series(pair2_aligned_fulldf.loc[start:end, feature])

    signals = generate_signals_with_exits(pair1_test_s, pair2_test_s, hedge_ratio_baseline, half_life_baseline)
    returns, trades = calculate_returns(pair1_test_s, pair2_test_s, signals, hedge_ratio_baseline)
    print_performance(pair1, pair2, signals, returns)
    fig = plot_results(pair1, pair2, signals, returns)
    
    return signals, returns, trades, fig