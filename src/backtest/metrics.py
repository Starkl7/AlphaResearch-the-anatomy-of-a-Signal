import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.loader import get_dynamic_freq


def annualization_factor(index_s):
    freq = get_dynamic_freq(index_s)
    freq_to_periods = {
        'T': 252 * 24 * 60,
        '5T': 252 * 24 * 12,
        '10T': 252 * 24 * 6,
        '15T': 252 * 24 * 4,
        '30T': 252 * 4 * 2,
        'H': 252 * 24,
        '3H': 252 * 8,
        '6H': 252 * 4,
        'D': 252,
    }
    if freq in freq_to_periods:
        return np.sqrt(freq_to_periods[freq])
    else:
        raise ValueError(f"Unknown Frequency: {freq}")


def get_portfolio_returns(signals, pair1_returns, pair2_returns, hedge_ratio_baseline):
    return signals.shift(1) * (pair2_returns - hedge_ratio_baseline * pair1_returns)


def apply_transaction_costs(returns, signals, cost_bps=0.0002):
    trades = signals.diff().abs()
    costs = trades * cost_bps
    net_returns = returns - costs
    return net_returns, costs


def get_cagr_return(net_returns, total_return):
    start = net_returns.index.min()
    end = net_returns.index.max()
    time_span_years = (end - start).days / 365.25
    if time_span_years > 0:
        return (1 + total_return) ** (1 / time_span_years) - 1
    return 0


def get_sharpe_ratio(net_returns, risk_free_rate=0.035):
    sqrt_periods = annualization_factor(net_returns.index)
    mean_return = net_returns.mean()
    std_return = net_returns.std()
    if std_return > 0:
        return ((mean_return - risk_free_rate) / std_return) * sqrt_periods
    return 0


def get_annual_volatility(net_returns):
    return net_returns.std() * annualization_factor(net_returns.index)


def calculate_metrics(returns, signals):
    if returns is None:
        raise ValueError("Must calculate returns first!")

    net_returns = returns['net_returns'].dropna()
    cumulative_returns = returns['cumulative_returns']

    total_return = cumulative_returns.iloc[-1] - 1
    annual_return_cagr = get_cagr_return(net_returns, total_return)
    sharpe_ratio = get_sharpe_ratio(net_returns)
    annual_volatility = get_annual_volatility(net_returns)

    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns / running_max) - 1
    max_drawdown = drawdown.min()

    winning_trades = (net_returns > 0).sum()
    total_trades = (net_returns != 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    in_position = (signals['position'] != 0).sum()
    total_time = len(signals)
    percent_time_in_market = in_position / total_time if total_time > 0 else 0

    return {
        'total_return': total_return,
        'annual_return_cagr': annual_return_cagr,
        'sharpe_ratio': sharpe_ratio,
        'annual_volatility': annual_volatility,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'percent_time_in_market': percent_time_in_market,
        'num_trades': total_trades,
    }


def print_performance(pair1, pair2, signals, returns, metrics=None):
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
    print(f"Total Return:          {metrics['total_return']*100:>10.2f}%")
    print(f"Annual Return:         {metrics['annual_return_cagr']*100:>10.2f}%")
    print(f"Annual Volatility:     {metrics['annual_volatility']*100:>10.2f}%")
    print()
    print("RISK METRICS")
    print("-" * 70)
    print(f"Sharpe Ratio:          {metrics['sharpe_ratio']:>10.2f}")
    print(f"Max Drawdown:          {metrics['max_drawdown']*100:>10.2f}%")
    print()
    print("TRADING ACTIVITY")
    print("-" * 70)
    print(f"Number of Trades:      {metrics['num_trades']:>10.0f}")
    print(f"Win Rate (Daily):      {metrics['win_rate']*100:>10.2f}%")
    print(f"Time in Market:        {metrics['percent_time_in_market']*100:>10.1f}%")
    print(f"Total TC Cost:         {returns['transaction_costs'].sum()*100:>10.4f}%")
    print("=" * 70)


def plot_results(pair1, pair2, signals, returns, figsize=(14, 12)):
    if returns is None or signals is None:
        raise ValueError("Must run backtest first")

    fig, axes = plt.subplots(4, 1, figsize=figsize)

    axes[0].plot(returns.index, returns['cumulative_returns'], label='Strategy', linewidth=2)
    axes[0].axhline(1, color='black', linestyle='--', alpha=0.3)
    axes[0].set_title(f'Equity Curve: {pair1}/{pair2}', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Cumulative Returns')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(signals.index, signals['z_score'], label='Z-Score', color='black', alpha=0.6)
    axes[1].axhline(0, color='gray', linestyle='-', alpha=0.3)
    axes[1].axhline(2, color='red', linestyle='--', alpha=0.5, label='Entry')
    axes[1].axhline(-2, color='red', linestyle='--', alpha=0.5)
    axes[1].axhline(0.5, color='green', linestyle='--', alpha=0.5, label='Exit')
    axes[1].axhline(-0.5, color='green', linestyle='--', alpha=0.5)
    long_periods = signals['position'] > 0
    short_periods = signals['position'] < 0
    axes[1].fill_between(signals.index, -5, 5, where=long_periods, alpha=0.2, color='green', label='Long Spread')
    axes[1].fill_between(signals.index, -5, 5, where=short_periods, alpha=0.2, color='red', label='Short Spread')
    axes[1].set_ylabel('Z-Score')
    axes[1].set_ylim(-5, 5)
    axes[1].legend(loc='upper right')
    axes[1].grid(alpha=0.3)

    axes[2].plot(signals.index, signals['spread'], label='Spread', color='blue')
    axes[2].axhline(signals['spread'].mean(), color='black', linestyle='--', alpha=0.5, label='Mean')
    axes[2].set_ylabel('Spread Value')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    cum_ret = returns['cumulative_returns']
    running_max = cum_ret.expanding().max()
    drawdown = (cum_ret - running_max) / running_max * 100
    axes[3].fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.3)
    axes[3].plot(drawdown.index, drawdown, color='red', linewidth=1)
    axes[3].set_ylabel('Drawdown (%)')
    axes[3].set_xlabel('Date')
    axes[3].grid(alpha=0.3)

    plt.tight_layout()
    return fig


def calculate_position_size(portfolio_value, pair_volatility, risk_per_trade=0.02, max_risk=0.05):
    if pair_volatility <= 0:
        return 0
    target_risk_dollars = portfolio_value * risk_per_trade
    position_size = target_risk_dollars / (pair_volatility * portfolio_value)
    return min(position_size, max_risk)


def calculate_returns_with_position_sizing(
    pair1_s, pair2_s, signals, hedge_ratio,
    portfolio_value=100000, risk_per_trade=0.02, max_risk=0.05,
):
    if signals is None:
        raise ValueError("Must generate signals first!")

    pair1_ret = pair1_s.pct_change().fillna(0)
    pair2_ret = pair2_s.pct_change().fillna(0)
    spread_ret = pair2_ret - hedge_ratio * pair1_ret

    lookback = 20
    spread_volatility = spread_ret.rolling(window=lookback, min_periods=10).std()
    annual_volatility = spread_volatility * annualization_factor(spread_ret.index)

    position_sizes = []
    for vol in annual_volatility:
        if pd.isna(vol) or vol == 0:
            position_sizes.append(risk_per_trade / 0.2)
        else:
            position_sizes.append(calculate_position_size(portfolio_value, vol, risk_per_trade, max_risk))

    position_size_series = pd.Series(position_sizes, index=signals.index)
    portfolio_ret = signals['position'] * spread_ret * position_size_series
    net_ret, costs = apply_transaction_costs(portfolio_ret, signals['position'], cost_bps=0.0002)

    returns = pd.DataFrame({
        'gross_returns': portfolio_ret,
        'transaction_costs': costs,
        'net_returns': net_ret,
        'cumulative_returns': (1 + net_ret).cumprod(),
        'position_size': position_size_series,
        'spread_volatility': annual_volatility,
    }, index=signals.index)

    trade_entries = signals['position'].diff().fillna(0)
    trades = trade_entries[trade_entries != 0]
    return returns, trades
