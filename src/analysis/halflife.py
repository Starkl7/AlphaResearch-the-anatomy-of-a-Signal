import numpy as np
import pandas as pd

from src.data.loader import load_resampled_returns, common_duration, get_train_test
from src.analysis.stats import get_beta_lr, adf_test


def get_baseline_hedge_ratio_and_half_life(pair1, pair2, interval, file_path, feature='Close', split_date='2012-12-31'):
    print("Loading data...")
    pair1_full = load_resampled_returns(pair1, interval, file_path)
    pair2_full = load_resampled_returns(pair2, interval, file_path)

    print("Aligning and preprocessing data...")
    pair1_aligned, pair2_aligned = common_duration(
        pair1_full, pair2_full,
        max_period_end=pd.to_datetime(split_date),
    )

    print("Processing data...")
    hedge_ratio = get_beta_lr(np.log(pair1_aligned[feature]), np.log(pair2_aligned[feature]))
    spread = np.log(pair2_aligned[feature]) - hedge_ratio * np.log(pair1_aligned[feature])

    spread_lag = spread.shift(1)
    spread_ret = spread - spread_lag
    spread_lag = spread_lag[1:]
    spread_ret = spread_ret[1:]

    half_life = -np.log(2) / get_beta_lr(spread_lag, spread_ret)
    return hedge_ratio, half_life


def get_rolling_half_life(pair1, pair2, interval, file_path, feature='Close', train_end=None, window_years=1, offset_months=1):
    pair1_full = load_resampled_returns(pair1, interval, file_path)
    pair2_full = load_resampled_returns(pair2, interval, file_path)

    max_period_end = pd.to_datetime(train_end) if train_end is not None else None
    pair1_aligned, pair2_aligned = common_duration(
        pair1_full, pair2_full, max_period_end=max_period_end,
    )
    pair1_train, _ = get_train_test(pair1_aligned, train_end)
    pair2_train, _ = get_train_test(pair2_aligned, train_end)

    window_start_min = pair1_train.index.min() + pd.DateOffset(months=offset_months)
    date_range_end = train_end if train_end is not None else pair1_train.index.max()
    month_starts = pd.date_range(start=window_start_min, end=date_range_end, freq='MS')

    records = []
    for month_date in month_starts:
        # Calibration window ends the day before month_date to avoid look-ahead bias
        window_end = month_date - pd.DateOffset(days=1)
        window_start = month_date - pd.DateOffset(years=window_years)

        s1 = pair1_train.loc[window_start:window_end, feature]
        s2 = pair2_train.loc[window_start:window_end, feature]

        if len(s1) < 30 or len(s2) < 30:
            records.append({'date': month_date, 'hedge_ratio': np.nan, 'half_life': np.nan})
            continue

        try:
            hedge_ratio = get_beta_lr(np.log(s1), np.log(s2))
            spread = np.log(s2) - hedge_ratio * np.log(s1)

            spread_lag = spread.shift(1)
            spread_ret = spread - spread_lag
            spread_lag = spread_lag[1:]
            spread_ret = spread_ret[1:]

            beta = get_beta_lr(spread_lag, spread_ret)
            half_life = -np.log(2) / beta if beta < 0 else np.nan
            records.append({'date': month_date, 'hedge_ratio': hedge_ratio, 'half_life': half_life})
        except Exception:
            records.append({'date': month_date, 'hedge_ratio': np.nan, 'half_life': np.nan})

    return pd.DataFrame(records).set_index('date')


def test_half_life_mean_reversion(pair1, pair2, interval, file_path, feature='Return', split_date='2017-12-31', test_adf=False):
    try:
        print("Loading data...")
        pair1_full = load_resampled_returns(pair1, interval, file_path)
        pair2_full = load_resampled_returns(pair2, interval, file_path)

        print("Aligning and preprocessing data...")
        pair1_full_new, pair2_full_new = common_duration(pair1_full, pair2_full)
        pair1_train, pair1_test = get_train_test(pair1_full_new, split_date)
        pair2_train, pair2_test = get_train_test(pair2_full_new, split_date)

        print("Processing data...")
        hedge_ratio = get_beta_lr(pair1_train[feature], pair2_train[feature])
        spread = pair2_train[feature] - hedge_ratio * pair1_train[feature]
        if test_adf:
            print(f"ADF test for spread between {pair1} and {pair2} at interval {interval}:")
            adf_test(spread)

        spread_lag = spread.shift(1)
        spread_ret = spread - spread_lag
        spread_lag = spread_lag[1:]
        spread_ret = spread_ret[1:]
        half_life = -np.log(2) / get_beta_lr(spread_lag, spread_ret)
        print(f"Estimated Half-Life for {pair1}/{pair2} at interval {interval}: {half_life:.2f}\n")
        return half_life, None
    except Exception as e:
        print(f"Error processing {pair1} and {pair2} at interval {interval}.\n")
        return pair1_train, pair2_train


def analyze_periods(pair1, pair2, periods, interval, file_path, half_life_bounds, max_period_end=pd.Timestamp('2012-12-31')):
    results = {}
    pair1_full = load_resampled_returns(pair1, interval, file_path)
    pair2_full = load_resampled_returns(pair2, interval, file_path)
    pair1_aligned, pair2_aligned = common_duration(
        pair1_full, pair2_full, max_period_end=max_period_end,
    )

    for period_name, (start, end) in periods.items():
        pair1_df = pair1_aligned.loc[start:end]
        pair2_df = pair2_aligned.loc[start:end]

        if len(pair1_df) < 100:
            results[period_name] = None
            continue

        hedge_ratio = get_beta_lr(np.log(pair1_df['Close']), np.log(pair2_df['Close']))
        spread = np.log(pair2_df['Close']) - hedge_ratio * np.log(pair1_df['Close'])

        spread_lag = spread.shift(1)
        spread_ret = spread - spread_lag
        spread_lag = spread_lag[1:]
        spread_ret = spread_ret[1:]

        beta = get_beta_lr(spread_lag, spread_ret)
        half_life = -np.log(2) / beta if beta < 0 else np.nan

        results[period_name] = {
            'hedge_ratio': hedge_ratio,
            'half_life': half_life,
            'valid_hl': half_life_bounds[interval][0] <= half_life <= half_life_bounds[interval][1],
        }
    return results


def classify_pair(period_results):
    pre    = period_results.get('pre_crisis')
    post   = period_results.get('post_crisis')
    crisis = period_results.get('crisis')

    if pre is None or post is None:
        return 'insufficient_data'

    pre_valid    = pre['valid_hl'] if pre else False
    post_valid   = post['valid_hl'] if post else False
    crisis_valid = crisis['valid_hl'] if crisis else False

    if pre_valid and post_valid and not crisis_valid:
        return 'TIER_1'   # structural, temporarily broke during crisis, recovered
    elif post_valid and not pre_valid:
        return 'TIER_2'   # new relationship post-crisis
    elif pre_valid and not post_valid:
        return 'TIER_3'   # pre-crisis only, relationship may be dead
    elif pre_valid and post_valid and crisis_valid:
        return 'TIER_4'   # survived everything, very stable
    else:
        return 'REJECT'
