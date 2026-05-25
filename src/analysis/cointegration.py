import gc
from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from hurst import compute_Hc
from statsmodels.tsa.stattools import coint, grangercausalitytests

from src.data.loader import (
    load_resampled_returns,
    common_duration,
    common_duration_c,
    forward_fill_df,
)
from src.analysis.stats import get_beta_lr


def compute_hurst_exponent(series):
    H, c, data = compute_Hc(series, kind='random_walk', simplified=True)
    return H


def wrap_hurst(fx_pairs, intervals, file_path, feature='Close'):
    """Pre-filter pair combos by Hurst exponent of the log-ratio. H < 0.5 → mean-reverting."""
    hurst_results = {}
    for interval in intervals:
        print(f"Computing Hurst exponent for {interval} interval...")
        interval_results = {}
        combos = list(combinations(fx_pairs, 2))
        for pair1, pair2 in combos:
            try:
                df1 = load_resampled_returns(pair1, interval, file_path)
                df2 = load_resampled_returns(pair2, interval, file_path)
                df1_aligned, df2_aligned = common_duration(
                    df1, df2,
                    period_in_Timedelta=pd.Timedelta(days=252 * 5),
                    max_period_end=pd.Timestamp('2012-12-31'),
                )
                log_ratio = np.log(df1_aligned[feature]) - np.log(df2_aligned[feature])
                H = compute_hurst_exponent(log_ratio)
                interval_results[(pair1, pair2)] = H
                print(f"{pair1}-{pair2}: H = {H:.4f}")
            except Exception as e:
                print(f"Error computing Hurst for {pair1} and {pair2} at interval {interval}: {e}")
                interval_results[(pair1, pair2)] = np.nan
        hurst_results[interval] = interval_results
    return hurst_results


def get_maxlag_coint(interval):
    maxlag_config = {
        '1T': 50, '5T': 50, '10T': 40, '15T': 35, '30T': 30,
        '1H': 25, '3H': 20, '6H': 15, '1D': 15,
    }
    return maxlag_config.get(interval, 20)


def granger_coint(pair1, pair2, interval, end_td, end_max, file_path):
    try:
        print(f"Processing {pair1} and {pair2} at interval {interval}...")
        import os
        pair1_full = pd.read_parquet(
            os.path.join(file_path, f"{pair1}/", f"{pair1}_resampled_{interval}_returns.parquet"),
            columns=['Close']
        )
        pair2_full = pd.read_parquet(
            os.path.join(file_path, f"{pair2}/", f"{pair2}_resampled_{interval}_returns.parquet"),
            columns=['Close']
        )
        if interval in ['1T', '5T', '10T']:
            subsample = {'1T': 10, '5T': 2, '10T': 1}
            pair1_full = pair1_full.iloc[::subsample[interval]]
            pair2_full = pair2_full.iloc[::subsample[interval]]
        pair1_aligned, pair2_aligned = common_duration_c(
            pair1_full, pair2_full,
            period_in_Timedelta=end_td, max_period_end=end_max,
        )
        t_stat, pval, crit_vals = coint(
            np.log(pair1_aligned['Close']), np.log(pair2_aligned['Close']),
            maxlag=get_maxlag_coint(interval), autolag='BIC',
        )
        del pair1_full, pair2_full, pair1_aligned, pair2_aligned
        return ((pair1, pair2), interval, pval)
    except Exception as e:
        print(f"Error processing {pair1} and {pair2} at interval {interval}: {e}")
        return ((pair1, pair2), interval, np.nan)


def initial_granger_cointegration(
    fx_pairs, combos, file_path,
    end_td=pd.Timedelta(days=252 * 5),
    end_max=pd.Timestamp('2012-12-31'),
    n_jobs=3,
):
    results = []
    for pair1, pair2, interval in combos:
        print(f"Running initial Granger cointegration test for {pair1} and {pair2} at interval {interval}...")
        result = granger_coint(pair1, pair2, interval, end_td=end_td, end_max=end_max, file_path=file_path)
        gc.collect()
        if result is not None:
            results.append(result)

    results_df = pd.DataFrame(
        [[{} for _ in fx_pairs] for _ in fx_pairs],
        index=fx_pairs, columns=fx_pairs,
    )
    for (pair1, pair2), interval, pval in results:
        cell_dict = results_df.at[pair1, pair2]
        cell_dict[interval] = np.round(pval, 3)
    return results_df


def batch_process(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def grangers_causation_matrix(df, handle_nans='drop', maxlag=12, test='ssr_chi2test', verbose=False):
    if handle_nans == 'drop':
        df = df.dropna()
    df = forward_fill_df(df)
    df2 = df.reset_index(drop=True)
    completed_list = []
    try:
        res = pd.DataFrame(
            np.zeros((len(df2.columns), len(df2.columns))),
            columns=df2.columns, index=df2.columns,
        )
        for c in res.columns:
            for r in res.index:
                if (r, c) in completed_list:
                    continue
                print(f'Testing Granger Causality: Y = {r}, X = {c}')
                test_result = grangercausalitytests(df2[[r, c]], maxlag=maxlag, verbose=False)
                p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(maxlag)]
                if verbose:
                    print(f'Y = {r}, X = {c}, P Values = {p_values}')
                res.loc[r, c] = np.min(p_values)
                completed_list.append((r, c))
        res.columns = [var + '_x' for var in res.columns]
        res.index = [var + '_y' for var in res.index]
    except Exception as e:
        print(f"Error in grangers_causation_matrix: {e}")
        return None
    return res
