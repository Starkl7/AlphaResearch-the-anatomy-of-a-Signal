import gc
from joblib import Parallel, delayed
import zipfile
import pandas as pd
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller, kpss, coint, grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import trade_functions as tf
from hurst import compute_Hc
from itertools import combinations


# file_path = '/Volumes/SEAGATE/FX_data/FX_histdata/'
# fx_pairs = ["EURUSD", "EURGBP", "USDJPY", "GBPUSD", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "EURJPY", "GBPJPY", "EURCHF", "AUDJPY", "EURAUD", "GBPAUD"]

def unzip_histdata(file_path, fx_pairs):
    for pair in fx_pairs:
        curr_file_path = os.path.join(file_path, pair)
        folder = Path(curr_file_path)
        print(f"Unzipping {pair}")
        for num, zip_file in enumerate(folder.glob("HISTDATA*.zip")):
            print(f"{num + 1}/{len(list(folder.glob('HISTDATA*.zip')))}")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(folder)

def create_parquet_files(file_path, fx_pairs):
    for pair in fx_pairs:
        curr_file_path = os.path.join(file_path, pair)
        folder = Path(curr_file_path)
        print(f"Deleting existing parquet file for {pair}")
        try:
            os.remove(os.path.join(curr_file_path, f"{pair}_merged_1M.parquet"))
        except FileNotFoundError:
            pass
        print(f"Combining Excel files for {pair}")
        all_data = []
        for num, xlsx_file in enumerate(folder.rglob("DAT*.xlsx")):
            print(f"{num + 1}/{len(list(folder.rglob('DAT*.xlsx')))}")
            df = pd.read_excel(xlsx_file, header=None)
            all_data.append(df)

        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.columns = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']
        combined_df['DateTime'] = pd.to_datetime(combined_df['DateTime'], format='%Y%m%d %H%M%S')
        combined_df.drop(columns=['Volume'], inplace=True)
        combined_df.to_parquet(f"/Volumes/SEAGATE/FX_data/FX_histdata/{pair}/{pair}_merged_1M.parquet")
        print(f"Created {pair}_merged_1M.parquet")

def replace_neg_prices(df):
    ohlc_cols = ['Open', 'High', 'Low', 'Close']
    print(f"Replacing {df[df['Close'] <= 0].shape[0]} rows with non-positive close prices with 0")
    df[ohlc_cols] = df[ohlc_cols].clip(lower=0)
    return df

def replace_high_prices(df, threshold):
    ohlc_cols = ['Open', 'High', 'Low', 'Close']
    print(f"Replacing {df[df['Close'] > (threshold*2)].shape[0]} rows with close prices above {threshold * 2} with {threshold}")
    df[ohlc_cols] = df[ohlc_cols].clip(upper=threshold)
    return df

def drop_duplicate_datetimes(df, keep='last'):
    duplicate_datetimes = df[df.duplicated(subset=['DateTime'], keep=False)]
    print(f"Found {duplicate_datetimes.shape[0]} rows with duplicate DateTime entries")
    print(f"Keeping only {keep} values for each duplicate DateTime entry")
    df.drop_duplicates(subset=['DateTime'], keep=keep, inplace=True)
    return df

def create_interval_datasets(fx_pairs, intervals, file_path):
    for interval in intervals:
        print(f"Creating {interval} datasets for all pairs")
        for pair in fx_pairs:
            print(f"Processing {pair} for {interval} interval")
            print(f"Creating {interval} dataset for {pair}")
            df = pd.read_parquet(os.path.join(file_path, f"{pair}/", f"{pair}_cleaned_1M.parquet"))
            resampled_df = df.resample(interval, label='right', closed='right').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'})
            resampled_df.dropna(inplace=True)
            resampled_df.to_parquet(os.path.join(file_path, f"{pair}/", f"{pair}_resampled_{interval}.parquet"))

def calculate_returns(fx_pairs, intervals, file_path):
    for interval in intervals:
        print(f"Calculating returns for {interval} interval")
        for num, pair in enumerate(fx_pairs):
            print(f"{num+1}/{len(fx_pairs)}...")
            df = pd.read_parquet(os.path.join(file_path, f"{pair}/", f"{pair}_resampled_{interval}.parquet"))
            df['Return'] = df['Close'].pct_change()
            df['logReturn'] = np.log(df['Close'] / df['Close'].shift(1))
            df['target'] = df['logReturn'].shift(-1)
            df.to_parquet(os.path.join(file_path, f"{pair}/", f"{pair}_resampled_{interval}_returns.parquet"))
        
def plot_returns_distribution(fx_pairs, intervals, file_path):
    for interval in intervals:
        print(f"Plotting returns distribution for {interval} interval")
        for pair in fx_pairs:
            df = pd.read_parquet(os.path.join(file_path, f"{pair}/", f"{pair}_resampled_{interval}_returns.parquet"))
            plt.figure(figsize=(7, 4))
            sns.histplot(df['logReturn'].dropna(), binwidth=0.001, kde=True, color='C1')
            plt.xlim(-0.05, 0.05)
            plt.title(f'{pair} Returns Distribution - {interval}')
            plt.xlabel('Return')
            plt.ylabel('Frequency')
            plt.grid()
            plt.tight_layout()
            plt.show()
            
def calc_returns_gap(main_pair, interval, fx_pairs, file_path):
    returns_gap = pd.DataFrame()
    fx_pairs = [p for p in fx_pairs if p != main_pair]
    main_df = pd.read_parquet(os.path.join(file_path, f"{main_pair}/", f"{main_pair}_resampled_{interval}_returns.parquet"))
    print(f"Calculating returns gap for {main_pair} at {interval} interval")
    for pair in fx_pairs:
        ref_df = pd.read_parquet(os.path.join(file_path, f"{pair}/", f"{pair}_resampled_{interval}_returns.parquet"))
        returns_gap[f'ReturnGap_{pair}_{interval}'] = main_df['logReturn'] - ref_df['logReturn']
        returns_gap[f'ReturnGap_{pair}_{interval}'] = returns_gap[f'ReturnGap_{pair}_{interval}'].replace([np.inf, -np.inf], np.nan)
    
    return returns_gap

def adf_test(series, verbosity=0):
    result = adfuller(series, autolag="AIC", maxlag=10)
    print('ADF Statistics: %f' % result[0])
    print('p-value: %f' % result[1])
    if verbosity > 0:
        print('Critical values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))

def kpss_test(series, verbosity=0):    
    statistic, p_value, n_lags, critical_values = kpss(series)
    
    print(f'KPSS Statistic: {statistic}')
    print(f'p-value: {p_value}')
    if verbosity > 0:
        print(f'num lags: {n_lags}')
        print('Critical Values:')
        for key, value in critical_values.items():
            print(f'   {key} : {value}')

def forward_fill_returns(df, freq):
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    missing_index = full_range.difference(df.index)
    # print(missing_index)
    df_filled = df.reindex(full_range)
    df_filled['Close'] = df_filled['Close'].ffill()

    for col in ['Open', 'High', 'Low']:
        df_filled[col] = df_filled[col].fillna(df_filled['Close'])
    
    for col in ['Return', 'logReturn', 'target']:
        df_filled[col] = df_filled[col].fillna(0)
    
    return df_filled


# model = VAR(df_train_transformed)
# for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
#     result = model.fit(i)
#     print('Lag Order =', i)
#     print('AIC : ', result.aic)
#     print('BIC : ', result.bic)
#     print('FPE : ', result.fpe)
#     print('HQIC: ', result.hqic, '\n')

def seasonal_decompose_and_plot(series, period):
    decomposition = seasonal_decompose(series, period=period)
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    axes[0].plot(decomposition.observed)
    axes[0].set_title('Observed')
    
    axes[1].plot(decomposition.trend)
    axes[1].set_title('Trend')
    
    axes[2].plot(decomposition.seasonal)
    axes[2].set_title('Seasonal')
    
    axes[3].plot(decomposition.resid)
    axes[3].set_title('Residual')
    
    plt.tight_layout()
    plt.show()

def acf_and_pacf(series, lags):
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    plot_acf(series, lags=lags, ax=axes[0])
    axes[0].set_title('Autocorrelation Function')
    axes[0].set_xlabel('Lags')
    axes[0].set_ylabel('Autocorrelation')
    axes[0].grid()
    plot_pacf(series, lags=lags, ax=axes[1])
    axes[1].set_title('Partial Autocorrelation Function')
    axes[1].set_xlabel('Lags')
    axes[1].set_ylabel('Partial Autocorrelation')
    axes[1].grid()
    plt.tight_layout()
    plt.show()

# ['1T', '5T', '10T', '15T', '30T', '1H', '3H', '6H', '1D']
def get_dynamic_freq(index):
    if isinstance(index, pd.DatetimeIndex):
        delta = index.diff().min()
        if delta <= pd.Timedelta(minutes=1):
            return 'T'
        elif delta <= pd.Timedelta(minutes=5):
            return '5T'
        elif delta <= pd.Timedelta(minutes=10):
            return '10T'
        elif delta <= pd.Timedelta(minutes=15):
            return '15T'
        elif delta <= pd.Timedelta(minutes=30):
            return '30T'
        elif delta <= pd.Timedelta(hours=1):
            return 'H'
        elif delta <= pd.Timedelta(hours=3):
            return '3H'
        elif delta <= pd.Timedelta(hours=6):
            return '6H'
        else:
            return 'D'
    else:
        raise ValueError(f"Index must be a DatetimeIndex: {index}")
    
def forward_fill_df(df):
    range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=get_dynamic_freq(df.index))
    df_filled = df.reindex(range)
    df_filled = df_filled.fillna(method='ffill')
    return df_filled

def common_duration(df1, df2, start_date=None, period_in_Timedelta=None, max_period_end=None):
    df1.replace([np.inf, -np.inf], np.nan, inplace=True)
    df2.replace([np.inf, -np.inf], np.nan, inplace=True)
    df1 = df1.dropna()
    df2 = df2.dropna()
    start = max(df1.index.min(), df2.index.min())
    if start_date is not None:
        start = pd.to_datetime(start_date)

    end = None
    if period_in_Timedelta is not None:
        end = start + period_in_Timedelta
    if max_period_end is not None:
        if end is not None:
            end = min(end, max_period_end)
        else:
            end = max_period_end
    if end is None:
        end = min(df1.index.max(), df2.index.max())

    interval1 = get_dynamic_freq(df1.index)
    interval2 = get_dynamic_freq(df2.index)
    if interval1 != interval2:
        print(f"df1: {interval1}, df2: {interval2}. Downsampling to common frequency.")
        common_freq = max(interval1, interval2, key=lambda x: pd.Timedelta(x))
    else:
        common_freq = interval1

    # Even returns were getting forward-filled. Fixed it!
    df1 = forward_fill_df(df1).resample(common_freq, label='right', closed='right').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).ffill()
    df2 = forward_fill_df(df2).resample(common_freq, label='right', closed='right').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).ffill()

    df1['Return'] = df1['Close'].pct_change().fillna(0)
    df2['Return'] = df2['Close'].pct_change().fillna(0)

    df1['logReturn'] = np.log(df1['Close'] / df1['Close'].shift(1)).fillna(0)
    df2['logReturn'] = np.log(df2['Close'] / df2['Close'].shift(1)).fillna(0)
    return df1.loc[start:end], df2.loc[start:end]

def common_duration_c(df1, df2, start_date=None, period_in_Timedelta=None, max_period_end=None):
    df1.replace([np.inf, -np.inf], np.nan, inplace=True)
    df2.replace([np.inf, -np.inf], np.nan, inplace=True)
    df1 = df1.dropna()
    df2 = df2.dropna()
    start = max(df1.index.min(), df2.index.min())
    if start_date is not None:
        start = pd.to_datetime(start_date)

    end = None
    if period_in_Timedelta is not None:
        end = start + period_in_Timedelta
    if max_period_end is not None:
        if end is not None:
            end = min(end, max_period_end)
        else:
            end = max_period_end
    if end is None:
        end = min(df1.index.max(), df2.index.max())

    interval1 = get_dynamic_freq(df1.index)
    interval2 = get_dynamic_freq(df2.index)
    if interval1 != interval2:
        print(f"df1: {interval1}, df2: {interval2}. Downsampling to common frequency.")
        common_freq = max(interval1, interval2, key=lambda x: pd.Timedelta(x))
    else:
        common_freq = interval1

    # Even returns were getting forward-filled. Fixed it!
    df1 = forward_fill_df(df1).resample(common_freq, label='right', closed='right').agg({
        'Close': 'last'
    }).ffill()
    df2 = forward_fill_df(df2).resample(common_freq, label='right', closed='right').agg({
        'Close': 'last'
    }).ffill()

    df1['Return'] = df1['Close'].pct_change().fillna(0)
    df2['Return'] = df2['Close'].pct_change().fillna(0)

    df1['logReturn'] = np.log(df1['Close'] / df1['Close'].shift(1)).fillna(0)
    df2['logReturn'] = np.log(df2['Close'] / df2['Close'].shift(1)).fillna(0)
    return df1.loc[start:end], df2.loc[start:end]

def grangers_causation_matrix(df, handle_nans='drop', maxlag=12,test='ssr_chi2test', verbose=False):
    if handle_nans == 'drop':
        df = df.dropna()
    df = forward_fill_df(df)
    df2 = df.reset_index(drop=True)
    completed_list = []
    try:
        res = pd.DataFrame(np.zeros((len(df2.columns), len(df2.columns))), columns=df2.columns, index=df2.columns)
        for c in res.columns:
            for r in res.index:
                if (r, c) in completed_list:
                    continue
                print(f'Testing Granger Causality: Y = {r}, X = {c}')
                test_result = grangercausalitytests(df2[[r, c]], maxlag=maxlag, verbose=False)
                p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
                if verbose: 
                    print(f'Y = {r}, X = {c}, P Values = {p_values}')
                min_p_value = np.min(p_values)
                res.loc[r, c] = min_p_value
                completed_list.append((r, c))
        res.columns = [var + '_x' for var in res.columns]
        res.index = [var + '_y' for var in res.index]
    except Exception as e:
        print(f"Error in grangers_causation_matrix: {e}")
        print(df2.index)
        return None
    return res

def get_train_test(df, split_date):
    train = df.loc[:split_date].copy()
    test = df.loc[split_date:].copy()
    return train, test

def get_test_data(df, start, end):
    test_data = df.loc[start:end].copy()
    return test_data

def get_beta_lr(series1, series2):
    model = LinearRegression()
    model.fit(series1.values.reshape(-1, 1), series2.values)
    beta = model.coef_[0]
    return beta

def test_half_life_mean_reversion(pair1, pair2, interval, file_path, feature='Return', split_date='2017-12-31', test_adf=False):
    try:
        print("Loading data...")
        pair1_full = pd.read_parquet(os.path.join(file_path, f"{pair1}/", f"{pair1}_resampled_{interval}_returns.parquet"))
        pair2_full = pd.read_parquet(os.path.join(file_path, f"{pair2}/", f"{pair2}_resampled_{interval}_returns.parquet"))
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
        # print("")

        spread_lag = spread.shift(1)
        spread_ret = spread - spread_lag
        spread_lag = spread_lag[1:]
        spread_ret = spread_ret[1:]
        half_life = -np.log(2) / get_beta_lr(spread_lag, spread_ret)
        print(f"Estimated Half-Life for {pair1}/{pair2} at interval {interval}: {half_life:.2f}")
        print("\n")
        return half_life, None
    except Exception as e:
        print(f"Error processing {pair1} and {pair2} at interval {interval}.\n")
        return pair1_train, pair2_train

def get_baseline_hedge_ratio_and_half_life(pair1, pair2, interval, file_path, feature='Close', split_date='2012-12-31'):
    print("Loading data...")
    pair1_full = tf.load_resampled_returns(pair1, interval, file_path)
    pair2_full = tf.load_resampled_returns(pair2, interval, file_path)
    
    print("Aligning and preprocessing data...")
    pair1_aligned, pair2_aligned= common_duration(pair1_full, pair2_full, start_date=None, period_in_Timedelta=None, max_period_end=pd.to_datetime(split_date))

    print("Processing data...")    
    hedge_ratio = get_beta_lr(pair1_aligned[feature], pair2_aligned[feature])
    spread = pair2_aligned[feature] - hedge_ratio * pair1_aligned[feature]
    
    spread_lag = spread.shift(1)
    spread_ret = spread - spread_lag
    spread_lag = spread_lag[1:]
    spread_ret = spread_ret[1:]
    
    half_life = -np.log(2) / get_beta_lr(spread_lag, spread_ret)
    return hedge_ratio, half_life


def get_rolling_half_life(pair1, pair2, interval, file_path, feature='Close', train_end='2012-12-31', window_years=1):

    pair1_full = pd.read_parquet(os.path.join(file_path, f"{pair1}/", f"{pair1}_resampled_{interval}_returns.parquet"))
    pair2_full = pd.read_parquet(os.path.join(file_path, f"{pair2}/", f"{pair2}_resampled_{interval}_returns.parquet"))

    pair1_aligned, pair2_aligned = common_duration(pair1_full, pair2_full, start_date=None, period_in_Timedelta=pd.Timedelta(days=252*5), max_period_end=pd.to_datetime(train_end))
    pair1_train, _ = get_train_test(pair1_aligned, train_end)
    pair2_train, _ = get_train_test(pair2_aligned, train_end)

    # First valid window end = data start + 1 year
    window_start_min = pair1_train.index.min() + pd.DateOffset(years=window_years)
    month_starts = pd.date_range(start=window_start_min, end=train_end, freq='MS')

    records = []
    for month_date in month_starts:
        window_start = month_date - pd.DateOffset(years=window_years)

        s1 = pair1_train.loc[window_start:month_date, feature]
        s2 = pair2_train.loc[window_start:month_date, feature]

        if len(s1) < 30 or len(s2) < 30:
            records.append({'date': month_date, 'hedge_ratio': np.nan, 'half_life': np.nan})
            continue

        try:
            hedge_ratio = get_beta_lr(s1, s2)
            spread = s2 - hedge_ratio * s1

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


def parallel_process_window(pair1_aligned, pair2_aligned, window_idx, window_start, window_end, autolag='AIC'):
        try:
            pair1_window = pair1_aligned.loc[window_start:window_end]
            pair2_window = pair2_aligned.loc[window_start:window_end]
            t_stat, pval, crit_vals = coint(pair1_window['Close'], pair2_window['Close'], autolag=autolag)
            return (window_end, pval)
        except Exception as e:
            print(f"Error at window {window_idx} ending {window_end}: {e}")
            return (window_end, np.nan)
        
def get_rolling_granger_pvalues(pair1, pair2, autolag='AIC', period=pd.Timedelta(days=252*5), n_jobs=-1):
    pair1_aligned, pair2_aligned = common_duration(pair1, pair2, period_in_Timedelta=None)
    
    start_date = pair1_aligned.index.min()
    end_date = pair1_aligned.index.max()
    freq = get_dynamic_freq(pair1_aligned.index)
    
    windows = []
    current_start = start_date
    window_end = current_start + period
    
    while window_end <= end_date:
        windows.append((current_start, window_end))
        current_start += pd.Timedelta(days=21)
        window_end = current_start + period
    
    print(f"Running rolling cointegration test from {start_date} to {end_date}...")
    print(f"Window size: {period}")
    print(f"Total windows to process: {len(windows)}")
    # print("Error incoming....")
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(parallel_process_window)(pair1_aligned, pair2_aligned, idx, start, end) 
        for idx, (start, end) in enumerate(windows)
    )

    results.sort(key=lambda x: x[0])
    dates = [r[0] for r in results]
    pvalues = [r[1] for r in results]
    print(f"Completed {len(results)} windows")
    
    pvalue_series = pd.Series(pvalues, index=dates, name='pval')
    return pvalue_series

def get_maxlag_coint(interval):
    maxlag_config = {
        '1T': 50,
        '5T': 50,
        '10T': 40,
        '15T': 35,
        '30T': 30,
        '1H': 25,
        '3H': 20,
        '6H': 15,
        '1D': 15
    }
    return maxlag_config.get(interval, 20)

def parallel_granger_coint(pair1, pair2, interval, end_td, end_max, file_path):
    try:
        print(f"Processing {pair1} and {pair2} at interval {interval}...")
        pair1_full = pd.read_parquet(os.path.join(file_path, f"{pair1}/", f"{pair1}_resampled_{interval}_returns.parquet"), columns=['Close'])
        pair2_full = pd.read_parquet(os.path.join(file_path, f"{pair2}/", f"{pair2}_resampled_{interval}_returns.parquet"), columns=['Close'])
        if interval in ['1T', '5T', '10T']:
            subsample = {
                '1T': 10,   # every 10th row → 260k rows
                '5T': 5,    # every 5th row → 104k rows
                '10T': 3
            }
            pair1_full = pair1_full.iloc[::subsample[interval]]
            pair2_full = pair2_full.iloc[::subsample[interval]]

        pair1_aligned, pair2_aligned = common_duration_c(pair1_full, pair2_full, start_date=None, period_in_Timedelta=end_td, max_period_end=end_max)
        # corr_precheck = pair1_aligned['Close'].pct_change().dropna().corr(pair2_aligned['Close'].pct_change().dropna())
        # if abs(corr_precheck) < 0.1:
        #     print(f"Skipping {pair1} and {pair2} at interval {interval} due to low correlation ({corr_precheck:.2f})")
        #     return ((pair1, pair2), np.nan)
        t_stat, pval, crit_vals = coint(pair1_aligned['Close'], pair2_aligned['Close'], maxlag=get_maxlag_coint(interval), autolag='BIC')
        del pair1_full, pair2_full, pair1_aligned, pair2_aligned
        return ((pair1, pair2), interval, pval)
    except Exception as e:
        print(f"Error processing {pair1} and {pair2} at interval {interval}: {e}")
        return ((pair1, pair2), interval, np.nan)

def initial_granger_cointegration(fx_pairs, combos, file_path, end_td=pd.Timedelta(days=252*5), end_max=pd.Timestamp('2012-12-31'), n_jobs=3):
    # res = {}
    # for pair1, pair2, interval in combos:
    # print(f"Running initial Granger cointegration tests for interval {interval}...")
    # results = Parallel(n_jobs=n_jobs, verbose=11)(
    #     delayed(parallel_granger_coint)(pair1, pair2, interval, end_td = end_td, end_max = end_max, file_path=file_path)
    #     for pair1, pair2, interval in combos
    # )
    results = []
    for pair1, pair2, interval in combos:
        print(f"Running initial Granger cointegration test for {pair1} and {pair2} at interval {interval}...")
        result = parallel_granger_coint(pair1, pair2, interval, end_td=end_td, end_max=end_max, file_path=file_path)
        gc.collect()
        if result is not None:
            results.append(result)

    results_df = pd.DataFrame(
        [[{} for _ in fx_pairs] for _ in fx_pairs],
        index=fx_pairs,
        columns=fx_pairs
    )
    for (pair1, pair2), interval, pval in results:
        cell_dict = results_df.at[pair1, pair2]
        cell_dict[interval] = np.round(pval, 3)
    
    return results_df

def compute_hurst_exponent(series):
    H, c, data = compute_Hc(series, kind='random_walk', simplified=True)
    return H

# Hurst exponent of log(price1/price2) is the same as of log(price2/price1)
# This is pre-filtering step, before computing Granger cointegration on combos with H < 0.5.
def wrap_hurst(fx_pairs, intervals, file_path, feature='Close'):
    hurst_results = {}
    for interval in intervals:
        print(f"Computing Hurst exponent for {interval} interval...")
        interval_results = {}
        combos = list(combinations(fx_pairs, 2))
        for pair1, pair2 in combos:
            # if pair1 != pair2:
            try:
                df1 = tf.load_resampled_returns(pair1, interval, file_path)
                df2 = tf.load_resampled_returns(pair2, interval, file_path)
                df1_aligned, df2_aligned = common_duration(df1, df2, start_date=None, period_in_Timedelta=pd.Timedelta(days=252*5), max_period_end=pd.Timestamp('2012-12-31'))
                log_df1 = np.log(df1_aligned[feature])
                log_df2 = np.log(df2_aligned[feature])
                log_ratio = log_df1 - log_df2
                H = compute_hurst_exponent(log_ratio)
                interval_results[(pair1, pair2)] = H
                print(f"{pair1}-{pair2}: H = {H:.4f}")
            except Exception as e:
                print(f"Error computing Hurst for {pair1} and {pair2} at interval {interval}: {e}")
                interval_results[(pair1, pair2)] = np.nan
        hurst_results[interval] = interval_results
    return hurst_results

def batch_process(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def analyze_periods(pair1, pair2, periods, interval, file_path, half_life_bounds, max_period_end=pd.Timestamp('2012-12-31')):
    results = {}
    pair1_full = tf.load_resampled_returns(pair1, interval, file_path)
    pair2_full = tf.load_resampled_returns(pair2, interval, file_path)
    pair1_aligned, pair2_aligned = common_duration(pair1_full, pair2_full, start_date=None, period_in_Timedelta=None, max_period_end=max_period_end)

    for period_name, (start, end) in periods.items():
        pair1_df = pair1_aligned.loc[start:end]
        pair2_df = pair2_aligned.loc[start:end]

        if len(pair1_df) < 100:  # insufficient data
            results[period_name] = None
            continue
        
        # Use Close prices for hedge ratio calculation
        hedge_ratio = get_beta_lr(pair1_df['Close'], pair2_df['Close'])
        spread = pair2_df['Close'] - hedge_ratio * pair1_df['Close']

        spread_lag = spread.shift(1)
        spread_ret = spread - spread_lag
        spread_lag = spread_lag[1:]
        spread_ret = spread_ret[1:]

        beta = get_beta_lr(spread_lag, spread_ret)
        half_life = -np.log(2) / beta if beta < 0 else np.nan
        
        results[period_name] = {
            'hedge_ratio': hedge_ratio,
            'half_life': half_life,
            'valid_hl': half_life_bounds[interval][0] <= half_life <= half_life_bounds[interval][1]
        }
    return results

def classify_pair(period_results):
    pre  = period_results.get('pre_crisis')
    post = period_results.get('post_crisis')
    crisis = period_results.get('crisis')
    
    if pre is None or post is None:
        return 'insufficient_data'
    
    pre_valid  = pre['valid_hl'] if pre else False
    post_valid = post['valid_hl'] if post else False
    crisis_valid = crisis['valid_hl'] if crisis else False
    
    if pre_valid and post_valid and not crisis_valid:
        return 'TIER_1'  # structural, temporarily broke, recovered
    elif post_valid and not pre_valid:
        return 'TIER_2'  # new relationship post-crisis
    elif pre_valid and not post_valid:
        return 'TIER_3'  # pre-crisis only, relationship may be dead
    elif pre_valid and post_valid and crisis_valid:
        return 'TIER_4'  # survived everything, very stable
    else:
        return 'REJECT'