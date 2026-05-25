import os
import numpy as np
import pandas as pd


def load_resampled_returns(pair, interval, file_path):
    df = pd.read_parquet(os.path.join(file_path, f"{pair}/", f"{pair}_resampled_{interval}_returns.parquet"))
    return df


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
    freq = get_dynamic_freq(df.index)
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df_filled = df.reindex(full_range)
    df_filled = df_filled.fillna(method='ffill')
    return df_filled


def forward_fill_returns(df, freq):
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df_filled = df.reindex(full_range)
    df_filled['Close'] = df_filled['Close'].ffill()
    for col in ['Open', 'High', 'Low']:
        df_filled[col] = df_filled[col].fillna(df_filled['Close'])
    for col in ['Return', 'logReturn', 'target']:
        df_filled[col] = df_filled[col].fillna(0)
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

    df1 = forward_fill_df(df1).resample(common_freq, label='right', closed='right').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).ffill()
    df2 = forward_fill_df(df2).resample(common_freq, label='right', closed='right').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).ffill()

    df1['Return'] = df1['Close'].pct_change().fillna(0)
    df2['Return'] = df2['Close'].pct_change().fillna(0)
    df1['logReturn'] = np.log(df1['Close'] / df1['Close'].shift(1)).fillna(0)
    df2['logReturn'] = np.log(df2['Close'] / df2['Close'].shift(1)).fillna(0)
    return df1.loc[start:end], df2.loc[start:end]


def common_duration_c(df1, df2, start_date=None, period_in_Timedelta=None, max_period_end=None):
    """Close-only variant of common_duration — faster for cointegration tests."""
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

    df1 = forward_fill_df(df1).resample(common_freq, label='right', closed='right').agg(
        {'Close': 'last'}
    ).ffill()
    df2 = forward_fill_df(df2).resample(common_freq, label='right', closed='right').agg(
        {'Close': 'last'}
    ).ffill()

    df1['Return'] = df1['Close'].pct_change().fillna(0)
    df2['Return'] = df2['Close'].pct_change().fillna(0)
    df1['logReturn'] = np.log(df1['Close'] / df1['Close'].shift(1)).fillna(0)
    df2['logReturn'] = np.log(df2['Close'] / df2['Close'].shift(1)).fillna(0)
    return df1.loc[start:end], df2.loc[start:end]


def get_train_test(df, split_date):
    train = df.loc[:split_date].copy()
    test = df.loc[split_date:].copy()
    return train, test


def get_test_data(df, start, end):
    return df.loc[start:end].copy()
