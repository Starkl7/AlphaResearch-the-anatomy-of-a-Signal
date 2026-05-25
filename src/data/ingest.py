import gc
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


FILE_PATH = '/Volumes/SEAGATE/FX_data/FX_histdata/'
FX_PAIRS = [
    "EURUSD", "EURGBP", "USDJPY", "GBPUSD", "USDCHF", "USDCAD",
    "AUDUSD", "NZDUSD", "EURJPY", "GBPJPY", "EURCHF", "AUDJPY",
    "EURAUD", "GBPAUD",
]


def unzip_histdata(file_path, fx_pairs):
    for pair in fx_pairs:
        curr_file_path = os.path.join(file_path, pair)
        folder = Path(curr_file_path)
        print(f"Unzipping {pair}")
        zip_files = list(folder.glob("HISTDATA*.zip"))
        for num, zip_file in enumerate(zip_files):
            print(f"{num + 1}/{len(zip_files)}")
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
        xlsx_files = list(folder.rglob("DAT*.xlsx"))
        for num, xlsx_file in enumerate(xlsx_files):
            print(f"{num + 1}/{len(xlsx_files)}")
            df = pd.read_excel(xlsx_file, header=None)
            all_data.append(df)
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.columns = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']
        combined_df['DateTime'] = pd.to_datetime(combined_df['DateTime'], format='%Y%m%d %H%M%S')
        combined_df.drop(columns=['Volume'], inplace=True)
        combined_df.to_parquet(os.path.join(file_path, f"{pair}/{pair}_merged_1M.parquet"))
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
            df = pd.read_parquet(os.path.join(file_path, f"{pair}/{pair}_cleaned_1M.parquet"))
            resampled_df = df.resample(interval, label='right', closed='right').agg(
                {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
            )
            resampled_df.dropna(inplace=True)
            resampled_df.to_parquet(os.path.join(file_path, f"{pair}/{pair}_resampled_{interval}.parquet"))


def calculate_returns(fx_pairs, intervals, file_path):
    for interval in intervals:
        print(f"Calculating returns for {interval} interval")
        for num, pair in enumerate(fx_pairs):
            print(f"{num+1}/{len(fx_pairs)}...")
            df = pd.read_parquet(os.path.join(file_path, f"{pair}/{pair}_resampled_{interval}.parquet"))
            df['Return'] = df['Close'].pct_change()
            df['logReturn'] = np.log(df['Close'] / df['Close'].shift(1))
            df['target'] = df['logReturn'].shift(-1)
            df.to_parquet(os.path.join(file_path, f"{pair}/{pair}_resampled_{interval}_returns.parquet"))


def clean_data(file_path=FILE_PATH, fx_pairs=FX_PAIRS):
    """Clean raw 1-minute merged parquets and save as *_cleaned_1M.parquet."""
    for pair in fx_pairs:
        pair_df = pd.read_parquet(os.path.join(file_path, f"{pair}/{pair}_merged_1M.parquet"))
        pair_df = replace_neg_prices(pair_df)
        percentile_99 = pair_df['Close'].quantile(0.99)
        pair_df = replace_high_prices(pair_df, threshold=percentile_99)
        pair_df = drop_duplicate_datetimes(pair_df, keep='last')
        pair_df.set_index('DateTime', inplace=True)
        pair_df.to_parquet(os.path.join(file_path, f"{pair}/{pair}_cleaned_1M.parquet"))


def timeline_info(pairs_list, file_path=FILE_PATH):
    """Return a DataFrame summarising data availability and gap statistics per pair."""
    timeline_data = []
    for pair in pairs_list:
        pair_df = pd.read_parquet(os.path.join(file_path, f"{pair}/{pair}_cleaned_1M.parquet"))
        start_mmyy = pair_df.index.min().strftime('%Y-%m')
        end_mmyy = pair_df.index.max().strftime('%Y-%m')
        diffs = pair_df.index.to_series().diff().dt.total_seconds().div(60)
        missing_mins = diffs.ne(1).sum()
        max_missing_mins = diffs.max()
        total_mins = (pair_df.index.max() - pair_df.index.min()).total_seconds() / 60
        percentage_missing = (missing_mins / total_mins) * 100 if total_mins > 0 else 0
        timeline_data.append({
            'Pair': pair,
            'Start': start_mmyy,
            'End': end_mmyy,
            'Missing Minutes': missing_mins,
            'Max Missing Minutes': max_missing_mins,
            'Percentage Missing': percentage_missing,
        })
    return pd.DataFrame(timeline_data)
