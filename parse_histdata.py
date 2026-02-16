import warnings
warnings.filterwarnings('ignore')

import zipfile
import pandas as pd
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import helper_functions as fx

file_path = '/Volumes/SEAGATE/FX_data/FX_histdata/'
fx_pairs = ["EURUSD", "EURGBP", "USDJPY", "GBPUSD", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "EURJPY", "GBPJPY", "EURCHF", "AUDJPY", "EURAUD", "GBPAUD"]

# fx.unzip_histdata(file_path, fx_pairs)
# fx.create_parquet_files(file_path, fx_pairs)

def clean_data():
    for pair in fx_pairs:
        pair_df = pd.read_parquet(os.path.join(file_path, f"{pair}/", f"{pair}_merged_1M.parquet"))

        pair_df = fx.replace_neg_prices(pair_df)
        percentile_99 = pair_df['Close'].quantile(0.99)
        pair_df = fx.replace_high_prices(pair_df, threshold=percentile_99)

        pair_df = fx.drop_duplicate_datetimes(pair_df, keep='last')

        pair_df.set_index('DateTime', inplace=True)

        pair_df.to_parquet(os.path.join(file_path, f"{pair}/", f"{pair}_cleaned_1M.parquet"))

def timeline_info(pairs_list):
    timeline_data = []
    for pair in pairs_list:
        pair_df = pd.read_parquet(os.path.join(file_path, f"{pair}/", f"{pair}_cleaned_1M.parquet"))
        start_mmyy = pair_df.index.min().strftime('%Y-%m')
        end_mmyy = pair_df.index.max().strftime('%Y-%m')
        missing_mins = pair_df.index.to_series().diff().dt.total_seconds().div(60).ne(1).sum()
        max_missing_mins = pair_df.index.to_series().diff().dt.total_seconds().div(60).max()
        percentage_missing = (missing_mins / ((pair_df.index.max() - pair_df.index.min()).total_seconds() / 60)) * 100
        timeline_data.append({'Pair': pair, 'Start': start_mmyy, 'End': end_mmyy, 'Missing Minutes': missing_mins, 'Max Missing Minutes': max_missing_mins, 'Percentage Missing': percentage_missing})
    return pd.DataFrame(timeline_data)