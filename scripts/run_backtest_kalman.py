"""
Run the Kalman-filter universe backtest.
Usage (from project root):
    python scripts/run_backtest_kalman.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.kalman import backtest_universe_main

FILE_PATH = '/Volumes/SEAGATE/FX_data/FX_histdata/'

PAIRS = [
    ('USDCHF', 'EURCHF'),
    ('GBPUSD', 'EURCHF'),
    ('AUDJPY', 'EURCHF'),
    ('EURAUD', 'EURCHF'),
    ('EURJPY', 'EURCHF'),
]

if __name__ == '__main__':
    equity_curve, trades_df, metrics = backtest_universe_main(
        pairs      = PAIRS,
        interval   = '6H',
        file_path  = FILE_PATH,
        test_start = '2015-01-01',
        test_end   = '2022-12-31',
        output_dir = './results/backtest_kalman',
        verbosity  = 2,
    )
