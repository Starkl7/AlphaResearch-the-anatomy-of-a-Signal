"""
Run z-score threshold optimisation (Bayesian + grid search).
Usage (from project root):
    python scripts/run_optimize_z.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimize.z_thresholds import optimize_z_thresholds

FILE_PATH = '/Volumes/SEAGATE/FX_data/FX_histdata/'

PAIRS = [
    ('USDCHF', 'EURCHF'), ('GBPUSD', 'EURCHF'), ('AUDJPY', 'EURCHF'),
    ('EURAUD', 'EURCHF'), ('EURJPY', 'EURCHF'),  ('AUDUSD', 'EURCHF'),
    ('CADJPY', 'EURCHF'), ('EURUSD', 'EURCHF'),  ('GBPJPY', 'EURCHF'),
    ('NZDUSD', 'EURCHF'), ('USDCAD', 'EURCHF'),  ('USDJPY', 'EURCHF'),
]
INTERVALS = ['1D', '6H', '3H', '1H', '30T']

if __name__ == '__main__':
    results = optimize_z_thresholds(
        pairs           = PAIRS,
        intervals       = INTERVALS,
        file_path       = FILE_PATH,
        train_start     = '2001-01-01',
        train_end       = '2014-12-31',
        test_start      = '2015-01-01',
        test_end        = '2023-12-31',
        initial_capital = 100_000.0,
        n_trials        = 50,
        output_dir      = './results/optimization',
    )
