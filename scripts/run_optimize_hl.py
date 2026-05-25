"""
Run rolling half-life crisis-period analysis.
Usage (from project root):
    python scripts/run_optimize_hl.py <valid_combos_csv> [file_path] [output_dir]

    valid_combos_csv must have columns 'pair1' and 'pair2'.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.optimize.halflife import run_halflife_analysis

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    combos_path = sys.argv[1]
    file_path   = sys.argv[2] if len(sys.argv) > 2 else '/Volumes/SEAGATE/FX_data/FX_histdata/'
    output_dir  = sys.argv[3] if len(sys.argv) > 3 else './results/plots/halflife'

    combos_df = pd.read_csv(combos_path)
    combos    = list(zip(combos_df['pair1'], combos_df['pair2']))
    print(f"Loaded {len(combos)} combos from {combos_path}")
    run_halflife_analysis(combos, file_path=file_path, output_dir=output_dir)
