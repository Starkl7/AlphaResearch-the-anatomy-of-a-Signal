import os
import matplotlib.pyplot as plt
from contextlib import redirect_stdout, redirect_stderr
import trade_functions as tf


# This function runs the backtest implementing the Hedge ratio as a Linear regression Beta and the Half-life as a measure of mean reversion speed.
def run_backtest(fx_pairs, intervals, baseline_hedge_ratios, baseline_half_lives, file_path):
    output_dir = './backtest_results'
    os.makedirs(output_dir, exist_ok=True)

    # Redirect text output
    text_output_file = os.path.join(output_dir, 'backtest2_output.log')

    # Set matplotlib to non-interactive backend to prevent plots from displaying
    plt.ioff()

    with open(text_output_file, 'w') as f:
        with redirect_stdout(f), redirect_stderr(f):
            # Call your function here
            for pair1 in fx_pairs:
                for pair2 in fx_pairs:
                    if pair1 != pair2:
                        for interval in intervals:
                            print(f"Processing {pair1} & {pair2} at {interval} interval...")
                            idx = intervals.index(interval)
                            try:
                                hedge_ratio_baseline = baseline_hedge_ratios.loc[pair1, pair2][idx]
                                half_life_baseline = baseline_half_lives.loc[pair1, pair2][idx]
                            except Exception as e:
                                print(f"Error retrieving baseline values for {pair1} & {pair2} at {interval}: {e}")
                                continue
                            if hedge_ratio_baseline == 0 or half_life_baseline == 0:
                                print("Skipping due to invalid baseline values...")
                                continue
                            signals, returns, trades, fig = tf.backtest_pairs_main2(
                                pair1, pair2, interval, 
                                hedge_ratio_baseline=hedge_ratio_baseline, 
                                half_life_baseline=half_life_baseline, 
                                start='2013-01-01', 
                                end='2017-12-31', 
                                file_path=file_path
                            )
                            # Save the plot if it was created
                            if fig is not None:
                                plot_file = os.path.join(output_dir, f'backtest2_plot_{pair1}_{pair2}_{interval}.png')
                                fig.savefig(plot_file, dpi=300, bbox_inches='tight')
                                plt.close(fig)

    # Turn interactive mode back on
    plt.ion()

    print(f"Text output saved to: {text_output_file}")
    print(f"Plots saved to: {output_dir}")