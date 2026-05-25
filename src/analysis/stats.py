import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


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


def get_beta_lr(series1, series2):
    model = LinearRegression()
    model.fit(series1.values.reshape(-1, 1), series2.values)
    return model.coef_[0]


def plot_returns_distribution(fx_pairs, intervals, file_path):
    import os
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


def seasonal_decompose_and_plot(series, period):
    decomposition = seasonal_decompose(series, period=period)
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(decomposition.observed);  axes[0].set_title('Observed')
    axes[1].plot(decomposition.trend);     axes[1].set_title('Trend')
    axes[2].plot(decomposition.seasonal);  axes[2].set_title('Seasonal')
    axes[3].plot(decomposition.resid);     axes[3].set_title('Residual')
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
