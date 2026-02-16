# Forex Pairs Trading Strategy

Statistical arbitrage strategy exploiting cointegrated relationships between forex pairs using mean reversion with comprehensive risk management.

## Overview

Identifies cointegrated forex pairs and trades mean-reverting spreads with rolling parameter estimation and multi-layered risk controls.

**Data**: 2001-2025, multiple timeframes (1min - 1D), major and cross forex pairs.

## Features

- Cointegration testing (Engle-Granger)
- Rolling hedge ratios (60-bar window)
- Z-score based entry/exit signals
- Dynamic position sizing (inverse volatility) (to be refined)
- Multiple risk stops (z-score, time, consecutive losses, drawdown)

## Methodology

### Pair Selection
1. Test cointegration (p-value < 0.05)
2. Calculate hedge ratio via regression on price levels
3. Verify spread stationarity (ADF test)
4. Calculate half-life (filter: 5-80 bars)

### Trading Rules
- **Entry**: Z-score > ±2.0
- **Exit**: |Z-score| < 0.5, |Z-score| > 3.5 (stop), or holding > 2.5× half-life
- **Position sizing**: Scaled by inverse volatility (0.5-1.5×)

## Installation

```bash
git clone https://github.com//forex-pairs-trading.git
cd forex-pairs-trading
pip install -r requirements.txt
```

**Requirements**: pandas, numpy, matplotlib, scikit-learn, statsmodels, scipy

## Next Steps

### Immediate
- Out-of-sample validation on 2017-2025 data
- Transaction cost sensitivity analysis
- Regime filters (VIX, rate differentials)
- Multi-pair portfolio construction

### Research
- ML for parameter selection
- Kalman filter for dynamic hedge ratios
- Sentiment/positioning data integration
- Central bank event analysis

## License

MIT License

---

*February 2026*