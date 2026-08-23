# Statistical arbitrage backtest report

## Configuration

- Seed: `42`
- Assets: 40, Days: 756
- Entry z: 2.0, Exit z: 0.5, Stop z: 3.5
- Notional per pair: $100,000
- Commission: 5.0 bps, Slippage: 3.0 bps

## Pair selection

| pair    |   hedge_ratio |   eg_pvalue | eg_reject_05   |   adf_pvalue | adf_reject_05   |   johansen_trace | joh_reject_95   |   half_life_days |   sigma_eq |
|:--------|--------------:|------------:|:---------------|-------------:|:----------------|-----------------:|:----------------|-----------------:|-----------:|
| S23/S24 |        0.2195 |      0.005  | yes            |       0.0009 | yes             |            20.14 | yes             |             91.1 |    0.17447 |
| S25/S27 |       -0.2578 |      0.0118 | yes            |       0.0025 | yes             |            12.07 | no              |            324.4 |    0.51103 |
| S23/S35 |       -0.1755 |      0.0169 | yes            |       0.0037 | yes             |             9.4  | no              |            105.9 |    0.19659 |
| S02/S33 |       -0.7236 |      0.0178 | yes            |       0.0027 | yes             |            22.39 | yes             |             98.4 |    0.19494 |
| S23/S27 |        0.0878 |      0.0238 | yes            |       0.0054 | yes             |            22.56 | yes             |            107   |    0.19662 |

![Pair selection table](pair_selection.png)

## Per-pair results

### S23/S24

**Statistical tests**

| Metric | Value |
|---|---|
| EG p-value | 0.0050 |
| ADF p-value | 0.0009 |
| Johansen trace | 20.14 |
| Hedge ratio | 0.2195 |
| OU half-life (days) | 91.1 |
| OU sigma_eq | 0.17447 |

**Performance**

| Metric | Value |
|---|---|
| Sharpe (net) | 3.455 |
| Sortino | 4.723 |
| Max drawdown | -49.87% |
| CAGR | 6329.75% |
| Calmar | 126.92 |
| Hit rate | 75.0% |
| Avg win ($) | 9,099 |
| Avg loss ($) | -6,512 |
| Profit factor | 4.19 |
| Trades | 56 |
| Total costs ($) | 16,733 |
| Avg holding (days) | 10.0 |

![Equity curve](equity_S23_S24.png)

![Z-score signals](zscore_S23_S24.png)

### S25/S27

**Statistical tests**

| Metric | Value |
|---|---|
| EG p-value | 0.0118 |
| ADF p-value | 0.0025 |
| Johansen trace | 12.07 |
| Hedge ratio | -0.2578 |
| OU half-life (days) | 324.4 |
| OU sigma_eq | 0.51103 |

**Performance**

| Metric | Value |
|---|---|
| Sharpe (net) | -6.854 |
| Sortino | -6.024 |
| Max drawdown | -33367.68% |
| CAGR | -34563.75% |
| Calmar | -1.04 |
| Hit rate | 18.6% |
| Avg win ($) | 4,312 |
| Avg loss ($) | -179,083 |
| Profit factor | 0.01 |
| Trades | 429 |
| Total costs ($) | 41,459,081 |
| Avg holding (days) | 2.5 |

![Equity curve](equity_S25_S27.png)

![Z-score signals](zscore_S25_S27.png)

### S23/S35

**Statistical tests**

| Metric | Value |
|---|---|
| EG p-value | 0.0169 |
| ADF p-value | 0.0037 |
| Johansen trace | 9.40 |
| Hedge ratio | -0.1755 |
| OU half-life (days) | 105.9 |
| OU sigma_eq | 0.19659 |

**Performance**

| Metric | Value |
|---|---|
| Sharpe (net) | 2.593 |
| Sortino | 2.799 |
| Max drawdown | -69.95% |
| CAGR | 5393.56% |
| Calmar | 77.11 |
| Hit rate | 68.9% |
| Avg win ($) | 9,096 |
| Avg loss ($) | -6,429 |
| Profit factor | 3.13 |
| Trades | 45 |
| Total costs ($) | 9,150 |
| Avg holding (days) | 10.2 |

![Equity curve](equity_S23_S35.png)

![Z-score signals](zscore_S23_S35.png)

### S02/S33

**Statistical tests**

| Metric | Value |
|---|---|
| EG p-value | 0.0178 |
| ADF p-value | 0.0027 |
| Johansen trace | 22.39 |
| Hedge ratio | -0.7236 |
| OU half-life (days) | 98.4 |
| OU sigma_eq | 0.19494 |

**Performance**

| Metric | Value |
|---|---|
| Sharpe (net) | 2.630 |
| Sortino | 3.780 |
| Max drawdown | -50.70% |
| CAGR | 6414.10% |
| Calmar | 126.50 |
| Hit rate | 54.0% |
| Avg win ($) | 5,570 |
| Avg loss ($) | -3,450 |
| Profit factor | 1.89 |
| Trades | 328 |
| Total costs ($) | 96,813 |
| Avg holding (days) | 2.4 |

![Equity curve](equity_S02_S33.png)

![Z-score signals](zscore_S02_S33.png)

### S23/S27

**Statistical tests**

| Metric | Value |
|---|---|
| EG p-value | 0.0238 |
| ADF p-value | 0.0054 |
| Johansen trace | 22.56 |
| Hedge ratio | 0.0878 |
| OU half-life (days) | 107.0 |
| OU sigma_eq | 0.19662 |

**Performance**

| Metric | Value |
|---|---|
| Sharpe (net) | -1.747 |
| Sortino | -1.327 |
| Max drawdown | -297.79% |
| CAGR | -6102.95% |
| Calmar | -20.49 |
| Hit rate | 37.6% |
| Avg win ($) | 8,480 |
| Avg loss ($) | -11,018 |
| Profit factor | 0.46 |
| Trades | 109 |
| Total costs ($) | 459,890 |
| Avg holding (days) | 5.6 |

![Equity curve](equity_S23_S27.png)

![Z-score signals](zscore_S23_S27.png)

