# Backtest report

Seed: `42` | Notional per pair: $100,000 | Entry z: 2.0 | Exit z: 0.5 | Stop z: 3.5
Commission: 5.0 bps | Slippage: 3.0 bps

## Pair selection — statistical tests

| pair    |   hedge_ratio |   eg_pvalue | eg_reject_05   |   adf_pvalue | adf_reject_05   |   johansen_trace | joh_reject_95   |   half_life_days |   sigma_eq |
|:--------|--------------:|------------:|:---------------|-------------:|:----------------|-----------------:|:----------------|-----------------:|-----------:|
| S23/S24 |        0.2195 |      0.005  | yes            |       0.0009 | yes             |            20.14 | yes             |             91.1 |    0.17447 |
| S25/S27 |       -0.2578 |      0.0118 | yes            |       0.0025 | yes             |            12.07 | no              |            324.4 |    0.51103 |
| S23/S35 |       -0.1755 |      0.0169 | yes            |       0.0037 | yes             |             9.4  | no              |            105.9 |    0.19659 |
| S02/S33 |       -0.7236 |      0.0178 | yes            |       0.0027 | yes             |            22.39 | yes             |             98.4 |    0.19494 |
| S23/S27 |        0.0878 |      0.0238 | yes            |       0.0054 | yes             |            22.56 | yes             |            107   |    0.19662 |

## Performance summary

| pair    |   sharpe |   sortino |   max_dd_pct |   total_return_pct |   n_trades | hit_rate   |   profit_factor |        costs_usd |
|:--------|---------:|----------:|-------------:|-------------------:|-----------:|:-----------|----------------:|-----------------:|
| S23/S24 |    3.455 |     4.723 |        -28.4 |              265.8 |         56 | 75%        |            4.19 |  16733           |
| S25/S27 |   -6.854 |    -6.024 |     -33367.7 |           -41651.1 |        429 | 19%        |            0.01 |      4.14591e+07 |
| S23/S35 |    2.593 |     2.799 |        -69.9 |              165.8 |         45 | 69%        |            3.13 |   9150           |
| S02/S33 |    2.63  |     3.78  |        -50.7 |              276.4 |        328 | 54%        |            1.89 |  96813           |
| S23/S27 |   -1.747 |    -1.327 |       -297.8 |             -238.7 |        109 | 38%        |            0.46 | 459890           |

## Charts

### S23/S24

![Equity curve](equity_S23_S24.png)

![Z-score signals](zscore_S23_S24.png)

### S25/S27

![Equity curve](equity_S25_S27.png)

![Z-score signals](zscore_S25_S27.png)

### S23/S35

![Equity curve](equity_S23_S35.png)

![Z-score signals](zscore_S23_S35.png)

### S02/S33

![Equity curve](equity_S02_S33.png)

![Z-score signals](zscore_S02_S33.png)

### S23/S27

![Equity curve](equity_S23_S27.png)

![Z-score signals](zscore_S23_S27.png)
