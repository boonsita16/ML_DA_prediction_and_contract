# Day-Ahead Electricity Price Forecasting and Hypothetical Trading

This project compares two XGBoost models for day-ahead (DA) electricity prices and tests how their forecasts affect decisions in a hypothetical fixed-price contract simulator. It explores whether better price accuracy translates into better commercial outcomes without requiring intraday (ID) price data.

Two separate workflows, each available as a Python script and a notebook:

| Workflow | Script | Notebook |
|---|---|---|
| Fetch DA/weather/fundamental forecasts, train, evaluate and export | `da_prediction.py` | `da_prediction.ipynb` |
| Read held-out forecasts, plan hypothetical trades and settle | `hypothetical_contract.py` | `hypothetical_contract.ipynb` |

The notebooks import their matching script, avoiding divergent implementations. Keep all four files in the same directory. Run the prediction workflow before the contract workflow.

## Quick start

Python 3.11+ recommended. From this directory:

```bash
python -m pip install -r requirements.txt
python da_prediction.py --demo --output prediction_demo
python hypothetical_contract.py --input prediction_demo --output contract_demo
python -m unittest -v test_project
```

The demo is entirely SYNTHETIC and requires no API token. Its scores do not represent real market performance. For notebooks, install a Jupyter front end if you do not already have one and set `DEMO = True` in both notebooks.

For actual data, set your ENTSO-E token as `ENTSOE_API_TOKEN`, or use your existing `../entsoe-api.txt` file. Do not commit credentials or notebook outputs containing secrets.

```bash
python da_prediction.py --output prediction_output
python hypothetical_contract.py --input prediction_output --output contract_output
```

Defaults: DK2, Copenhagen coordinates, 2026-06-01 through 2026-10-01 exclusive. Change zone, location and weather model together where appropriate:

```bash
python da_prediction.py --zone DK_2 --start 2026-06-01 --end 2026-10-01 --latitude 55.6759 --longitude 12.5655 --timezone Europe/Copenhagen --weather-model dmi_seamless
python hypothetical_contract.py --input prediction_output --spread 4 --buffer 5 --fee 0.5 --slippage 0.5 --max-mw 1 --max-daily-mwh 10 --stress-move 100 --stress-budget 500
```

## Prediction design

- Historical DA prices are the training target. Weather inputs versus generation + residual-load inputs; identical calendar features and XGBoost parameters.
- UTC joins and bounded within-interval forward fills for forecast MW; never fill target prices. Missing renewable components raise an error, not an assumed zero.
- Same complete-case observations for both models; chronological floor(80% of rows) train, 20% test. No test labels in model fitting, early stopping, parameter selection, or quote generation.
- Fixed 300 trees, depth 4, learning rate .05. No tuning on the held-out period.
- Overall and training-P90 high-price MAE, RMSE, bias, R², sample count and fixed EUR 200/MWh diagnostic. High-price flags are evaluation-only.
- Quote benchmark: frozen training mean by weekday/hour/quarter-hour, with hour/global fallback. It is not an observed executable price.
- All models are frozen through the test window. Forecast vintage/publication times are not verified by these historical endpoints.

### Interpreting prediction accuracy

| Metric | Interpretation |
|---|---|
| Overall MAE | Average absolute error across all retained test intervals, in EUR/MWh |
| High-price MAE | Average absolute error when actual test prices exceed the training-price 90th percentile |
| RMSE | Gives greater weight to large forecast errors |
| Bias | Mean prediction minus actual price; negative means underprediction |
| R² | Improvement relative to predicting the evaluated sample mean; can be negative |

For example, an MAE of EUR 20/MWh means an average absolute price error of EUR 20/MWh, not 20% and not a maximum error. High-price MAE is reported with its sample count. The threshold is calculated from training prices only; actual test prices identify the evaluation subset but never enter fitting. MAPE is omitted because zero and negative prices make it unreliable.

### Prediction export

`test_predictions.csv` and `metadata.json` are the interface to the contract project. Keep them together. A checksum guards against mismatched versions. Other outputs: `prediction_metrics.csv`, `data_quality.csv`, and saved XGBoost model JSON files.

| Column | Meaning |
|---|---|
| delivery_start_utc | Quarter-hour delivery timestamp |
| actual_da | Observed settlement price; evaluation only |
| pred_weather / pred_fundamentals | Held-out model forecasts |
| quote_mid | Training-only hypothetical counterparty midpoint |
| decision_time_utc | Assumed 10:00 Copenhagen time on preceding calendar day |
| duration_hours | 0.25; converts MW to MWh |
| eligible_for_contract | Decision strictly after conservative training-label availability |
| high_price | Actual price >= training quantile; evaluation only |

## Contract design

This is a hypothetical financially settled fixed-price contract on the DA price for one delivery interval, not a physical exchange DA order. No ID or balancing data are needed.

A buy executes at midpoint + half spread + adverse slippage. A sell executes at midpoint - half spread - adverse slippage. Trade only if forecast edge exceeds total per-MWh fee plus an extra buffer. Fees are charged once; automatic settlement does not create an additional closing trade.

Signed Q is positive for buys and negative for sells:

`net P&L = Q × (actual DA − execution price) − abs(Q) × fee`.

A 1 MW position for 15 minutes has Q = 0.25 MWh. There is no assumed physical energy inventory in this cash-settled simulation.

For each model separately, rank each day's opportunities by expected net edge. Apply per-interval power and daily gross-energy caps. Also cap daily gross volume by stress budget / assumed adverse price move. With defaults, EUR 500 / EUR 100/MWh = 5 MWh/day. This is a stress proxy, not a guaranteed loss limit or hindsight stop-loss. All daily positions are chosen together, before settlement.

The planner accepts only predictions, quotes, duration and decision timestamps. Settlement receives actual prices separately. Days at the split boundary whose decision predates training availability are excluded. This conservative check does not resolve the separate forecast-vintage issue.

### Example trade

Assume the weather model predicts EUR 120/MWh and the counterparty midpoint is EUR 100/MWh. With a full spread of EUR 4/MWh, the ask is EUR 102/MWh. Adding EUR 0.50/MWh slippage gives an execution price of EUR 102.50/MWh.

After the EUR 0.50/MWh fee, expected net edge is `120 − 102.50 − 0.50 = EUR 17/MWh`, above the default EUR 5/MWh buffer. If the risk allocation permits it, the simulator buys 1 MW for 15 minutes, or 0.25 MWh.

If actual DA settles at EUR 90/MWh:

```text
Net P&L = 0.25 × (90 − 102.50) − 0.25 × 0.50 = −EUR 3.25
```

Positive forecast edge does not guarantee a profitable outcome. The model prediction is not the contract execution price.

### Contract export

- `trade_ledger.csv`: every eligible interval for each strategy, including no-trades, forecast/quote/execution/settlement, quantities, costs, expected and realised hypothetical P&L, drawdown and stress sensitivities.
- `summary.csv`: total P&L, traded volume, win rate, worst day and maximum drawdown per alternative strategy.
- `daily.csv`: daily P&L, volumes, stress exposure and observed interval counts.
- `regimes.csv`: high-price versus other-period P&L and forecast errors.
- `assumptions.json`: configuration, data status, source checksum and explicit modelling assumptions.

## Interpretation and limitations

Lower MAE need not give higher contract P&L. Examine whether forecasts improve on the quoted benchmark, trade direction, price-spike misses, costs and exposure. Compare against zero P&L from not trading. Win rate excludes no-trades. Drawdown is based on delivery-time settled P&L, starts from zero, and is not an intraday mark-to-market or collateral measure.

Quotes, spreads, liquidity and full fills are assumed. There are no counterparty-default, collateral, funding or tax costs. A weak calendar benchmark can make results look unusually favourable. Net profit depends strongly on the quote assumption. Do not treat it as evidence of attainable trading returns.

Historical Open-Meteo data stitch successive weather runs; historical ENTSO-E data may include revisions. A strict pre-auction backtest requires archived publication/issue timestamps and vintages available at decision time. Current outputs explicitly say `forecast_vintages_verified: false`.

Dropping incomplete rows can bias scored intervals and leave incomplete delivery days. The short seasonal period is not a basis for annualisation. Preset risk/cost assumptions before evaluating; do not select the highest-profit test configuration. Tune using chronological validation inside the first 80%, then evaluate once on a fresh holdout.

## Troubleshooting

| Issue | Action |
|---|---|
| Notebook cannot import its script | Keep each notebook beside its matching `.py` file and open Jupyter in the project directory |
| Missing ENTSO-E token | Set `ENTSOE_API_TOKEN`, provide `../entsoe-api.txt`, or enable synthetic demo mode |
| Contract input files are missing | Run prediction first and use the matching `prediction_output` or `prediction_demo` folder |
| CSV checksum mismatch | Re-export the prediction bundle; keep its CSV and metadata together |
| Missing renewable forecast component | Check source coverage for that zone; do not silently replace missing data with zero |
| Unsupported price or forecast interval | Choose a period with 15-minute DA prices and supported forecast intervals |
| No contract-eligible intervals | Extend the sample so test decisions occur after the assumed training-completion timestamp |

## Verification

`test_project.py` checks long/short/flat P&L, costs and MWh conversion, negative prices, risk caps, missing-hour handling, DST decision time, checksum validation, and test-label isolation. Perturbing test settlement prices must not change either model's predictions, benchmark quotes or planned positions. Synthetic end-to-end execution exercises both scripts; live API calls require your credentials and have not been validated on your market data.

Sources: [Open-Meteo Historical Forecast](https://open-meteo.com/en/docs/historical-forecast-api), [Single Runs](https://open-meteo.com/en/docs/single-runs-api), [entsoe-py](https://github.com/EnergieID/entsoe-py).
