# Ontario Electricity Demand Forecasting

[![tests](https://github.com/zekiyerdem/ontario-demand-forecast/actions/workflows/tests.yml/badge.svg)](https://github.com/zekiyerdem/ontario-demand-forecast/actions/workflows/tests.yml)

Day-ahead forecasting of Ontario's hourly electricity demand using public IESO data (2019–2026) and hourly weather.

**Result on the 2025 test year:** a LightGBM model with calendar and weather features forecasts next-day hourly demand with **2.6% MAPE (448 MW MAE)**, **41% lower error** than the "same hour yesterday" baseline, and predicts each day's peak within **2.8%** on average.

![Forecast vs actual, winter and summer weeks of 2025](figures/forecast_weeks_2025.png)

## Results — 2025 test year (8,760 hours, never used in training)

| Model | MAE (MW) | RMSE (MW) | MAPE (%) | Daily peak MAPE (%) |
|---|---:|---:|---:|---:|
| Persistence (same hour yesterday) | 758 | 1,030 | 4.6 | 3.8 |
| Weekly persistence (same hour last week) | 1,197 | 1,673 | 7.0 | 6.6 |
| Ridge regression (calendar + weather + lags) | 576 | 719 | 3.5 | 3.1 |
| LightGBM, no weather features | 638 | 876 | 3.7 | 3.9 |
| **LightGBM, with weather features** | **448** | **583** | **2.6** | **2.8** |

Setup: at the end of day D-1, forecast all 24 hours of day D. Train 2019–2023, tune on 2024, refit on 2019–2024, test on 2025.

## Key findings

- **Demand is U-shaped in temperature.** Lowest around a 12–14 °C daily mean, rising with heating load in winter and steeply with cooling load above ~25 °C. A linear correlation (r ≈ 0.02) misses this entirely.

  ![Demand vs temperature](figures/demand_vs_temperature.png)

- **Weather is what makes the model useful.** Removing weather features raises MAPE from 2.6% to 3.7%, and the daily-peak forecast becomes no better than persistence.
- **Summer peaks are the hardest.** June–July errors are highest (~3.0–3.3% MAPE). The model under-forecasts the afternoon peak on the hottest days, which are rare in the training data.
- **Load is growing.** Average 2025 demand was ~4% above 2024 (16,621 vs 15,986 MW), the largest year-over-year increase since 2019.
- **Seasonal shape differs.** Winter weekdays have morning and ~18:00 evening peaks; summer has one broad afternoon cooling peak.

  ![Seasonal daily profile](figures/seasonal_daily_profile.png)

## What drives the forecast (SHAP, 2025 test year)

SHAP splits every hourly forecast into additive contributions in MW from each feature (`04_shap_explainability.ipynb`).

![SHAP summary, 2025](figures/shap_summary_2025.png)

- **Yesterday's demand is the backbone.** Same-hour-yesterday demand (`lag_24`) moves the forecast by 1,176 MW on average, about three times the next feature (hour of day, 375 MW). Recent-demand features account for 47% of the total SHAP contribution, weather 29%, calendar 24%.
- **The temperature response is U-shaped and lopsided.** The temperature contribution bottoms out near a 12 °C daily mean (−949 MW). The coldest 2025 day (−13.8 °C) adds +933 MW; the hottest (30.6 °C) adds +3,235 MW. Days above 25 °C average +2,037 MW against +860 MW below −10 °C: cooling load moves Ontario demand far more than heating load.
- **Weather corrects persistence.** A warm previous day pushes the forecast down while a warm forecast day pushes it up, so the model effectively swaps yesterday's temperature effect (already inside `lag_24`) for today's. That is how it anticipates cold snaps and heat waves that persistence misses.

  ![Temperature effect learned by the model](figures/shap_temperature_2025.png)

## Where the model misses (error analysis, 2025)

Breaking the 2025 error down by hour, day type and sign (`05_error_analysis.ipynb`):

![Error by hour of day and day type, 2025](figures/error_by_hour_daytype_2025.png)

| Day type | Days | LightGBM MAPE (%) | Persistence MAPE (%) | Bias (MW) |
|---|---:|---:|---:|---:|
| Regular weekday | 246 | 2.40 | 4.31 | −240 |
| Weekend | 102 | 3.07 | 5.07 | −400 |
| Statutory holiday | 9 | 3.53 | 5.02 | −386 |
| Day after a holiday | 8 | 3.37 | 6.17 | −558 |

- **Midday is hardest.** MAPE is 2.30% overnight (22:00–04:59) and 3.14% for 10:00–15:59, rising to 3.55% for those hours in June–August.
- **Weekends and holidays are harder than weekdays.** After a holiday, persistence breaks down (6.17%) because yesterday was a low-demand day; the calendar features keep the model at 3.37%. Boxing Day was the worst day of 2025 (9.17% MAPE, 1,657 MW under-forecast).
- **The model under-forecasts 2025 across the board.** Mean error is −295 MW, 76% of hours are under-forecast, and every month has a negative bias. Demand grew 4.0% over 2024, and tree models cannot predict above the levels they were trained on. As a diagnostic, removing that offset would cut MAPE from 2.63% to 2.21%, so handling load growth is the largest single improvement available.
- **The worst days are hot weekends.** Of the 10 worst days, 9 are under-forecast, 8 are weekends or holidays and 5 have a daily mean of 23 °C or more.

## Notebooks

| Notebook | What it covers |
|---|---|
| `01_exploration.ipynb` | Loading 2025 IESO data, timestamp handling, daily/weekly profiles, winter peak analysis |
| `02_weather_and_baselines.ipynb` | Demand–temperature relationship, seasonal profiles, annual trend, persistence baselines |
| `03_day_ahead_models.ipynb` | Ridge and LightGBM day-ahead models, weather ablation, 2025 test, error by month, feature importance |
| `04_shap_explainability.ipynb` | SHAP explanation of the LightGBM model on 2025: top drivers, lags vs weather vs calendar, learned temperature curve |
| `05_error_analysis.ipynb` | Error by hour of day, weekday/weekend, holidays and the day after; forecast bias and load growth; the 10 worst days of 2025 |

## Data

- **Demand:** [IESO public hourly demand reports](https://reports-public.ieso.ca/public/Demand/). IESO reports in Eastern Standard Time all year (no daylight saving).
- **Weather:** hourly Toronto temperature, humidity and cloud cover from the [Open-Meteo historical archive](https://open-meteo.com/), pulled on the same fixed UTC-5 clock so it lines up with IESO hours.
- One missing hour (2025-05-01 00:00) is linearly interpolated.

**Assumption:** the models use *observed* weather for the forecast day as a stand-in for a day-ahead weather forecast, so real operational accuracy would be somewhat lower.

## Project structure

```
src/data.py        download + merge IESO demand and weather (cached to data/)
src/features.py    calendar, holiday, heating/cooling-degree and lag features
tests/             pytest unit tests for the feature pipeline (incl. a no-leakage check)
figures/           charts used in this README
results/           baseline and test-set metric tables (CSV)
```

## Run it

```bash
pip install -r requirements.txt
python -m src.data                 # downloads data into data/
jupyter notebook                   # run notebooks 02 to 05
python -m pytest                   # unit tests (no data download needed)
```

The tests run on small synthetic series. Besides checking lags, degree-days and holiday flags, they verify there is no leakage: changing demand on the forecast day never changes that day's features. GitHub Actions runs them on every push (Python 3.11 and 3.12).

## Next steps

- Handle load growth (trend term or year-over-year target) to remove the 2025 under-forecast bias
- Archived weather *forecasts* instead of observed weather
- Population-weighted temperature across Ontario cities; humidex for summer peaks
- Probabilistic (quantile) forecasts for peak-risk planning
- SARIMAX and LSTM comparisons

**Stack:** Python, pandas, scikit-learn, LightGBM, SHAP, matplotlib
