
# Ontario Electricity Demand Forecasting

Forecasting Ontario's hourly electricity demand using public IESO data (2002–present).

**Status: work in progress.** Data exploration and load characterization complete —
daily/weekly/seasonal demand structure, peak analysis (Ontario winter demand peaks
~18:00 at ≈20,400 MW average in January). Next: baseline models, then ARIMA and LSTM.

**Stack:** Python, pandas, matplotlib · Data: [IESO public reports](https://reports-public.ieso.ca/public/Demand/)

**Notebooks:**
- `01_exploration` — loading, timestamp engineering, demand profiles, peak analysis
