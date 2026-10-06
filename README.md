# SocialPulse

A research prototype for exploring youth unemployment, comparing historical country conditions, and evaluating early-warning models using World Bank Indicators API data.

## Features

- Country-level exploration of six socioeconomic indicators.
- Historical charts and missing-data information.
- Descriptive historical analogues with subsequent unemployment outcomes.
- Transparent development, backtest, and reserved-test results.

## Research question

Can information associated with a country-year identify a substantial increase in youth unemployment within the following two years?

The primary event is an increase of at least **3 percentage points** in either following year relative to the starting year. Both future years must be available to create a label.

## Data

The initial panel covers 217 countries/economies and 2000–2025. Coverage varies by indicator.

Indicators include youth unemployment, GDP growth, inflation, FDI, trade, and youth labor-force participation.

Source: [World Bank Indicators API](https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation).

The labor indicators used here are ILO modeled estimates distributed through World Bank. Download timestamps and indicator metadata are included with the demo snapshot.

Data remain subject to their source terms and attribution requirements; inclusion in this repository does not change their licensing. See [World Bank data licensing](https://datacatalog.worldbank.org/public-licenses) and the bundled indicator metadata.

## Modeling and evaluation

The project compares a constant training-prevalence baseline, logistic regression using unemployment history, and logistic regression using all 18 engineered features.

Preprocessing is fitted on training data only. Expanding-window backtests respect the two-year label horizon.

The selected history-only model was fitted on origin years 2002–2019 and evaluated without retraining on reserved origin years 2022–2023.

| Test year | Event rate | Model AP | Constant baseline AP | Positive cases in top 10 |
|---|---:|---:|---:|---:|
| 2022 | 3.85% | 0.0340 | 0.0385 | 0 |
| 2023 | 1.65% | 0.0307 | 0.0165 | 0 |

These results do **not** establish reliable operational early-warning performance. The dashboard presents a research prototype and does not assign validated risk categories.

## Historical analogues

The dashboard compares six standardized indicator levels against complete historical cases whose two-year outcomes end before the selected reference year.

It shows at most five distinct countries within an exploratory distance cutoff. Matching outcomes are descriptive, not forecast probabilities or causal explanations.

## Run the dashboard

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app/12_dashboard.py
```

The packaged data in `app/data` allow the dashboard to run without downloading data or training models.

## Reproduce the research pipeline

Run the numbered scripts in `src/socialpulse`.

- `01`–`02`: single-country API example.
- `03`–`05`: panel download, preparation and coverage audit.
- `06`: development-period target audit.
- `07`: features and event labels.
- `08`–`10`: baseline evaluation and historical backtests.
- `11`: reserved-test evaluation and model saving.
- `14`: package the prepared panel for the dashboard.

`docs/experiment_config.json` identifies the research snapshot and configuration. `docs/model_decision.json` records the selected final model.

A fresh API download creates a new snapshot and may contain revised historical values. To study it, update the configured snapshot and regenerate downstream artifacts as a new experiment. Original results are not guaranteed to reproduce from a later API release.

The final-test script intentionally refuses to overwrite existing final metrics. Do not repeatedly tune models against the reserved test.

## Limitations

- Evaluation uses a current data snapshot, not historical release vintages.
- Publication delays are not fully reconstructed.
- Modeled estimates are not all direct observations.
- Overlapping forecast windows and shared economic shocks create dependence.
- The reserved test contains few positive windows.
- Model probabilities are not established as well calibrated.
- Historical similarity does not imply the same future outcome.

## Repository structure

- `src/socialpulse`: research scripts.
- `app`: dashboard, analogue component and packaged demo data.
- `docs`: experiment configuration and model decision.
- `reports`: coverage and model-evaluation reports.
- `data`: local research data, excluded from Git by default.
- `models`: local fitted models, excluded from Git by default.