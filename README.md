# SocialPulse

An early-warning research project for sharp increases in youth
unemployment using World Bank Indicators API data.

## Research question

Can information available at a prediction date identify countries
at risk of a sharp increase in youth unemployment within the next
two years, better than simple baseline methods?

## Scope

- Unit of analysis: country-year.
- Forecast horizon: two years.
- Primary outcome: a threshold increase in youth unemployment
  in either of the following two years.
- The event threshold will be selected after training-data analysis.
- Historical analogues will provide context for model predictions.

## Data source

World Bank Indicators API.

Candidate target indicator:
SL.UEM.1524.ZS — Youth unemployment, total, modeled ILO estimate.

Country coverage, time coverage, and predictor availability
will be checked before modeling.

## Planned workflow

1. Retrieve and version API data and metadata.
2. Audit coverage, missing values, and data quality.
3. Build country-year features and future event labels.
4. Evaluate simple baselines with time-based backtesting.
5. Compare candidate models and assess probability calibration.
6. Add model explanations and historical analogues.
7. Build an interactive country analysis interface.

## Project structure

- src/socialpulse/: data and modeling code
- data/raw/: original API responses
- data/processed/: prepared datasets
- notebooks/: exploratory analysis
- reports/: data audits and evaluation results
- models/: trained model artifacts
- app/: application interface
- docs/: project plan and methodology
- tests/: checks for critical pipeline logic

## Current status

Project setup. No trained model or validated predictions yet.

## Reproducibility

Setup and execution instructions will be added as the pipeline
is implemented. Downloaded datasets and model artifacts are
excluded from Git by default.

## Limitations

This project estimates the risk of a defined unemployment increase.
It does not measure every aspect of youth employment conditions
or establish causal effects.