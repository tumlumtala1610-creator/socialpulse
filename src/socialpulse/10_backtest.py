"""Compare three models with expanding-window historical backtests."""

import csv
import json
import runpy
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
BACKTEST_YEARS = range(2010, 2020)


def main():
    with (ROOT / "docs" / "experiment_config.json").open(
        encoding="utf-8"
    ) as file:
        config = json.load(file)

    data_dir = ROOT / "data" / "processed" / config["source_run"]

    with (data_dir / "feature_schema.json").open(
        encoding="utf-8"
    ) as file:
        schema = json.load(file)

    if schema["config"] != config:
        raise ValueError("Configuration changed. Regenerate features first.")

    if max(BACKTEST_YEARS) >= config["test_years"][0]:
        raise ValueError("Backtest must not enter the reserved test period.")

    # Reuse the conversion function without running script 08's main().
    helpers = runpy.run_path(
        str(ROOT / "src" / "socialpulse" / "08_train_baselines.py")
    )
    feature_matrix = helpers["feature_matrix"]

    with (data_dir / "model_dataset.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        rows = [
            row for row in csv.DictReader(file)
            if int(row["year"]) <= max(BACKTEST_YEARS)
        ]

    seen = set()
    for row in rows:
        key = (row["country_code"], int(row["year"]))

        if key in seen:
            raise ValueError(f"Duplicate country-year: {key}")
        seen.add(key)

        if row["source_run"] != config["source_run"]:
            raise ValueError("Unexpected source run.")

        if row["target_primary"].strip() not in ("", "0", "1"):
            raise ValueError("Unexpected target value.")

    labeled = [
        row for row in rows
        if row["target_primary"].strip() != ""
    ]

    all_features = schema["feature_columns"]
    feature_sets = {
        "B1_history": [
            name for name in all_features
            if name.startswith("youth_unemployment_pct__")
        ],
        "M1_full": all_features,
    }

    metrics = []
    saved_predictions = []

    for year in BACKTEST_YEARS:
        # A training label needs t+2 to end before the forecast year.
        train_end = year - config["forecast_horizon_years"] - 1

        train = [
            row for row in labeled
            if config["train_years"][0] <= int(row["year"]) <= train_end
        ]
        evaluation = [
            row for row in labeled
            if int(row["year"]) == year
        ]

        if not train or not evaluation:
            raise ValueError(f"No usable data for backtest year {year}.")

        y_train = np.array([
            int(row["target_primary"]) for row in train
        ])
        y_eval = np.array([
            int(row["target_primary"]) for row in evaluation
        ])

        if len(np.unique(y_train)) != 2:
            raise ValueError(f"Train has only one class for year {year}.")

        predictions = {
            "B0_constant": np.full(len(evaluation), y_train.mean())
        }

        for name, columns in feature_sets.items():
            x_train = feature_matrix(train, columns)
            x_eval = feature_matrix(evaluation, columns)

            if np.isnan(x_train).all(axis=0).any():
                raise ValueError(
                    f"Entirely missing training feature: {name}, {year}"
                )

            if np.isinf(x_train).any() or np.isinf(x_eval).any():
                raise ValueError("Infinite feature value detected.")

            model = Pipeline([
                ("imputer", SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                )),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=3000,
                )),
            ])

            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(x_train, y_train)

            predictions[name] = model.predict_proba(x_eval)[:, 1]

        print(
            f"\n{year} | Train through {train_end} | "
            f"Train n={len(train)} | Eval n={len(evaluation)} | "
            f"Event rate={y_eval.mean():.2%}"
        )

        for name, probabilities in predictions.items():
            ap = (
                float(average_precision_score(y_eval, probabilities))
                if len(np.unique(y_eval)) == 2 else None
            )

            result = {
                "origin_year": year,
                "model": name,
                "train_end_year": train_end,
                "train_rows": len(train),
                "evaluation_rows": len(evaluation),
                "event_rate": float(y_eval.mean()),
                "average_precision": ap,
                "brier_score": float(
                    brier_score_loss(y_eval, probabilities)
                ),
                "log_loss": float(
                    log_loss(y_eval, probabilities, labels=[0, 1])
                ),
                "mean_probability": float(probabilities.mean()),
            }
            metrics.append(result)

            ap_text = f"{ap:.4f}" if ap is not None else "N/A"
            print(
                f"  {name}: AP={ap_text}, "
                f"Brier={result['brier_score']:.4f}, "
                f"Mean probability={probabilities.mean():.2%}"
            )

        for index, row in enumerate(evaluation):
            saved_predictions.append({
                "country_code": row["country_code"],
                "origin_year": year,
                "target_primary": int(y_eval[index]),
                **{
                    name: float(values[index])
                    for name, values in predictions.items()
                },
            })

    report_dir = (
        ROOT / "reports" / config["source_run"] / "backtest"
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    for filename, records in (
        ("metrics_by_year.csv", metrics),
        ("predictions.csv", saved_predictions),
    ):
        with (report_dir / filename).open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    protocol = {
        "config": config,
        "backtest_years": list(BACKTEST_YEARS),
        "training_rule": "origin year <= prediction year - horizon - 1",
        "data_limitation": "Retrospective backtest using current data vintage",
        "test_evaluated": False,
    }
    with (report_dir / "protocol.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(protocol, file, ensure_ascii=False, indent=2)

    print("\nSUMMARY: NUMBER OF YEARS BETTER THAN B0")
    for name in feature_sets:
        ap_wins = 0
        ap_years = 0
        brier_wins = 0

        for year in BACKTEST_YEARS:
            baseline = next(
                row for row in metrics
                if row["origin_year"] == year
                and row["model"] == "B0_constant"
            )
            candidate = next(
                row for row in metrics
                if row["origin_year"] == year
                and row["model"] == name
            )

            if candidate["average_precision"] is not None:
                ap_years += 1
                ap_wins += (
                    candidate["average_precision"]
                    > baseline["average_precision"]
                )

            brier_wins += (
                candidate["brier_score"] < baseline["brier_score"]
            )

        print(
            f"{name}: AP better in {ap_wins}/{ap_years} years; "
            f"Brier better in {brier_wins}/{len(BACKTEST_YEARS)} years"
        )

    print(f"\nSaved to: {report_dir}")
    print("Reserved test period was not evaluated.")


if __name__ == "__main__":
    main()
