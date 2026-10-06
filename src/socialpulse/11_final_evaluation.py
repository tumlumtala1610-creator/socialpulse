"""Evaluate the selected B1 model on the reserved test period."""

import csv
import json
import pickle
import platform
import runpy
import warnings
from pathlib import Path

import numpy as np
import sklearn
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
TOP_K = 10


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    with (ROOT / "docs" / "experiment_config.json").open(
        encoding="utf-8"
    ) as file:
        config = json.load(file)

    run_id = config["source_run"]
    data_dir = ROOT / "data" / "processed" / run_id
    report_dir = ROOT / "reports" / run_id / "final_test"
    model_dir = ROOT / "models" / run_id

    if (report_dir / "test_metrics.csv").exists():
        raise RuntimeError(
            "Final test results already exist. Review them without rerunning."
        )

    with (data_dir / "feature_schema.json").open(
        encoding="utf-8"
    ) as file:
        schema = json.load(file)

    if schema["config"] != config:
        raise ValueError("Feature configuration does not match.")

    features = [
        "youth_unemployment_pct__level",
        "youth_unemployment_pct__change_1y",
        "youth_unemployment_pct__trend_3y",
    ]

    if not set(features).issubset(schema["feature_columns"]):
        raise ValueError("Required features are missing.")

    train_start = config["train_years"][0]
    test_start, test_end = config["test_years"]
    train_end = test_start - config["forecast_horizon_years"] - 1

    decision = {
        "source_run": run_id,
        "selected_model": "B1_history",
        "features": features,
        "primary_threshold_pp": config["primary_threshold_pp"],
        "forecast_horizon_years": config["forecast_horizon_years"],
        "training_origin_years": [train_start, train_end],
        "test_origin_years": [test_start, test_end],
        "top_k": TOP_K,
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 3000,
        "probabilities_calibrated": False,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "limitation": "Retrospective evaluation using current data vintage",
    }

    # Record the decision before computing test results.
    decision_path = ROOT / "docs" / "model_decision.json"
    with decision_path.open("w", encoding="utf-8") as file:
        json.dump(decision, file, ensure_ascii=False, indent=2)

    with (data_dir / "model_dataset.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    seen = set()
    for row in rows:
        key = (row["country_code"], int(row["year"]))
        if key in seen:
            raise ValueError(f"Duplicate country-year: {key}")
        seen.add(key)

        if row["source_run"] != run_id:
            raise ValueError("Unexpected source run.")
        if row["target_primary"].strip() not in ("", "0", "1"):
            raise ValueError("Invalid target.")

    train = [
        row for row in rows
        if train_start <= int(row["year"]) <= train_end
        and row["target_primary"].strip() != ""
    ]

    test = [
        row for row in rows
        if test_start <= int(row["year"]) <= test_end
        and row["target_primary"].strip() != ""
    ]

    if not train or not test:
        raise ValueError("No eligible training or test rows.")

    helpers = runpy.run_path(
        str(ROOT / "src" / "socialpulse" / "08_train_baselines.py")
    )
    feature_matrix = helpers["feature_matrix"]

    x_train = feature_matrix(train, features)
    x_test = feature_matrix(test, features)
    y_train = np.array([int(row["target_primary"]) for row in train])

    if len(np.unique(y_train)) != 2:
        raise ValueError("Training requires both target classes.")
    if np.isnan(x_train).all(axis=0).any():
        raise ValueError("A feature is entirely missing in training.")
    if np.isinf(x_train).any() or np.isinf(x_test).any():
        raise ValueError("Infinite feature value.")

    model = Pipeline([
        ("imputer", SimpleImputer(
            strategy="median",
            add_indicator=True,
            keep_empty_features=True,
        )),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=3000,
        )),
    ])

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    baseline_probability = float(y_train.mean())

    predictions = [
        {
            "country_code": row["country_code"],
            "origin_year": int(row["year"]),
            "target_primary": int(row["target_primary"]),
            "probability": float(probabilities[index]),
        }
        for index, row in enumerate(test)
    ]

    metrics = []
    top_countries = []

    print(f"Training origins: {train_start}-{train_end}")
    print(f"Training rows: {len(train)}")
    print(f"Training event rate: {baseline_probability:.2%}")

    for year in range(test_start, test_end + 1):
        subset = [
            row for row in predictions if row["origin_year"] == year
        ]
        if not subset:
            raise ValueError(f"No eligible test rows for {year}.")

        actual = np.array([row["target_primary"] for row in subset])
        predicted = np.array([row["probability"] for row in subset])

        print(
            f"\nTEST {year}: n={len(subset)}, event rate={actual.mean():.2%}")

        for name, scores in (
            ("B0_constant", np.full(len(subset), baseline_probability)),
            ("B1_history", predicted),
        ):
            ap = (
                float(average_precision_score(actual, scores))
                if len(np.unique(actual)) == 2 else None
            )
            result = {
                "year": year,
                "model": name,
                "rows": len(subset),
                "event_rate": float(actual.mean()),
                "average_precision": ap,
                "brier_score": float(brier_score_loss(actual, scores)),
                "log_loss": float(log_loss(actual, scores, labels=[0, 1])),
                "mean_probability": float(scores.mean()),
            }
            metrics.append(result)

            ap_text = f"{ap:.4f}" if ap is not None else "N/A"
            print(
                f"  {name}: AP={ap_text}, "
                f"Brier={result['brier_score']:.4f}, "
                f"Mean probability={scores.mean():.2%}"
            )

        ranked = sorted(
            subset,
            key=lambda row: (-row["probability"], row["country_code"]),
        )
        selected = ranked[:TOP_K]
        hits = sum(row["target_primary"] for row in selected)
        positives = int(actual.sum())
        recall = hits / positives if positives else None

        recall_text = f"{recall:.2%}" if recall is not None else "N/A"
        print(
            f"  Top {len(selected)}: {hits} positive windows, "
            f"precision={hits / len(selected):.2%}, recall={recall_text}"
        )

        for rank, row in enumerate(selected, start=1):
            top_countries.append({**row, "rank": rank})

    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    write_csv(report_dir / "test_predictions.csv", predictions)
    write_csv(report_dir / "top10_by_year.csv", top_countries)

    # Save the fitted preprocessing and classifier together.
    with (model_dir / "b1_history.pkl").open("wb") as file:
        pickle.dump({
            "pipeline": model,
            "features": features,
            "decision": decision,
        }, file)

    write_csv(report_dir / "test_metrics.csv", metrics)

    print(f"\nReports: {report_dir}")
    print(f"Saved model: {model_dir / 'b1_history.pkl'}")


if __name__ == "__main__":
    main()
