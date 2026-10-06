"""Compare baseline models using train and validation only."""

import csv
import json
import platform
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def feature_matrix(rows, columns):
    return np.array([
        [
            float(row[column])
            if row[column].strip()
            else np.nan
            for column in columns
        ]
        for row in rows
    ], dtype=float)


def evaluate(name, y_true, probabilities):
    return {
        "model": name,
        "validation_rows": len(y_true),
        "validation_event_rate": float(y_true.mean()),
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),
        "log_loss": float(
            log_loss(y_true, probabilities, labels=[0, 1])
        ),
        "mean_predicted_probability": float(probabilities.mean()),
    }


def main():
    config = read_json(
        PROJECT_ROOT / "docs" / "experiment_config.json"
    )

    run_id = config["source_run"]
    data_dir = PROJECT_ROOT / "data" / "processed" / run_id

    schema = read_json(data_dir / "feature_schema.json")

    if schema["config"] != config:
        raise ValueError(
            "Configuration changed. Regenerate features first."
        )

    all_features = schema["feature_columns"]

    expected_features = [
        f"{indicator}__{suffix}"
        for indicator in config["indicators"]
        for suffix in ("level", "change_1y", "trend_3y")
    ]

    if all_features != expected_features:
        raise ValueError("Unexpected feature list.")

    with (data_dir / "model_dataset.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        # Test rows are excluded before labels/features are converted.
        rows = [
            row for row in csv.DictReader(file)
            if row["split"] in ("train", "validation")
        ]

    train_rows = []
    validation_rows = []
    seen = set()

    for row in rows:
        year = int(row["year"])
        key = (row["country_code"], year)

        if key in seen:
            raise ValueError(f"Duplicate country-year: {key}")
        seen.add(key)

        if row["source_run"] != run_id:
            raise ValueError("Unexpected source run.")

        start, end = config[f"{row['split']}_years"]
        if not start <= year <= end:
            raise ValueError("Row does not match configured time split.")

        if row["target_primary"].strip() == "":
            continue

        if row["target_primary"] not in ("0", "1"):
            raise ValueError("Unexpected target value.")

        if row["split"] == "train":
            train_rows.append(row)
        else:
            validation_rows.append(row)

    if not train_rows or not validation_rows:
        raise ValueError("Train or validation has no labeled rows.")

    y_train = np.array([
        int(row["target_primary"]) for row in train_rows
    ])
    y_validation = np.array([
        int(row["target_primary"]) for row in validation_rows
    ])

    if len(np.unique(y_train)) != 2:
        raise ValueError("Train must contain both target classes.")

    if len(np.unique(y_validation)) != 2:
        raise ValueError(
            "Validation has only one class; review the evaluation design."
        )

    print(f"Train rows: {len(train_rows)}")
    print(f"Validation rows: {len(validation_rows)}")
    print(f"Train event rate: {y_train.mean():.2%}")
    print(f"Validation event rate: {y_validation.mean():.2%}")

    predictions = {
        "B0_constant": np.full(
            len(validation_rows),
            y_train.mean(),
            dtype=float,
        )
    }

    model_features = {
        "B1_unemployment_history": [
            name for name in all_features
            if name.startswith("youth_unemployment_pct__")
        ],
        "M1_all_features": all_features,
    }

    for name, columns in model_features.items():
        print(f"\nTraining {name} with {len(columns)} input features...")

        x_train = feature_matrix(train_rows, columns)
        x_validation = feature_matrix(validation_rows, columns)

        if np.isinf(x_train).any() or np.isinf(x_validation).any():
            raise ValueError("Infinite feature values detected.")

        if np.isnan(x_train).all(axis=0).any():
            raise ValueError("A feature is entirely missing in train.")

        model = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=3000,
                ),
            ),
        ])

        # Stop if optimization fails to converge.
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(x_train, y_train)

        predictions[name] = model.predict_proba(x_validation)[:, 1]

    metrics = [
        evaluate(name, y_validation, probabilities)
        for name, probabilities in predictions.items()
    ]

    prediction_rows = []

    for index, row in enumerate(validation_rows):
        output = {
            "country_code": row["country_code"],
            "year": int(row["year"]),
            "target_primary": int(y_validation[index]),
        }

        for name, probabilities in predictions.items():
            output[f"probability_{name}"] = float(probabilities[index])

        prediction_rows.append(output)

    report_dir = PROJECT_ROOT / "reports" / run_id / "baselines"
    report_dir.mkdir(parents=True, exist_ok=True)

    write_csv(report_dir / "validation_metrics.csv", metrics)
    write_csv(
        report_dir / "validation_predictions.csv",
        prediction_rows,
    )

    experiment = {
        "config": config,
        "model_features": model_features,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "imputation": "Training medians with missing-value indicators",
        "scaling": "StandardScaler fitted on train",
        "logistic_regression": {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 3000,
        },
        "test_evaluated": False,
    }

    with (report_dir / "experiment.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(experiment, file, ensure_ascii=False, indent=2)

    print("\nVALIDATION RESULTS")
    for row in metrics:
        print(
            f"{row['model']}: "
            f"AP={row['average_precision']:.4f}, "
            f"Brier={row['brier_score']:.4f}, "
            f"LogLoss={row['log_loss']:.4f}, "
            f"Mean probability={row['mean_predicted_probability']:.4f}"
        )

    print(f"\nReports saved to: {report_dir}")
    print("Test set was not evaluated.")


if __name__ == "__main__":
    main()
