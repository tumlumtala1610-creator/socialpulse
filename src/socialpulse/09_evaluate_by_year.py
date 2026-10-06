"""Evaluate saved validation predictions separately by origin year."""

import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS = [
    "B0_constant",
    "B1_unemployment_history",
    "M1_all_features",
]


def main():
    config_path = PROJECT_ROOT / "docs" / "experiment_config.json"

    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)

    report_dir = (
        PROJECT_ROOT / "reports"
        / config["source_run"] / "baselines"
    )

    # Ensure predictions were generated with the current configuration.
    with (report_dir / "experiment.json").open(
        encoding="utf-8"
    ) as file:
        experiment = json.load(file)

    if experiment["config"] != config:
        raise ValueError(
            "Configuration changed. Run train_baselines.py again."
        )

    input_path = report_dir / "validation_predictions.csv"

    with input_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("No validation predictions found.")

    start, end = config["validation_years"]
    seen = set()

    for row in rows:
        year = int(row["year"])
        key = (row["country_code"], year)

        if not start <= year <= end:
            raise ValueError("A row is outside the validation period.")

        if key in seen:
            raise ValueError(f"Duplicate prediction: {key}")
        seen.add(key)

    results = []

    for year in range(start, end + 1):
        subset = [
            row for row in rows
            if int(row["year"]) == year
        ]

        if not subset:
            raise ValueError(f"No predictions for validation year {year}.")

        y_true = np.array([
            int(row["target_primary"]) for row in subset
        ])

        if not np.isin(y_true, [0, 1]).all():
            raise ValueError("Target must be 0 or 1.")

        event_rate = float(y_true.mean())
        has_both_classes = len(np.unique(y_true)) == 2

        print(
            f"\nYEAR {year}: "
            f"{len(subset)} rows, "
            f"{int(y_true.sum())} positive windows, "
            f"event rate={event_rate:.2%}"
        )

        for model in MODELS:
            probabilities = np.array([
                float(row[f"probability_{model}"])
                for row in subset
            ])

            if (
                not np.isfinite(probabilities).all()
                or (probabilities < 0).any()
                or (probabilities > 1).any()
            ):
                raise ValueError(f"Invalid probabilities: {model}, {year}")

            ap = (
                float(average_precision_score(y_true, probabilities))
                if has_both_classes else None
            )

            result = {
                "origin_year": year,
                "model": model,
                "rows": len(subset),
                "positive_windows": int(y_true.sum()),
                "event_rate": event_rate,
                "average_precision": ap,
                "brier_score": float(
                    brier_score_loss(y_true, probabilities)
                ),
                "log_loss": float(
                    log_loss(y_true, probabilities, labels=[0, 1])
                ),
                "mean_probability": float(probabilities.mean()),
            }
            results.append(result)

            ap_text = f"{ap:.4f}" if ap is not None else "not applicable"

            print(
                f"  {model}: "
                f"AP={ap_text}, "
                f"Brier={result['brier_score']:.4f}, "
                f"LogLoss={result['log_loss']:.4f}, "
                f"Mean probability={result['mean_probability']:.2%}"
            )

    output_path = report_dir / "validation_metrics_by_year.csv"

    with output_path.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved to: {output_path}")
    print("No models were retrained. Test set was not evaluated.")


if __name__ == "__main__":
    main()
