"""Create past-only features and future event labels."""

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def split_for_year(year, config):
    for name in ("train", "validation", "test"):
        start, end = config[f"{name}_years"]
        if start <= year <= end:
            return name

    if year > config["test_years"][1]:
        return "future_unlabeled"

    return "gap"


def make_label(current, next_year, second_year, threshold):
    if any(
        value is None
        for value in (current, next_year, second_year)
    ):
        return None

    increase = max(next_year, second_year) - current
    return int(increase >= threshold)


def main():
    config = read_json(
        PROJECT_ROOT / "docs" / "experiment_config.json"
    )

    if config["forecast_horizon_years"] != 2:
        raise ValueError("This implementation supports a two-year horizon.")

    run_id = config["source_run"]
    indicators = config["indicators"]
    target_indicator = "youth_unemployment_pct"

    manifest = read_json(
        PROJECT_ROOT / "data" / "raw" / run_id / "manifest.json"
    )

    if manifest["status"] != "complete":
        raise ValueError("The configured download is not complete.")

    input_path = (
        PROJECT_ROOT / "data" / "processed"
        / run_id / "country_year_panel.csv"
    )

    with input_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"country_code", "year", "source_run", *indicators}

        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("Input panel is missing required columns.")

        raw_rows = list(reader)

    lookup = {}
    countries = set()

    for row in raw_rows:
        if row["source_run"] != run_id:
            raise ValueError("Input data has a different source run.")

        key = (row["country_code"], int(row["year"]))

        if key in lookup:
            raise ValueError(f"Duplicate country-year: {key}")

        lookup[key] = {
            name: float(row[name]) if row[name].strip() else None
            for name in indicators
        }
        countries.add(key[0])

    def value_at(country, year, indicator):
        return lookup.get((country, year), {}).get(indicator)

    feature_columns = [
        f"{name}__{suffix}"
        for name in indicators
        for suffix in ("level", "change_1y", "trend_3y")
    ]

    output_rows = []

    for country in sorted(countries):
        for year in range(
            config["feature_start_year"],
            config["feature_end_year"] + 1,
        ):
            output = {
                "country_code": country,
                "year": year,
                "split": split_for_year(year, config),
                "source_run": run_id,
            }

            # Features use only year t and earlier years.
            for name in indicators:
                current = value_at(country, year, name)
                lag1 = value_at(country, year - 1, name)
                lag2 = value_at(country, year - 2, name)

                output[f"{name}__level"] = current

                output[f"{name}__change_1y"] = (
                    current - lag1
                    if current is not None and lag1 is not None
                    else None
                )

                # Slope across t-2, t-1, t, requiring all three values.
                output[f"{name}__trend_3y"] = (
                    (current - lag2) / 2
                    if all(v is not None for v in (current, lag1, lag2))
                    else None
                )

            current_u = value_at(country, year, target_indicator)
            next_u = value_at(country, year + 1, target_indicator)
            second_u = value_at(country, year + 2, target_indicator)

            # Future values are used only for labels, never features.
            output["target_primary"] = make_label(
                current_u,
                next_u,
                second_u,
                config["primary_threshold_pp"],
            )
            output["target_sensitivity"] = make_label(
                current_u,
                next_u,
                second_u,
                config["sensitivity_threshold_pp"],
            )

            output_rows.append(output)

    output_dir = PROJECT_ROOT / "data" / "processed" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "model_dataset.csv"

    with output_path.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(output_rows[0])
        )
        writer.writeheader()
        writer.writerows(output_rows)

    # This explicit list will be used to select model inputs.
    schema = {
        "source_run": run_id,
        "config": config,
        "feature_columns": feature_columns,
        "target_columns": ["target_primary", "target_sensitivity"],
        "missing_values": "Preserved; no imputation performed",
    }

    schema_path = output_dir / "feature_schema.json"
    with schema_path.open("w", encoding="utf-8") as file:
        json.dump(schema, file, ensure_ascii=False, indent=2)

    print(f"Source run: {run_id}")
    print(f"Features: {len(feature_columns)}")
    print(f"Total rows: {len(output_rows)}")

    for split in (
        "train", "validation", "test", "gap", "future_unlabeled"
    ):
        subset = [row for row in output_rows if row["split"] == split]
        labeled = [
            row for row in subset if row["target_primary"] is not None
        ]

        print(
            f"{split}: {len(subset)} rows, "
            f"{len(labeled)} with complete target"
        )

        # Do not inspect validation/test event rates at this stage.
        if split == "train" and labeled:
            positives = sum(row["target_primary"] for row in labeled)
            print(
                f"  Train positive rate: "
                f"{100 * positives / len(labeled):.2f}%"
            )

    print(f"\nDataset saved to: {output_path}")
    print(f"Feature schema saved to: {schema_path}")


if __name__ == "__main__":
    main()
