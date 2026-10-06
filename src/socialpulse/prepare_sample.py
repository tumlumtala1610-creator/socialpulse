"""Validate the Netherlands sample and export a year-by-year CSV."""

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

COUNTRY = "NLD"
INDICATOR = "SL.UEM.1524.ZS"
START_YEAR = 2000
END_YEAR = 2025


def main():
    # Filenames contain UTC timestamps, so sorting selects the latest.
    files = sorted(
        RAW_DIR.glob(f"{COUNTRY}_{INDICATOR}_*.json")
    )

    if not files:
        raise FileNotFoundError(
            "No Netherlands snapshot found. Run fetch_sample.py first."
        )

    input_path = files[-1]

    with input_path.open(encoding="utf-8") as file:
        snapshot = json.load(file)

    if (
        snapshot["country_code"] != COUNTRY
        or snapshot["indicator_code"] != INDICATOR
    ):
        raise ValueError("Snapshot country or indicator does not match.")

    records = snapshot["api_response"][1]
    by_year = {}

    for record in records:
        if record["countryiso3code"] != COUNTRY:
            raise ValueError("Unexpected country in the API response.")

        if record["indicator"]["id"] != INDICATOR:
            raise ValueError("Unexpected indicator in the API response.")

        year = int(record["date"])

        if year in by_year:
            raise ValueError(f"Duplicate observation for year {year}.")

        if not START_YEAR <= year <= END_YEAR:
            raise ValueError(f"Year outside the expected range: {year}.")

        value = record["value"]

        if value is not None:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= value <= 100
            ):
                raise ValueError(
                    f"Invalid unemployment percentage in {year}: {value}"
                )

        by_year[year] = value

    rows = []

    # Include every expected year, even if the API omitted a record.
    for year in range(START_YEAR, END_YEAR + 1):
        value = by_year.get(year)

        if year not in by_year:
            status = "missing_record"
        elif value is None:
            status = "missing_value"
        else:
            status = "available"

        rows.append({
            "country_code": COUNTRY,
            "year": year,
            "youth_unemployment_pct": value,
            "data_status": status,
            "retrieved_at_utc": snapshot["retrieved_at_utc"],
            "source_file": input_path.name,
        })

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / f"{input_path.stem}.csv"

    with output_path.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    available = [
        row for row in rows
        if row["data_status"] == "available"
    ]
    missing_years = [
        row["year"] for row in rows
        if row["data_status"] != "available"
    ]

    print(f"Source: {input_path.name}")
    print(f"Expected years: {START_YEAR}-{END_YEAR}")
    print(f"API records: {len(records)}")
    print(f"Years with values: {len(available)} / {len(rows)}")
    print(f"Missing years: {missing_years}")
    print("Duplicate years: 0")
    print("Value range check: passed")

    print("\nLatest five calendar years:")
    for row in rows[-5:]:
        value = row["youth_unemployment_pct"]
        display = "missing" if value is None else f"{value:.2f}%"
        print(f"{row['year']}: {display}")

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
