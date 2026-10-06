"""Inspect candidate event thresholds using an early development period."""

import csv
from pathlib import Path
from statistics import median


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET_COLUMN = "youth_unemployment_pct"

# Only this early period is used to inspect candidate thresholds.
DEV_START = 2002
DEV_END = 2014

# Percentage-point increases, not relative percentage changes.
THRESHOLDS = [1.0, 2.0, 3.0, 4.0, 5.0]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    files = sorted(
        PROCESSED_DIR.glob("panel_*/country_year_panel.csv")
    )

    if not files:
        raise FileNotFoundError("Run build_panel.py first.")

    input_path = files[-1]

    with input_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    values = {}
    countries = set()

    for row in rows:
        country = row["country_code"]
        year = int(row["year"])
        key = (country, year)

        if key in values:
            raise ValueError(f"Duplicate country-year: {key}")

        raw_value = row[TARGET_COLUMN].strip()
        values[key] = float(raw_value) if raw_value else None
        countries.add(country)

    events = []
    incomplete = 0

    for country in sorted(countries):
        for year in range(DEV_START, DEV_END + 1):
            current = values.get((country, year))
            next_year = values.get((country, year + 1))
            second_year = values.get((country, year + 2))

            if any(
                value is None
                for value in (current, next_year, second_year)
            ):
                incomplete += 1
                continue

            max_increase = max(next_year, second_year) - current

            events.append({
                "country_code": country,
                "origin_year": year,
                "unemployment_t": current,
                "unemployment_t1": next_year,
                "unemployment_t2": second_year,
                "max_increase_pp": max_increase,
            })

    if not events:
        raise ValueError("No complete target windows in development data.")

    summary = []

    for threshold in THRESHOLDS:
        positives = [
            row for row in events
            if row["max_increase_pp"] >= threshold
        ]

        summary.append({
            "threshold_pp": threshold,
            "eligible_windows": len(events),
            "positive_windows": len(positives),
            "event_rate_pct": round(
                100 * len(positives) / len(events), 2
            ),
            "countries_with_events": len({
                row["country_code"] for row in positives
            }),
        })

    by_year = []

    for year in range(DEV_START, DEV_END + 1):
        subset = [row for row in events if row["origin_year"] == year]

        for threshold in THRESHOLDS:
            positive_count = sum(
                row["max_increase_pp"] >= threshold
                for row in subset
            )

            by_year.append({
                "origin_year": year,
                "threshold_pp": threshold,
                "eligible_windows": len(subset),
                "positive_windows": positive_count,
                "event_rate_pct": (
                    round(100 * positive_count / len(subset), 2)
                    if subset else None
                ),
            })

    report_dir = REPORTS_DIR / input_path.parent.name
    report_dir.mkdir(parents=True, exist_ok=True)

    write_csv(report_dir / "target_thresholds_development.csv", summary)
    write_csv(report_dir / "target_by_year_development.csv", by_year)

    increases = [row["max_increase_pp"] for row in events]

    print(f"Source run: {input_path.parent.name}")
    print(f"Development origin years: {DEV_START}-{DEV_END}")
    print(f"Latest outcome year used: {DEV_END + 2}")
    print(f"Eligible two-year windows: {len(events)}")
    print(f"Incomplete windows excluded: {incomplete}")
    print(f"Median maximum increase: {median(increases):.2f} pp")

    print("\nCandidate thresholds:")
    for row in summary:
        print(
            f"  >= {row['threshold_pp']:.1f} pp: "
            f"{row['positive_windows']}/{row['eligible_windows']} "
            f"({row['event_rate_pct']:.2f}%), "
            f"{row['countries_with_events']} countries"
        )

    print(f"\nReports saved to: {report_dir}")
    print("No final threshold has been selected.")


if __name__ == "__main__":
    main()
