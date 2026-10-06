"""Inspect missing data without changing or imputing any values."""

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

INDICATORS = [
    "youth_unemployment_pct",
    "gdp_growth_pct",
    "inflation_pct",
    "fdi_pct_gdp",
    "trade_pct_gdp",
    "youth_labor_force_participation_pct",
]


def has_value(row, column):
    return row[column].strip() != ""


def main():
    files = sorted(
        PROCESSED_DIR.glob("panel_*/country_year_panel.csv")
    )

    if not files:
        raise FileNotFoundError("Run build_panel.py first.")

    input_path = files[-1]

    with input_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        required = {"country_code", "year", *INDICATORS}
        missing_columns = required - set(reader.fieldnames or [])

        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")

        rows = list(reader)

    if not rows:
        raise ValueError("The panel is empty.")

    netherlands = sorted(
        [row for row in rows if row["country_code"] == "NLD"],
        key=lambda row: int(row["year"]),
    )

    if not netherlands:
        raise ValueError("Netherlands is missing from the panel.")

    lines = [
        f"Source run: {input_path.parent.name}",
        f"Total country-year rows: {len(rows)}",
        "",
        "NETHERLANDS: MISSING YEARS",
    ]

    for indicator in INDICATORS:
        missing_years = [
            int(row["year"])
            for row in netherlands
            if not has_value(row, indicator)
        ]
        lines.append(f"{indicator}: {missing_years}")

    complete_rows = [
        row for row in rows
        if all(has_value(row, name) for name in INDICATORS)
    ]

    lines.extend([
        "",
        "JOINT COVERAGE",
        f"Rows with all six indicators: {len(complete_rows)}/{len(rows)}",
        f"Joint coverage: {100 * len(complete_rows) / len(rows):.2f}%",
        "",
        "RECENT YEARS: AVAILABLE VALUES",
    ])

    years = sorted({int(row["year"]) for row in rows})

    for year in years[-5:]:
        subset = [row for row in rows if int(row["year"]) == year]

        lines.append(f"\nYear {year}: {len(subset)} countries/economies")

        for indicator in INDICATORS:
            count = sum(has_value(row, indicator) for row in subset)
            lines.append(f"  {indicator}: {count}/{len(subset)}")

        complete_count = sum(
            all(has_value(row, name) for name in INDICATORS)
            for row in subset
        )
        lines.append(
            f"  All six indicators together: {complete_count}/{len(subset)}"
        )

    report_dir = REPORTS_DIR / input_path.parent.name
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "missingness_audit.txt"

    report = "\n".join(lines)
    report_path.write_text(report + "\n", encoding="utf-8")

    print(report)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
