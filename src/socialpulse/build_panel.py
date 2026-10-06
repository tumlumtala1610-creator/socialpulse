"""Build a country-year panel and coverage reports from saved API data."""

import csv
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

BOUNDED_PERCENTAGES = {
    "youth_unemployment_pct",
    "youth_labor_force_participation_pct",
}


def read_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_pages(folder, expected):
    """Check saved page numbers and counts before combining records."""
    files = sorted(folder.glob("page_*.json"))

    if len(files) != expected["pages"]:
        raise ValueError(f"Missing or extra page files in {folder}")

    records = []

    for page_number, path in enumerate(files, start=1):
        payload = read_json(path)["api_response"]
        metadata, page_records = payload

        if (
            int(metadata["page"]) != page_number
            or int(metadata["pages"]) != expected["pages"]
            or int(metadata["total"]) != expected["records"]
        ):
            raise ValueError(f"Inconsistent pagination: {path}")

        records.extend(page_records)

    if len(records) != expected["records"]:
        raise ValueError(f"Record count mismatch in {folder}")

    return records


def main():
    # Select the newest successful download, ignoring failed runs.
    candidates = []

    for folder in sorted(RAW_DIR.glob("panel_*")):
        manifest_path = folder / "manifest.json"

        if manifest_path.exists():
            manifest = read_json(manifest_path)

            if manifest.get("status") == "complete":
                candidates.append((folder, manifest))

    if not candidates:
        raise FileNotFoundError(
            "No complete panel download found. Run fetch_panel.py first."
        )

    run_dir, manifest = candidates[-1]
    indicators = manifest["indicators"]
    years = list(range(
        manifest["start_year"],
        manifest["end_year"] + 1,
    ))

    country_records = read_pages(
        run_dir / "countries",
        manifest["downloads"]["countries"],
    )

    # Retain country/economy records; exclude regional/income aggregates.
    all_countries = {}

    for country in country_records:
        code = country["id"]

        if code in all_countries:
            raise ValueError(f"Duplicate country metadata: {code}")

        all_countries[code] = country

    iso2_to_iso3 = {
        item["iso2Code"]: item["id"]
        for item in country_records
        if item.get("iso2Code")
    }

    countries = {
        code: country
        for code, country in all_countries.items()
        if country["region"]["id"] != "NA"
        and country["region"]["value"] != "Aggregates"
    }

    if "NLD" not in countries:
        raise ValueError("Netherlands is missing from country metadata.")

    # Create every country-year, including years with no data.
    panel = {}

    for code, country in sorted(countries.items()):
        for year in years:
            row = {
                "country_code": code,
                "country_name": country["name"],
                "region_at_download": country["region"]["value"],
                "income_group_at_download": country["incomeLevel"]["value"],
                "year": year,
                "source_run": run_dir.name,
            }

            for name in indicators:
                row[name] = None

            panel[(code, year)] = row

    indicator_coverage = []
    year_coverage = []
    country_coverage = []

    for name, indicator_code in indicators.items():
        records = read_pages(
            run_dir / "indicators" / indicator_code,
            manifest["downloads"][name]["data"],
        )

        seen = set()
        aggregates_excluded = 0

        for record in records:
            if record["indicator"]["id"] != indicator_code:
                raise ValueError(f"Wrong indicator in {name}")

            code = record.get("countryiso3code")

            # Some aggregate records may have an empty ISO3 field.
            if not code:
                code = iso2_to_iso3.get(record["country"]["id"])

            if code not in all_countries:
                raise ValueError(
                    f"Unrecognized country: {record['country']}"
                )

            if code not in countries:
                aggregates_excluded += 1
                continue

            year = int(record["date"])
            key = (code, year)

            if key not in panel:
                raise ValueError(f"Unexpected country-year: {key}")

            if key in seen:
                raise ValueError(f"Duplicate {name} record: {key}")

            seen.add(key)
            value = record["value"]

            if value is not None:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(f"Invalid value for {name}: {key}")

                if name in BOUNDED_PERCENTAGES and not 0 <= value <= 100:
                    raise ValueError(f"Percentage out of range: {name}, {key}")

            panel[key][name] = value

        available = sum(
            row[name] is not None for row in panel.values()
        )

        indicator_coverage.append({
            "indicator": name,
            "indicator_code": indicator_code,
            "expected_country_years": len(panel),
            "available_values": available,
            "missing_values": len(panel) - available,
            "coverage_pct": round(100 * available / len(panel), 2),
            "aggregate_records_excluded": aggregates_excluded,
        })

        for year in years:
            count = sum(
                panel[(code, year)][name] is not None
                for code in countries
            )

            year_coverage.append({
                "indicator": name,
                "year": year,
                "expected_countries": len(countries),
                "available_countries": count,
                "coverage_pct": round(100 * count / len(countries), 2),
            })

        for code, country in sorted(countries.items()):
            valid_years = [
                year for year in years
                if panel[(code, year)][name] is not None
            ]

            country_coverage.append({
                "country_code": code,
                "country_name": country["name"],
                "indicator": name,
                "expected_years": len(years),
                "available_years": len(valid_years),
                "coverage_pct": round(
                    100 * len(valid_years) / len(years), 2
                ),
                "first_available_year": min(valid_years) if valid_years else None,
                "last_available_year": max(valid_years) if valid_years else None,
            })

    output_dir = PROCESSED_DIR / run_dir.name
    report_dir = REPORTS_DIR / run_dir.name
    panel_rows = list(panel.values())

    write_csv(output_dir / "country_year_panel.csv", panel_rows)
    write_csv(
        output_dir / "netherlands_panel.csv",
        [row for row in panel_rows if row["country_code"] == "NLD"],
    )

    write_csv(report_dir / "coverage_by_indicator.csv", indicator_coverage)
    write_csv(report_dir / "coverage_by_year.csv", year_coverage)
    write_csv(report_dir / "coverage_by_country.csv", country_coverage)

    print(f"Source run: {run_dir.name}")
    print(f"Countries/economies: {len(countries)}")
    print(f"Years: {years[0]}-{years[-1]}")
    print(f"Panel rows: {len(panel_rows)}")

    print("\nOverall coverage:")
    for row in indicator_coverage:
        print(f"  {row['indicator']}: {row['coverage_pct']:.2f}%")

    print("\nNetherlands coverage:")
    for row in country_coverage:
        if row["country_code"] == "NLD":
            print(
                f"  {row['indicator']}: "
                f"{row['available_years']}/{row['expected_years']} years"
            )

    print(f"\nData saved to: {output_dir}")
    print(f"Reports saved to: {report_dir}")


if __name__ == "__main__":
    main()
