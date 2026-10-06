"""Download a sample youth unemployment series from the World Bank."""

import json
from datetime import datetime, timezone
from pathlib import Path

import requests


# Locate the project folder from this file's location.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

COUNTRY = "VNM"
INDICATOR = "SL.UEM.1524.ZS"

URL = (
    f"https://api.worldbank.org/v2/country/{COUNTRY}"
    f"/indicator/{INDICATOR}"
)


def main():
    params = {
        "format": "json",
        "date": "2000:2025",
        "per_page": 100,
        "page": 1,
    }

    print("Downloading Vietnam youth unemployment data...")

    response = requests.get(
        URL,
        params=params,
        timeout=60,
    )
    response.raise_for_status()

    payload = response.json()

    # The API normally returns [metadata, records].
    if (
        not isinstance(payload, list)
        or len(payload) != 2
        or not isinstance(payload[0], dict)
        or not isinstance(payload[1], list)
    ):
        raise ValueError(
            "Unexpected World Bank response. "
            f"Response preview: {str(payload)[:300]}"
        )

    metadata, records = payload

    # This small sample should fit on one page.
    # Stop rather than silently save incomplete data.
    if int(metadata.get("pages", 0)) != 1:
        raise ValueError(
            "The response requires pagination. "
            "This sample script expects exactly one page."
        )

    if not records:
        raise ValueError("The API returned no records.")

    retrieved_at = datetime.now(timezone.utc)
    timestamp = retrieved_at.strftime("%Y%m%dT%H%M%S%fZ")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RAW_DIR / (
        f"{COUNTRY}_{INDICATOR}_{timestamp}.json"
    )

    # Preserve missing values and the complete API response.
    snapshot = {
        "retrieved_at_utc": retrieved_at.isoformat(),
        "request_url": response.url,
        "country_code": COUNTRY,
        "indicator_code": INDICATOR,
        "api_response": payload,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            snapshot,
            file,
            ensure_ascii=False,
            indent=2,
        )

    available = [
        row for row in records
        if row["value"] is not None
    ]
    available.sort(key=lambda row: int(row["date"]))

    print(f"Total records: {len(records)}")
    print(f"Records with values: {len(available)}")
    print(f"Missing values: {len(records) - len(available)}")

    print("\nLatest available observations:")
    for row in available[-5:]:
        print(f"{row['date']}: {row['value']:.2f}%")

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()