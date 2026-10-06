"""Download a versioned World Bank dataset for coverage analysis."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
BASE_URL = "https://api.worldbank.org/v2"

START_YEAR = 2000
END_YEAR = 2025
SOURCE_ID = 2  # World Development Indicators

INDICATORS = {
    "youth_unemployment_pct": "SL.UEM.1524.ZS",
    "gdp_growth_pct": "NY.GDP.MKTP.KD.ZG",
    "inflation_pct": "FP.CPI.TOTL.ZG",
    "fdi_pct_gdp": "BX.KLT.DINV.WD.GD.ZS",
    "trade_pct_gdp": "NE.TRD.GNFS.ZS",
    "youth_labor_force_participation_pct": "SL.TLF.ACTI.1524.ZS",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, content):
    with path.open("w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2)


def request_page(session, endpoint, params):
    """Try a request up to three times, then stop with an error."""
    url = f"{BASE_URL}/{endpoint}"

    for attempt in range(1, 4):
        try:
            response = session.get(url, params=params, timeout=60)
            response.raise_for_status()
            payload = response.json()

            if (
                not isinstance(payload, list)
                or len(payload) != 2
                or not isinstance(payload[0], dict)
                or not isinstance(payload[1], list)
            ):
                raise ValueError(
                    f"Unexpected API response: {str(payload)[:200]}"
                )

            return {
                "retrieved_at_utc": utc_now(),
                "request_url": response.url,
                "api_response": payload,
            }

        except (requests.RequestException, ValueError) as error:
            if attempt == 3:
                raise RuntimeError(
                    f"Request failed after 3 attempts: {url}"
                ) from error

            print(f"  Attempt {attempt} failed; retrying...")
            time.sleep(2 * attempt)


def download_all_pages(session, endpoint, params, output_dir):
    """Save every API page and verify the returned record count."""
    output_dir.mkdir(parents=True, exist_ok=True)

    page = 1
    expected_pages = None
    expected_total = None
    count = 0

    while True:
        query = {
            **params,
            "format": "json",
            "per_page": 1000,
            "page": page,
        }

        snapshot = request_page(session, endpoint, query)
        metadata, records = snapshot["api_response"]

        pages = int(metadata["pages"])
        total = int(metadata["total"])

        if int(metadata["page"]) != page:
            raise ValueError("The API returned an unexpected page.")

        if expected_pages is None:
            expected_pages = pages
            expected_total = total
        elif pages != expected_pages or total != expected_total:
            raise ValueError(
                "API pagination changed during download. Run again."
            )

        if pages < 1:
            raise ValueError("The API returned no usable pages.")

        save_json(output_dir / f"page_{page:04d}.json", snapshot)
        count += len(records)

        print(f"  Page {page}/{pages}: {len(records)} records")

        if page == pages:
            break

        page += 1
        time.sleep(0.3)

    if count != expected_total:
        raise ValueError(
            f"Incomplete download: {count} of {expected_total} records."
        )

    return {"pages": expected_pages, "records": count}


def main():
    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    run_dir = RAW_DIR / f"panel_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "status": "running",
        "started_at_utc": utc_now(),
        "start_year": START_YEAR,
        "end_year": END_YEAR,
        "source_id": SOURCE_ID,
        "indicators": INDICATORS,
        "downloads": {},
    }

    manifest_path = run_dir / "manifest.json"
    save_json(manifest_path, manifest)

    try:
        with requests.Session() as session:
            session.headers.update({
                "User-Agent": "SocialPulse research project"
            })

            print("Downloading country metadata...")
            manifest["downloads"]["countries"] = download_all_pages(
                session,
                "country",
                {},
                run_dir / "countries",
            )
            save_json(manifest_path, manifest)

            for name, code in INDICATORS.items():
                print(f"\nDownloading metadata: {code}")

                metadata_summary = download_all_pages(
                    session,
                    f"indicator/{code}",
                    {"source": SOURCE_ID},
                    run_dir / "indicator_metadata" / code,
                )

                print(f"Downloading values: {name}")

                data_summary = download_all_pages(
                    session,
                    f"country/all/indicator/{code}",
                    {
                        "source": SOURCE_ID,
                        "date": f"{START_YEAR}:{END_YEAR}",
                    },
                    run_dir / "indicators" / code,
                )

                manifest["downloads"][name] = {
                    "metadata": metadata_summary,
                    "data": data_summary,
                }
                save_json(manifest_path, manifest)

        manifest["status"] = "complete"

    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        raise

    finally:
        manifest["finished_at_utc"] = utc_now()
        save_json(manifest_path, manifest)

    print("\nDownload complete.")
    print(f"Saved to: {run_dir}")


if __name__ == "__main__":
    main()
