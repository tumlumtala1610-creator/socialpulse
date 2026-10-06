"""Package a small, versioned data snapshot for the hosted dashboard."""

import gzip
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main():
    with (ROOT / "docs" / "experiment_config.json").open(
        encoding="utf-8"
    ) as file:
        config = json.load(file)

    run_id = config["source_run"]
    raw_dir = ROOT / "data" / "raw" / run_id
    source = (
        ROOT / "data" / "processed" / run_id / "country_year_panel.csv"
    )

    with (raw_dir / "manifest.json").open(encoding="utf-8") as file:
        manifest = json.load(file)

    if manifest["status"] != "complete" or not source.exists():
        raise ValueError("A complete prepared dataset is required.")

    destination = ROOT / "app" / "data" / run_id
    destination.mkdir(parents=True, exist_ok=True)

    source_bytes = source.read_bytes()
    compressed_path = destination / "country_year_panel.csv.gz"

    # Fixed gzip timestamp makes the same data reproducible byte-for-byte.
    compressed_path.write_bytes(gzip.compress(source_bytes, mtime=0))

    shutil.copy2(raw_dir / "manifest.json", destination / "manifest.json")
    shutil.copytree(
        raw_dir / "indicator_metadata",
        destination / "indicator_metadata",
        dirs_exist_ok=True,
    )

    provenance = {
        "source_run": run_id,
        "source": "World Bank Indicators API / World Development Indicators",
        "source_url": "https://api.worldbank.org/v2",
        "download_finished_at_utc": manifest["finished_at_utc"],
        "panel_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "description": (
            "Country-year table assembled from API responses. "
            "Aggregate regions excluded; missing values preserved."
        ),
    }

    with (destination / "provenance.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(provenance, file, ensure_ascii=False, indent=2)

    print(f"Packaged data: {compressed_path}")
    print(f"Compressed size: {compressed_path.stat().st_size / 1024:.1f} KB")
    print("Source manifest, indicator metadata and checksum included.")


if __name__ == "__main__":
    main()
