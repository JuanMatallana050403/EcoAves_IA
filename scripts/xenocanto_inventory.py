"""Collect and optionally download licensed Xeno-canto recordings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import requests


API_URL = "https://xeno-canto.org/api/3/recordings"
QUALITY_ALLOWED = {"A", "B"}
FIELDS = [
    "recording_id", "species", "scientific_name", "common_name", "quality",
    "source_url", "download_url", "license", "creator", "rights_holder",
    "country", "locality", "latitude", "longitude", "recording_date",
    "recording_time", "duration", "sha256", "status", "manual_review",
    "downloaded_file", "source_recording_id",
]


def text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def absolute_url(value: str) -> str:
    if value.startswith("http"):
        return value
    if value.startswith("//"):
        return f"https:{value}"
    return f"https://xeno-canto.org{value}"


def species_name(recording: dict[str, Any]) -> str:
    return f"{text(recording.get('gen'))} {text(recording.get('sp'))}".strip()


def recording_row(recording: dict[str, Any], status: str = "candidate") -> dict[str, str]:
    recording_id = text(recording.get("id"))
    return {
        "recording_id": recording_id,
        "species": species_name(recording),
        "scientific_name": species_name(recording),
        "common_name": text(recording.get("en")),
        "quality": text(recording.get("q")),
        "source_url": f"https://xeno-canto.org/{recording_id}",
        "download_url": absolute_url(text(recording.get("file"))),
        "license": text(recording.get("lic")),
        "creator": text(recording.get("rec")),
        "rights_holder": text(recording.get("rec")),
        "country": text(recording.get("cnt")),
        "locality": text(recording.get("loc")),
        "latitude": text(recording.get("lat")),
        "longitude": text(recording.get("lng")),
        "recording_date": text(recording.get("date")),
        "recording_time": text(recording.get("time")),
        "duration": text(recording.get("length")),
        "sha256": "",
        "status": status,
        "manual_review": "pending",
        "downloaded_file": "",
        "source_recording_id": recording_id,
    }


def fetch_recordings(session: requests.Session, query: str, api_key: str, max_pages: int) -> list[dict[str, Any]]:
    recordings: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        response = session.get(API_URL, params={"query": query, "key": api_key, "page": page}, timeout=60)
        response.raise_for_status()
        payload = response.json()
        batch = payload.get("recordings", [])
        recordings.extend(batch)
        if page >= int(payload.get("numPages", page)) or not batch:
            break
        time.sleep(1)
    return recordings


def download_recording(session: requests.Session, row: dict[str, str], audio_root: Path) -> None:
    destination = audio_root / row["species"].replace(" ", "_") / f"{row['recording_id']}.mp3"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        row["downloaded_file"] = str(destination)
        row["sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
        row["status"] = "downloaded"
        return
    response = session.get(row["download_url"], stream=True, timeout=120)
    response.raise_for_status()
    temporary = destination.with_suffix(".download")
    with temporary.open("wb") as output:
        for chunk in response.iter_content(chunk_size=1024 * 128):
            if chunk:
                output.write(chunk)
    if temporary.stat().st_size < 1024:
        temporary.unlink(missing_ok=True)
        row["status"] = "rejected_too_small"
        return
    row["sha256"] = hashlib.sha256(temporary.read_bytes()).hexdigest()
    temporary.replace(destination)
    row["downloaded_file"] = str(destination)
    row["status"] = "downloaded"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", action="append", default=['cnt:Peru loc:"Tarapoto"', 'cnt:Peru loc:"Lamas"'])
    parser.add_argument("--species", nargs="*", help="Optional scientific names to keep")
    parser.add_argument("--max-per-species", type=int, default=40)
    parser.add_argument("--max-pages", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("dataset/audio"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("XENO_CANTO_API_KEY")
    if not api_key:
        raise SystemExit("Falta XENO_CANTO_API_KEY. Configúrala como variable de entorno; no la escribas en el script.")

    session = requests.Session()
    session.headers.update({"User-Agent": "EcoAvesPeruIA/0.1 research inventory"})
    raw_dir = args.output / "metadata" / "raw_xenocanto"
    rows_by_id: dict[str, dict[str, str]] = {}
    raw_dir.mkdir(parents=True, exist_ok=True)
    for query in args.query:
        recordings = fetch_recordings(session, query, api_key, args.max_pages)
        (raw_dir / f"query_{len(rows_by_id)}.json").write_text(json.dumps(recordings, ensure_ascii=False, indent=2), encoding="utf-8")
        for recording in recordings:
            row = recording_row(recording)
            if row["quality"] not in QUALITY_ALLOWED or not row["recording_id"]:
                continue
            if args.species and row["scientific_name"] not in args.species:
                continue
            if row["recording_id"] not in rows_by_id:
                rows_by_id[row["recording_id"]] = row

    counts: dict[str, int] = {}
    rows: list[dict[str, str]] = []
    for row in rows_by_id.values():
        if counts.get(row["species"], 0) >= args.max_per_species:
            continue
        counts[row["species"]] = counts.get(row["species"], 0) + 1
        if args.download:
            download_recording(session, row, args.output / "raw" / "audio")
        rows.append(row)
    write_csv(args.output / "metadata" / "audio_inventory.csv", rows)
    print(f"Recordings in inventory: {len(rows)}")
    print(f"Inventory: {args.output / 'metadata' / 'audio_inventory.csv'}")


if __name__ == "__main__":
    main()