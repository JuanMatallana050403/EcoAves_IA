"""Build a licensed image inventory from GBIF occurrence records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import imagehash
import requests
from PIL import Image


GBIF_URL = "https://api.gbif.org/v1/occurrence/search"
DEFAULT_SPECIES = [
    "Spizaetus isidori",
    "Grallaricula ochraceifrons",
    "Xenoglaux loweryi",
    "Ara militaris",
    "Heliangelus regalis",
    "Penelope barbata",
    "Spizaetus ornatus",
]
ALLOWED_LICENSES = ("cc0", "creativecommons.org/publicdomain/zero", "cc-by", "cc-by-sa")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CSV_FIELDS = [
    "species", "scientific_name", "gbif_key", "media_id", "media_url", "source_url",
    "license", "license_scope", "creator", "rights_holder", "occurrence_status", "country",
    "state_province", "locality", "latitude", "longitude", "event_date",
    "quality_grade", "width", "height", "sha256", "phash", "status",
    "manual_review", "split", "downloaded_file",
]


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalized_region(value: str) -> str:
    value = text(value).replace("Ã¡", "a").replace("Ã©", "e").replace("Ã­", "i")
    value = value.replace("Ã³", "o").replace("Ãº", "u").replace("Ã±", "n")
    value = unicodedata.normalize("NFKD", value)
    return "".join(character for character in value if not unicodedata.combining(character)).lower()


def license_scope(value: str, allow_noncommercial: bool = False) -> str:
    normalized = text(value).lower().replace(" ", "")
    if "/by-nc-nd/" in normalized or "/by-nd/" in normalized:
        return "rejected"
    if "/by-nc/" in normalized:
        return "research_only" if allow_noncommercial else "rejected"
    if "/by/" in normalized or "/by-sa/" in normalized or any(item in normalized for item in ALLOWED_LICENSES):
        return "redistributable"
    return "rejected"


def media_license(record: dict[str, Any], media: dict[str, Any]) -> str:
    return text(media.get("license") or record.get("license"))


def media_creator(record: dict[str, Any], media: dict[str, Any]) -> str:
    return text(media.get("creator") or media.get("creatorName") or record.get("recordedBy"))


def fetch_species(session: requests.Session, scientific_name: str, state_province: str, max_records: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    offset = 0
    while len(records) < max_records:
        limit = min(300, max_records - len(records))
        params = {
            "scientificName": scientific_name,
            "country": "PE",
            "media_type": "StillImage",
            "limit": limit,
            "offset": offset,
        }
        retryable_statuses = {429, 500, 502, 503, 504}
        for attempt in range(5):
            try:
                response = session.get(GBIF_URL, params=params, timeout=60)
            except requests.RequestException:
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 30))
                continue
            if response.status_code not in retryable_statuses:
                response.raise_for_status()
                break
            retry_after = response.headers.get("Retry-After")
            delay = int(retry_after) if retry_after and retry_after.isdigit() else 2 ** attempt
            time.sleep(min(delay, 30))
        else:
            raise RuntimeError("GBIF no respondió correctamente después de varios reintentos")
        payload = response.json()
        batch = payload.get("results", [])
        target_region = normalized_region(state_province)
        records.extend(
            record for record in batch
            if normalized_region(record.get("stateProvince")) == target_region
        )
        if len(batch) < limit:
            break
        offset += len(batch)
    return records[:max_records]


def first_image(record: dict[str, Any]) -> dict[str, Any] | None:
    for media in record.get("media", []):
        media_type = text(media.get("type")).lower()
        identifier = text(media.get("identifier") or media.get("references"))
        if identifier and (not media_type or media_type == "stillimage"):
            return media
    return None


def initial_row(record: dict[str, Any], media: dict[str, Any], species: str, scope: str) -> dict[str, Any]:
    return {
        "species": species,
        "scientific_name": text(record.get("scientificName") or species),
        "gbif_key": text(record.get("key")),
        "media_id": text(media.get("identifier") or media.get("references")),
        "media_url": text(media.get("identifier") or media.get("references")),
        "source_url": text(record.get("references")),
        "license": media_license(record, media),
        "license_scope": scope,
        "creator": media_creator(record, media),
        "rights_holder": text(media.get("rightsHolder") or record.get("rightsHolder")),
        "occurrence_status": text(record.get("occurrenceStatus")),
        "country": text(record.get("country")),
        "state_province": text(record.get("stateProvince")),
        "locality": text(record.get("locality")),
        "latitude": text(record.get("decimalLatitude")),
        "longitude": text(record.get("decimalLongitude")),
        "event_date": text(record.get("eventDate") or record.get("year")),
        "quality_grade": text(record.get("identificationVerificationStatus")),
        "width": text(media.get("width")),
        "height": text(media.get("height")),
        "sha256": "", "phash": "", "status": "candidate",
        "manual_review": "pending", "split": "", "downloaded_file": "",
    }


def download_image(session: requests.Session, row: dict[str, Any], destination: Path) -> None:
    response = session.get(row["media_url"], stream=True, timeout=90)
    response.raise_for_status()
    if not response.headers.get("content-type", "").lower().startswith("image/"):
        row["status"] = "rejected_not_image"
        return

    temporary = destination.with_suffix(".download")
    with temporary.open("wb") as output:
        for chunk in response.iter_content(chunk_size=1024 * 128):
            if chunk:
                output.write(chunk)
    try:
        with Image.open(temporary) as image:
            image.verify()
        with Image.open(temporary) as image:
            width, height = image.size
            row["width"], row["height"] = str(width), str(height)
            if min(width, height) < 256:
                row["status"] = "rejected_low_resolution"
                temporary.unlink(missing_ok=True)
                return
            row["phash"] = str(imagehash.phash(image))
    except Exception:
        row["status"] = "rejected_invalid_image"
        temporary.unlink(missing_ok=True)
        return

    row["sha256"] = hashlib.sha256(temporary.read_bytes()).hexdigest()
    extension = Path(response.url).suffix.lower()
    if extension not in IMAGE_EXTENSIONS:
        extension = ".jpg"
    final_path = destination.with_name(f"{row['sha256'][:16]}{extension}")
    temporary.replace(final_path)
    row["downloaded_file"] = str(final_path)
    row["status"] = "downloaded"


def group_key(row: dict[str, Any]) -> str:
    latitude, longitude = text(row["latitude"]), text(row["longitude"])
    event_date = text(row["event_date"])[:7]
    try:
        place = f"{float(latitude):.2f},{float(longitude):.2f}" if latitude and longitude else "unknown_place"
    except ValueError:
        place = "unknown_place"
    return f"{place}|{event_date or 'unknown_date'}"


def assign_splits(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if not row["status"].startswith("downloaded") and row["status"] != "candidate":
            continue
        digest = hashlib.sha256(group_key(row).encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF
        row["split"] = "train" if bucket < 0.70 else "validation" if bucket < 0.85 else "test"


def mark_duplicates(rows: list[dict[str, Any]]) -> None:
    exact: dict[str, dict[str, Any]] = {}
    hashes: list[tuple[dict[str, Any], imagehash.ImageHash]] = []
    for row in rows:
        if not row["status"].startswith("downloaded"):
            continue
        if row["sha256"] in exact:
            row["status"] = "duplicate_exact"
            continue
        exact[row["sha256"]] = row
        if row["phash"]:
            hashes.append((row, imagehash.hex_to_hash(row["phash"])))
    for index, (row, current_hash) in enumerate(hashes):
        for previous, previous_hash in hashes[:index]:
            if row["status"].startswith("downloaded") and previous["species"] == row["species"] and current_hash - previous_hash <= 6:
                row["status"] = "duplicate_near"
                break


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", nargs="+", default=DEFAULT_SPECIES)
    parser.add_argument("--state-province", default="San Martin")
    parser.add_argument("--max-per-species", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("dataset"))
    parser.add_argument("--download", action="store_true", help="Download accepted image files")
    parser.add_argument(
        "--allow-noncommercial",
        action="store_true",
        help="Also download CC BY-NC images as research_only; never accepts CC BY-NC-ND",
    )
    args = parser.parse_args()
    raw_dir, image_dir = args.output / "metadata" / "raw_gbif", args.output / "raw" / "images"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if args.download:
        image_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "EcoAvesPeruIA/0.1 research inventory"})
    rows: list[dict[str, Any]] = []
    for species in args.species:
        records = fetch_species(session, species, args.state_province, args.max_per_species)
        (raw_dir / f"{slug(species)}.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        for record in records:
            media = first_image(record)
            if not media:
                continue
            scope = license_scope(media_license(record, media), args.allow_noncommercial)
            row = initial_row(record, media, species, scope)
            if scope == "rejected":
                row["status"] = "rejected_license"
            elif args.download:
                target = image_dir / slug(species) / "pending.jpg"
                target.parent.mkdir(parents=True, exist_ok=True)
                download_image(session, row, target)
                if row["status"] == "downloaded" and scope == "research_only":
                    row["status"] = "downloaded_research_only"
            rows.append(row)
    if args.download:
        mark_duplicates(rows)
    assign_splits(rows)
    output_csv = args.output / "metadata" / "image_inventory.csv"
    write_csv(output_csv, rows)
    print(f"Records with media: {len(rows)}")
    print(f"Inventory: {output_csv}")


if __name__ == "__main__":
    main()