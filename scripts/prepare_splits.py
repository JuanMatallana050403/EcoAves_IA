"""Create train, validation, and test CSVs without splitting place/date groups."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from math import ceil
from pathlib import Path


def text(value: object) -> str:
    return "" if value is None else str(value).strip()


def group_key(row: dict[str, str]) -> str:
    latitude = text(row.get("latitude"))
    longitude = text(row.get("longitude"))
    event_date = text(row.get("event_date"))[:7]
    try:
        place = f"{float(latitude):.2f},{float(longitude):.2f}" if latitude and longitude else "unknown_place"
    except ValueError:
        place = "unknown_place"
    return f"{place}|{event_date or 'unknown_date'}"


def assign_group_splits(groups: list[str]) -> dict[str, str]:
    ordered = sorted(groups, key=lambda group: hashlib.sha256(group.encode("utf-8")).hexdigest())
    test_count = max(1, ceil(len(ordered) * 0.15)) if len(ordered) >= 3 else 0
    validation_count = max(1, ceil(len(ordered) * 0.15)) if len(ordered) >= 2 else 0
    train_end = len(ordered) - test_count - validation_count
    if train_end < 1 and len(ordered) >= 3:
        train_end, validation_count, test_count = 1, 1, len(ordered) - 2
    return {
        group: "train" if index < train_end else "validation" if index < train_end + validation_count else "test"
        for index, group in enumerate(ordered)
    }


def read_approved(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    return [
        row for row in rows
        if text(row.get("manual_review")).lower() == "approved"
        and text(row.get("downloaded_file"))
        and not text(row.get("status")).startswith("duplicate")
    ]


def write_rows(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset/metadata/image_inventory.csv"))
    parser.add_argument("--output", type=Path, default=Path("dataset/processed/splits"))
    args = parser.parse_args()

    rows = read_approved(args.input)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        group = group_key(row)
        row["split_group"] = group
        groups[group].append(row)

    group_splits = assign_group_splits(list(groups))
    split_rows: dict[str, list[dict[str, str]]] = {"train": [], "validation": [], "test": []}
    for group, group_rows in groups.items():
        split_rows[group_splits[group]].extend(group_rows)

    args.output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["species", "split_group"]
    for split, split_data in split_rows.items():
        write_rows(args.output / f"{split}.csv", split_data, fields)

    print(f"Approved unique images: {len(rows)}")
    print(f"Groups by place/month: {len(groups)}")
    for split, split_data in split_rows.items():
        print(f"{split}: {len(split_data)}")


if __name__ == "__main__":
    main()