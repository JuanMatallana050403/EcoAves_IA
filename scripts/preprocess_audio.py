"""Create Mel-spectrogram segments while retaining recording provenance."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np


def process_audio(audio_path: Path, output_dir: Path, duration: float, sample_rate: int) -> list[dict[str, str]]:
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    chunk_length = int(duration * sample_rate)
    if len(audio) < chunk_length:
        audio = librosa.util.fix_length(audio, size=chunk_length)
    segments: list[dict[str, str]] = []
    for index, start in enumerate(range(0, max(len(audio) - chunk_length + 1, 1), chunk_length)):
        chunk = audio[start:start + chunk_length]
        if len(chunk) < chunk_length:
            chunk = librosa.util.fix_length(chunk, size=chunk_length)
        mel = librosa.feature.melspectrogram(y=chunk, sr=sample_rate, n_mels=128, fmax=8000)
        db = librosa.power_to_db(mel, ref=np.max)
        normalized = (db - db.min()) / (db.max() - db.min() + 1e-6)
        output_path = output_dir / f"{audio_path.stem}_{index}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(output_path, normalized, origin="lower", cmap="viridis")
        segments.append({"recording_id": audio_path.stem, "segment_index": str(index), "start_seconds": str(start / sample_rate), "end_seconds": str((start + chunk_length) / sample_rate), "spectrogram_file": str(output_path)})
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, default=Path("dataset/audio/raw/audio"))
    parser.add_argument("--output-root", type=Path, default=Path("dataset/audio/processed/spectrograms"))
    parser.add_argument("--manifest", type=Path, default=Path("dataset/audio/processed/spectrogram_manifest.csv"))
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--sample-rate", type=int, default=22050)
    args = parser.parse_args()
    rows: list[dict[str, str]] = []
    for audio_path in args.audio_root.rglob("*.mp3"):
        species = audio_path.parent.name
        rows.extend({"species": species, **segment} for segment in process_audio(audio_path, args.output_root / species, args.duration, args.sample_rate))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    fields = ["species", "recording_id", "segment_index", "start_seconds", "end_seconds", "spectrogram_file"]
    with args.manifest.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Spectrogram segments: {len(rows)}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()