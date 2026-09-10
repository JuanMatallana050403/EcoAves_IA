"""Evaluate a saved MobileNet model on the held-out test split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from train_mobilenet import BirdDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--test", type=Path, default=Path("dataset/processed/splits/test.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/evaluation"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu")
    classes = checkpoint["classes"]
    weights = MobileNet_V3_Small_Weights.DEFAULT
    dataset = BirdDataset(args.test, classes, weights.transforms())
    loader = DataLoader(dataset, batch_size=args.batch_size)

    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(classes))
    model.load_state_dict(checkpoint["model"])
    model.eval()

    actual: list[int] = []
    predicted: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            predicted.extend(model(images).argmax(1).tolist())
            actual.extend(labels.tolist())

    report = classification_report(
        actual,
        predicted,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(actual, predicted, labels=list(range(len(classes))))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "classification_report.json").write_text(
        json.dumps({"classes": classes, "test_images": len(actual), "report": report}, indent=2),
        encoding="utf-8",
    )
    with (args.output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["actual/predicted", *classes])
        writer.writerows([[classes[index], *row] for index, row in enumerate(matrix.tolist())])

    present = {classes[index] for index in actual}
    missing = sorted(set(classes) - present)
    print(f"Test images: {len(actual)}")
    print(f"Accuracy: {report['accuracy']:.3f}")
    print(f"Macro F1: {report['macro avg']['f1-score']:.3f}")
    print("\nPer class:")
    for name in classes:
        values = report[name]
        print(f"{name}: precision={values['precision']:.3f} recall={values['recall']:.3f} f1={values['f1-score']:.3f} support={int(values['support'])}")
    if missing:
        print(f"WARNING: test sin ejemplos de: {', '.join(missing)}")
    print(f"Report: {args.output / 'classification_report.json'}")
    print(f"Confusion matrix: {args.output / 'confusion_matrix.csv'}")


if __name__ == "__main__":
    main()