"""Train a first transfer-learning classifier from prepared split CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.transforms import v2


class BirdDataset(Dataset):
    def __init__(self, csv_path: Path, classes: list[str], transform: v2.Compose) -> None:
        with csv_path.open(newline="", encoding="utf-8-sig") as source:
            self.rows = list(csv.DictReader(source))
        self.classes = classes
        self.class_to_index = {name: index for index, name in enumerate(classes)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        image_path = Path(row["downloaded_file"])
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), self.class_to_index[row["species"]]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, int]:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            correct += int((outputs.argmax(1).cpu() == labels).sum())
            total += labels.numel()
    return (correct / total if total else 0.0), total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=Path("dataset/processed/splits"))
    parser.add_argument("--output", type=Path, default=Path("models/mobilenet_v3_small.pt"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()

    train_rows = read_rows(args.splits / "train.csv")
    validation_rows = read_rows(args.splits / "validation.csv")
    test_rows = read_rows(args.splits / "test.csv")
    classes = sorted({row["species"] for row in train_rows})
    if len(classes) < 2:
        raise ValueError("Se necesitan al menos dos especies en train.csv")

    for name, rows in (("validation", validation_rows), ("test", test_rows)):
        missing = sorted(set(classes) - {row["species"] for row in rows})
        if missing:
            print(f"ADVERTENCIA: {name} no contiene: {', '.join(missing)}")

    weights = MobileNet_V3_Small_Weights.DEFAULT
    eval_transform = weights.transforms()
    train_transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(8),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=eval_transform.mean, std=eval_transform.std),
    ])
    train_dataset = BirdDataset(args.splits / "train.csv", classes, train_transform)
    validation_dataset = BirdDataset(args.splits / "validation.csv", classes, eval_transform)
    test_dataset = BirdDataset(args.splits / "test.csv", classes, eval_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mobilenet_v3_small(weights=weights)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    model.to(device)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
        validation_accuracy, validation_total = evaluate(model, validation_loader, device)
        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"loss={total_loss / max(len(train_dataset), 1):.4f} "
            f"validation_accuracy={validation_accuracy:.3f} "
            f"validation_images={validation_total}"
        )

    test_accuracy, test_total = evaluate(model, test_loader, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "classes": classes}, args.output)
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps({"classes": classes, "test_accuracy": test_accuracy, "test_images": test_total}, indent=2),
        encoding="utf-8",
    )
    print(f"test_accuracy={test_accuracy:.3f} test_images={test_total}")
    print(f"model={args.output}")


if __name__ == "__main__":
    main()