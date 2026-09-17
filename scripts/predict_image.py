"""Run one-image inference with a saved EcoAves MobileNet model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/mobilenet_balanced.pt"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--unknown-threshold", type=float, default=0.55)
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu")
    classes = checkpoint["classes"]
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(classes))
    model.load_state_dict(checkpoint["model"])
    model.eval()

    image = Image.open(args.image).convert("RGB")
    tensor = weights.transforms()(image).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor)[0], dim=0)
    values, indices = probabilities.topk(min(args.top_k, len(classes)))

    best_index = int(indices[0])
    best_probability = float(values[0])
    best_class = classes[best_index]
    if best_class != "unknown" and best_probability < args.unknown_threshold:
        best_class = "unknown"

    print(f"prediction={best_class}")
    print(f"confidence={best_probability:.4f}")
    print("top_predictions:")
    for value, index in zip(values.tolist(), indices.tolist()):
        print(f"  {classes[index]}: {value:.4f}")


if __name__ == "__main__":
    main()