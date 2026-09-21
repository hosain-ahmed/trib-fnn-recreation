"""Train and compare the three standalone Day 3 baseline models.

Run from the repository root after generating data/processed/*_tensors.pt:

    python -m train.train_baselines --epochs 1 --max-batches 50

The max-batches option intentionally supports a quick architecture sanity
pass.  Remove it for a complete training run.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model.baseline_models import BASELINE_MODELS


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(path: Path) -> TensorDataset:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run data/data_pipeline.py and data/tokenizer.py first."
        )
    data = torch.load(path, map_location="cpu", weights_only=True)
    return TensorDataset(data["input_ids"], data["attention_mask"], data["labels"])


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            logits = model(input_ids.to(device), attention_mask.to(device))
            correct += (logits.argmax(dim=1).cpu() == labels).sum().item()
            total += labels.numel()
    return correct / total if total else 0.0


def train_one(
    name: str,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    vocab_size: int,
    device: torch.device,
    epochs: int,
    max_batches: int | None,
    learning_rate: float,
) -> dict[str, float | int | str]:
    model = BASELINE_MODELS[name](vocab_size=vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    batches_seen = 0
    last_loss = 0.0

    for epoch in range(epochs):
        model.train()
        for input_ids, attention_mask, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids.to(device), attention_mask.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            last_loss = loss.item()
            batches_seen += 1
            if max_batches is not None and batches_seen >= max_batches:
                break
        if max_batches is not None and batches_seen >= max_batches:
            break

    accuracy = evaluate(model, validation_loader, device)
    return {
        "model": name,
        "epochs_requested": epochs,
        "batches_trained": batches_seen,
        "last_training_loss": round(last_loss, 6),
        "validation_accuracy": round(accuracy, 6),
        "device": str(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("eval/baseline_results.json"))
    args = parser.parse_args()

    set_seed(args.seed)
    train_data = load_dataset(args.data_dir / "train_tensors.pt")
    validation_data = load_dataset(args.data_dir / "val_tensors.pt")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size)
    vocab_size = int(max(train_data.tensors[0].max(), validation_data.tensors[0].max()).item()) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = [
        train_one(
            name,
            train_loader,
            validation_loader,
            vocab_size,
            device,
            args.epochs,
            args.max_batches,
            args.learning_rate,
        )
        for name in ("cnn", "bilstm", "gru")
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
