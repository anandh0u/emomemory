"""Train a compact PyTorch CNN for FER2013 image emotion classification."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from data.fer_loader import find_default_fer_csv, load_fer2013_csv


@dataclass(frozen=True)
class FERArrays:
    images: np.ndarray
    labels: np.ndarray


class FERImageDataset(Dataset):
    def __init__(self, split: FERArrays, augment: bool = False) -> None:
        self.images = split.images
        self.labels = split.labels
        self.augment = augment

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index].astype(np.float32) / 255.0
        if self.augment:
            if random.random() < 0.5:
                image = np.fliplr(image).copy()
            if random.random() < 0.35:
                scale = random.uniform(0.88, 1.12)
                shift = random.uniform(-0.08, 0.08)
                image = np.clip(image * scale + shift, 0.0, 1.0)
        tensor = torch.from_numpy(image).unsqueeze(0)
        label = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return tensor, label


class SmallFERNet(nn.Module):
    def __init__(self, class_count: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 16),
            conv_block(16, 16),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.05),
            conv_block(16, 32),
            conv_block(32, 32),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),
            conv_block(32, 64),
            conv_block(64, 64),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            conv_block(64, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(64, class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a compact FER2013 CNN.")
    parser.add_argument("--data-path", type=Path, default=find_default_fer_csv())
    parser.add_argument("--task", choices=["binary", "fer7"], default="binary")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-out", type=Path, default=Path("artifacts/fer_binary_cnn.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.data_path is None:
        raise FileNotFoundError("Could not find fer2013.csv. Pass --data-path explicitly.")

    set_seed(args.random_state)
    dataset = load_fer2013_csv(args.data_path, task=args.task)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        json.dumps(
            {
                "source": str(dataset.source),
                "task": dataset.task,
                "class_names": dataset.class_names,
                "device": str(device),
                "train": int(dataset.train.labels.shape[0]),
                "val": int(dataset.val.labels.shape[0]),
                "test": int(dataset.test.labels.shape[0]),
            },
            indent=2,
        ),
        flush=True,
    )

    train_split = FERArrays(dataset.train.images, dataset.train.labels)
    val_split = FERArrays(dataset.val.images, dataset.val.labels)
    test_split = FERArrays(dataset.test.images, dataset.test.labels)
    train_loader = DataLoader(FERImageDataset(train_split, augment=True), batch_size=args.batch_size, shuffle=True)
    eval_train_loader = DataLoader(FERImageDataset(train_split), batch_size=args.batch_size)
    val_loader = DataLoader(FERImageDataset(val_split), batch_size=args.batch_size)
    test_loader = DataLoader(FERImageDataset(test_split), batch_size=args.batch_size)

    model = SmallFERNet(len(dataset.class_names)).to(device)
    labels, counts = np.unique(dataset.train.labels, return_counts=True)
    weights = np.ones(len(dataset.class_names), dtype=np.float32)
    weights[labels] = counts.sum() / (len(labels) * counts)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["accuracy"])
        history.append({"epoch": epoch, "loss": loss, "validation": val_metrics})
        print(
            f"epoch={epoch:02d} loss={loss:.4f} "
            f"val_acc={val_metrics['accuracy'] * 100:.2f}% "
            f"val_f1={val_metrics['weighted_f1'] * 100:.2f}%",
            flush=True,
        )

        score = val_metrics["accuracy"]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after epoch {epoch}; best epoch was {best_epoch}.", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {
        "train": evaluate(model, eval_train_loader, device),
        "validation": evaluate(model, val_loader, device),
        "test": evaluate(model, test_loader, device),
        "task": dataset.task,
        "classifier": "small_cnn",
        "class_names": dataset.class_names,
        "source": str(dataset.source),
        "best_epoch": best_epoch,
        "history": history,
    }
    payload = {
        "model_state": model.state_dict(),
        "model_class": "SmallFERNet",
        "metrics": metrics,
        "class_names": dataset.class_names,
        "task": dataset.task,
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.model_out)
    joblib.dump({"metrics": metrics, "class_names": dataset.class_names, "task": dataset.task}, args.model_out.with_suffix(".joblib"), compress=3)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved FER CNN to {args.model_out}", flush=True)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.shape[0]
        total_count += int(labels.shape[0])
    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    predictions: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    for images, labels in loader:
        logits = model(images.to(device))
        predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
        labels_out.append(labels.numpy())
    y_pred = np.concatenate(predictions)
    y_true = np.concatenate(labels_out)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


if __name__ == "__main__":
    main()
