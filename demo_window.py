"""Tkinter demo window for the trained CMU-MOSEI fusion model."""

from __future__ import annotations

import argparse
import random
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error


LABEL_NAMES = [
    "Very Negative",
    "Negative",
    "Neutral",
    "Positive",
    "Very Positive",
]


DEFAULT_MODEL = Path("artifacts/mosei_aligned_native_concat_catboost_depth8.joblib")
DEFAULT_CACHE = Path("artifacts/mosei_aligned_native_concat_features.joblib")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open a local demo window for the trained sentiment model.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Saved joblib model artifact.")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="Saved feature cache artifact.")
    parser.add_argument("--smoke-test", action="store_true", help="Load artifacts and print metrics without opening a GUI.")
    return parser.parse_args()


def load_artifacts(model_path: Path, cache_path: Path) -> tuple[object, dict[str, Any], dict[str, np.ndarray]]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not cache_path.exists():
        raise FileNotFoundError(f"Feature cache not found: {cache_path}")

    payload = joblib.load(model_path)
    model = payload["model"]
    metrics = dict(payload.get("metrics", {}))
    features = joblib.load(cache_path)

    metrics["train"] = evaluate_split(model, features["X_train"], features["y_train"])
    metrics["val"] = evaluate_split(model, features["X_val"], features["y_val"])
    metrics["test"] = evaluate_split(model, features["X_test"], features["y_test"])
    return model, metrics, features


def evaluate_split(model: object, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    predictions = np.asarray(model.predict(features)).reshape(-1).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "mae": float(mean_absolute_error(labels, predictions)),
    }


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


class DemoWindow:
    def __init__(
        self,
        root: tk.Tk,
        model: object,
        metrics: dict[str, Any],
        features: dict[str, np.ndarray],
        model_path: Path,
    ) -> None:
        self.root = root
        self.model = model
        self.metrics = metrics
        self.features = features
        self.model_path = model_path
        self.sample_index = tk.IntVar(value=0)
        self.prediction_text = tk.StringVar(value="Select a sample and run prediction.")
        self.detail_text = tk.StringVar(value="")
        self.status_text = tk.StringVar(value=f"Loaded model: {model_path}")
        self.probability_vars = [tk.DoubleVar(value=0.0) for _ in LABEL_NAMES]
        self.probability_labels = [tk.StringVar(value="0.00%") for _ in LABEL_NAMES]

        self.root.title("Multimodal Emotion Recognition Demo")
        self.root.geometry("820x620")
        self.root.minsize(760, 560)
        self._build()
        self.predict_current()

    def _build(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, padding=(22, 18, 22, 8))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        title = ttk.Label(header, text="CMU-MOSEI Multimodal Emotion Recognition", font=("Segoe UI", 18, "bold"))
        title.grid(row=0, column=0, sticky="w")
        subtitle = ttk.Label(
            header,
            text="Final model demo using the saved CatBoost fusion classifier and MOSEI test features.",
            font=("Segoe UI", 10),
        )
        subtitle.grid(row=1, column=0, sticky="w", pady=(4, 0))

        body = ttk.Frame(self.root, padding=(22, 8, 22, 12))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(1, weight=1)

        metrics_frame = ttk.LabelFrame(body, text="Accuracy Summary", padding=14)
        metrics_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        for column in range(3):
            metrics_frame.columnconfigure(column, weight=1)

        self._metric_card(metrics_frame, 0, "Training accuracy", percent(self.metrics["train"]["accuracy"]), "Colab-style fit score")
        self._metric_card(metrics_frame, 1, "Validation accuracy", percent(self.metrics["val"]["accuracy"]), "Model selection score")
        self._metric_card(metrics_frame, 2, "Final test accuracy", percent(self.metrics["test"]["accuracy"]), "Report this number")

        controls = ttk.LabelFrame(body, text="Test Sample Prediction", padding=14)
        controls.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        controls.columnconfigure(1, weight=1)

        max_index = int(self.features["X_test"].shape[0] - 1)
        ttk.Label(controls, text="Sample index").grid(row=0, column=0, sticky="w")
        sample = ttk.Spinbox(controls, from_=0, to=max_index, textvariable=self.sample_index, width=12)
        sample.grid(row=0, column=1, sticky="ew", padx=(10, 0))

        button_row = ttk.Frame(controls)
        button_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        button_row.columnconfigure(0, weight=1)
        button_row.columnconfigure(1, weight=1)
        ttk.Button(button_row, text="Random Sample", command=self.random_sample).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(button_row, text="Predict", command=self.predict_current).grid(row=0, column=1, sticky="ew", padx=(6, 0))

        result = ttk.Label(controls, textvariable=self.prediction_text, font=("Segoe UI", 15, "bold"), wraplength=330)
        result.grid(row=2, column=0, columnspan=2, sticky="w", pady=(22, 8))
        details = ttk.Label(controls, textvariable=self.detail_text, font=("Segoe UI", 10), wraplength=330)
        details.grid(row=3, column=0, columnspan=2, sticky="w")

        note = ttk.Label(
            controls,
            text=(
                "Use final test accuracy for the project report. Training accuracy is shown only because "
                "many notebook demos display it."
            ),
            wraplength=330,
        )
        note.grid(row=4, column=0, columnspan=2, sticky="sw", pady=(28, 0))

        probabilities = ttk.LabelFrame(body, text="Prediction Confidence", padding=14)
        probabilities.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        probabilities.columnconfigure(1, weight=1)
        for index, label in enumerate(LABEL_NAMES):
            ttk.Label(probabilities, text=label).grid(row=index, column=0, sticky="w", pady=6)
            ttk.Progressbar(
                probabilities,
                variable=self.probability_vars[index],
                maximum=1.0,
                length=240,
            ).grid(row=index, column=1, sticky="ew", padx=10, pady=6)
            ttk.Label(probabilities, textvariable=self.probability_labels[index], width=8).grid(
                row=index,
                column=2,
                sticky="e",
                pady=6,
            )

        footer = ttk.Frame(self.root, padding=(22, 0, 22, 14))
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.status_text, font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w")

    def _metric_card(self, parent: ttk.Frame, column: int, title: str, value: str, caption: str) -> None:
        frame = ttk.Frame(parent, padding=10)
        frame.grid(row=0, column=column, sticky="ew", padx=6)
        ttk.Label(frame, text=title, font=("Segoe UI", 10)).grid(row=0, column=0, sticky="w")
        ttk.Label(frame, text=value, font=("Segoe UI", 20, "bold")).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Label(frame, text=caption, font=("Segoe UI", 9)).grid(row=2, column=0, sticky="w")

    def random_sample(self) -> None:
        self.sample_index.set(random.randint(0, int(self.features["X_test"].shape[0] - 1)))
        self.predict_current()

    def predict_current(self) -> None:
        try:
            index = int(self.sample_index.get())
            max_index = int(self.features["X_test"].shape[0] - 1)
            if index < 0 or index > max_index:
                raise ValueError(f"Sample index must be between 0 and {max_index}.")

            row = self.features["X_test"][index : index + 1]
            true_label = int(self.features["y_test"][index])
            predicted_label = int(np.asarray(self.model.predict(row)).reshape(-1)[0])
            probabilities = self._predict_probabilities(row)
            confidence = float(probabilities[predicted_label]) if probabilities is not None else 0.0

            self.prediction_text.set(f"Predicted: {LABEL_NAMES[predicted_label]}")
            self.detail_text.set(
                f"Ground truth: {LABEL_NAMES[true_label]}\n"
                f"Confidence: {percent(confidence)}\n"
                f"Correct: {'Yes' if predicted_label == true_label else 'No'}"
            )
            self._set_probabilities(probabilities, predicted_label)
        except Exception as exc:
            messagebox.showerror("Prediction failed", str(exc))

    def _predict_probabilities(self, row: np.ndarray) -> np.ndarray | None:
        if not hasattr(self.model, "predict_proba"):
            return None
        probabilities = np.asarray(self.model.predict_proba(row)).reshape(-1)
        if probabilities.shape[0] != len(LABEL_NAMES):
            return None
        return probabilities

    def _set_probabilities(self, probabilities: np.ndarray | None, predicted_label: int) -> None:
        if probabilities is None:
            probabilities = np.zeros(len(LABEL_NAMES), dtype=np.float32)
            probabilities[predicted_label] = 1.0
        for index, probability in enumerate(probabilities):
            value = float(probability)
            self.probability_vars[index].set(value)
            self.probability_labels[index].set(percent(value))


def run_smoke_test(model: object, metrics: dict[str, Any], features: dict[str, np.ndarray]) -> None:
    row = features["X_test"][0:1]
    prediction = int(np.asarray(model.predict(row)).reshape(-1)[0])
    truth = int(features["y_test"][0])
    print(f"Train accuracy: {percent(metrics['train']['accuracy'])}")
    print(f"Validation accuracy: {percent(metrics['val']['accuracy'])}")
    print(f"Final test accuracy: {percent(metrics['test']['accuracy'])}")
    print(f"Sample 0 prediction: {LABEL_NAMES[prediction]} | truth: {LABEL_NAMES[truth]}")


def main() -> None:
    args = parse_args()
    model, metrics, features = load_artifacts(args.model, args.cache)
    if args.smoke_test:
        run_smoke_test(model, metrics, features)
        return

    root = tk.Tk()
    DemoWindow(root, model, metrics, features, args.model)
    root.mainloop()


if __name__ == "__main__":
    main()
