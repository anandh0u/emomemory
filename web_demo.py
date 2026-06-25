"""Browser demo for the trained CMU-MOSEI fusion model.

This is a dependency-light local deployment: it uses Python's stdlib HTTP
server and the already-trained joblib artifacts.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

from data.fer_loader import extract_fer_features, frame_to_face_or_gray
from data.savee_loader import extract_audio_features
from data.video_processor import VideoProcessor


LABEL_NAMES = [
    "Very Negative",
    "Negative",
    "Neutral",
    "Positive",
    "Very Positive",
]

DEFAULT_MODEL = Path("artifacts/mosei_aligned_native_concat_catboost_depth8.joblib")
DEFAULT_CACHE = Path("artifacts/mosei_aligned_native_concat_features.joblib")
DEFAULT_FER_MODEL = Path("artifacts/fer_binary_cnn.pt")
DEFAULT_SAVEE_MODEL = Path("artifacts/savee_audio_best_search.joblib")
UPLOAD_DIR = Path("outputs/uploads")
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
AUDIO_SUFFIXES = {".flac", ".m4a", ".mp3", ".ogg", ".wav"}
VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the local MMER browser demo.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Saved model artifact.")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="Saved feature cache artifact.")
    parser.add_argument("--fer-model", type=Path, default=DEFAULT_FER_MODEL, help="Optional saved FER model artifact.")
    parser.add_argument("--savee-model", type=Path, default=DEFAULT_SAVEE_MODEL, help="Optional saved SAVEE audio model artifact.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind.")
    parser.add_argument("--smoke-test", action="store_true", help="Load artifacts and print a sample prediction.")
    return parser.parse_args()


def load_artifacts(
    model_path: Path,
    cache_path: Path,
    fer_model_path: Path = DEFAULT_FER_MODEL,
    savee_model_path: Path = DEFAULT_SAVEE_MODEL,
) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not cache_path.exists():
        raise FileNotFoundError(f"Feature cache not found: {cache_path}")

    model_payload = joblib.load(model_path)
    features = joblib.load(cache_path)
    model = model_payload["model"]
    metrics = {
        "train": evaluate_split(model, features["X_train"], features["y_train"]),
        "validation": evaluate_split(model, features["X_val"], features["y_val"]),
        "test": evaluate_split(model, features["X_test"], features["y_test"]),
    }
    return {
        "model": model,
        "features": features,
        "metrics": metrics,
        "model_path": str(model_path),
        "cache_path": str(cache_path),
        "fer": load_fer_artifact(fer_model_path),
        "savee": load_savee_artifact(savee_model_path),
    }


def load_fer_artifact(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if path.suffix.lower() == ".pt":
        torch = _import_torch()
        from train_fer_cnn import SmallFERNet

        payload = torch.load(path, map_location="cpu")
        class_names = list(payload.get("class_names", []))
        model = SmallFERNet(len(class_names))
        model.load_state_dict(payload["model_state"])
        model.eval()
        return {
            "model": model,
            "kind": "torch_cnn",
            "metrics": payload.get("metrics", {}),
            "class_names": class_names,
            "feature_size": 48,
            "task": payload.get("task", "binary"),
            "path": str(path),
        }
    payload = joblib.load(path)
    return {
        "model": payload["model"],
        "kind": "sklearn",
        "metrics": payload.get("metrics", {}),
        "class_names": list(payload.get("class_names", [])),
        "feature_size": int(payload.get("feature_size", 24)),
        "task": payload.get("task", "fer7"),
        "path": str(path),
    }


def load_savee_artifact(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = joblib.load(path)
    return {
        "model": payload["model"],
        "metrics": payload.get("metrics", {}),
        "class_names": list(payload.get("class_names", [])),
        "sample_rate": int(payload.get("sample_rate", 16_000)),
        "path": str(path),
    }


def evaluate_split(model: object, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    predictions = np.asarray(model.predict(features)).reshape(-1).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "mae": float(mean_absolute_error(labels, predictions)),
    }


def predict_sample(state: dict[str, Any], index: int | None = None) -> dict[str, Any]:
    features = state["features"]
    model = state["model"]
    max_index = int(features["X_test"].shape[0] - 1)
    if index is None:
        index = random.randint(0, max_index)
    if index < 0 or index > max_index:
        raise ValueError(f"Sample index must be between 0 and {max_index}.")

    row = features["X_test"][index : index + 1]
    truth = int(features["y_test"][index])
    prediction = int(np.asarray(model.predict(row)).reshape(-1)[0])
    probabilities = predict_probabilities(model, row, prediction)
    return {
        "index": index,
        "truth": truth,
        "truth_name": LABEL_NAMES[truth],
        "prediction": prediction,
        "prediction_name": LABEL_NAMES[prediction],
        "correct": bool(prediction == truth),
        "confidence": float(probabilities[prediction]),
        "probabilities": [
            {"label": label, "value": float(value)}
            for label, value in zip(LABEL_NAMES, probabilities, strict=True)
        ],
    }


def predict_probabilities(model: object, row: np.ndarray, prediction: int) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(row)).reshape(-1).astype(float)
        if probabilities.shape[0] == len(LABEL_NAMES):
            return probabilities
    probabilities = np.zeros(len(LABEL_NAMES), dtype=float)
    probabilities[prediction] = 1.0
    return probabilities


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


class DemoHandler(BaseHTTPRequestHandler):
    state: dict[str, Any] = {}

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self.send_html(render_chat_page())
            elif parsed.path == "/health":
                self.send_json({"status": "ok"})
            elif parsed.path == "/api/summary":
                features = self.state["features"]
                fer = self.state.get("fer")
                savee = self.state.get("savee")
                self.send_json(
                    {
                        "metrics": self.state["metrics"],
                        "label_names": LABEL_NAMES,
                        "test_count": int(features["X_test"].shape[0]),
                        "feature_count": int(features["X_test"].shape[1]),
                        "model_path": self.state["model_path"],
                        "cache_path": self.state["cache_path"],
                        "fer": summarize_fer(fer),
                        "savee": summarize_savee(savee),
                    }
                )
            elif parsed.path == "/api/predict":
                query = parse_qs(parsed.query)
                index = None
                if query.get("index", [""])[0] != "":
                    index = int(query["index"][0])
                self.send_json(predict_sample(self.state, index))
            else:
                self.send_error(404, "Not found")
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/chat":
                fields, files = parse_multipart_form(self)
                self.send_json(handle_chat_request(self.state, fields, files))
            elif parsed.path == "/api/upload-video":
                saved_path = save_uploaded_video(self)
                processed = process_uploaded_video(saved_path, self.state)
                self.send_json(
                    {
                        "filename": saved_path.name,
                        "saved_path": str(saved_path),
                        "frames_dir": str(processed["frames_dir"]),
                        "audio_path": str(processed["audio_path"]) if processed["audio_path"] else None,
                        "audio_error": processed["audio_error"],
                        "frame_count": processed["frame_count"],
                        "source_fps": processed["source_fps"],
                        "duration_seconds": processed["duration_seconds"],
                        "sample_rate": processed["sample_rate"],
                        "fer_prediction": processed["fer_prediction"],
                        "savee_prediction": processed["savee_prediction"],
                        "message": (
                            "Video uploaded and preprocessed. The final trained classifier uses CMU-MOSEI "
                            "OpenFace/COVAREP/GloVe feature vectors, so raw-video scoring needs those "
                            "extractors before it can be an honest final-model prediction."
                        ),
                    }
                )
            elif parsed.path == "/api/upload-face":
                saved_path = save_uploaded_video(self)
                self.send_json(
                    {
                        "filename": saved_path.name,
                        "saved_path": str(saved_path),
                        "fer_prediction": predict_fer_image_path(self.state, saved_path),
                    }
                )
            elif parsed.path == "/api/upload-audio":
                saved_path = save_uploaded_video(self)
                self.send_json(
                    {
                        "filename": saved_path.name,
                        "saved_path": str(saved_path),
                        "savee_prediction": predict_savee_audio_path(self.state, saved_path),
                    }
                )
            else:
                self.send_error(404, "Not found")
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def save_uploaded_video(handler: BaseHTTPRequestHandler) -> Path:
    content_type = handler.headers.get("Content-Type", "")
    if "multipart/form-data" not in content_type or "boundary=" not in content_type:
        raise ValueError("Expected multipart/form-data upload.")

    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        raise ValueError("Upload body is empty.")
    if length > MAX_UPLOAD_BYTES:
        raise ValueError("Upload is too large. Keep videos under 500 MB.")

    boundary = content_type.split("boundary=", 1)[1].strip().strip('"').encode("utf-8")
    body = handler.rfile.read(length)
    delimiter = b"--" + boundary
    for raw_part in body.split(delimiter):
        part = raw_part.strip(b"\r\n")
        if not part or part == b"--" or b"\r\n\r\n" not in part:
            continue
        raw_headers, content = part.split(b"\r\n\r\n", 1)
        headers = raw_headers.decode("utf-8", errors="ignore")
        has_video_field = re.search(r'name="?video"?', headers, flags=re.IGNORECASE) is not None
        has_file = re.search(r"filename=", headers, flags=re.IGNORECASE) is not None
        if not has_video_field and not has_file:
            continue
        filename_match = re.search(r'filename="?([^";\r\n]+)"?', headers, flags=re.IGNORECASE)
        original_name = filename_match.group(1) if filename_match else "uploaded_video.mp4"
        safe_name = sanitize_filename(original_name)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        saved_path = UPLOAD_DIR / f"{int(time.time())}_{safe_name}"
        saved_path.write_bytes(content.rstrip(b"\r\n"))
        return saved_path

    raise ValueError("No video file field named 'video' was found.")


def parse_multipart_form(handler: BaseHTTPRequestHandler) -> tuple[dict[str, str], dict[str, Path]]:
    content_type = handler.headers.get("Content-Type", "")
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}, {}
    if length > MAX_UPLOAD_BYTES:
        raise ValueError("Upload is too large. Keep files under 500 MB.")

    body = handler.rfile.read(length)
    if "multipart/form-data" not in content_type:
        if "application/json" in content_type:
            payload = json.loads(body.decode("utf-8", errors="replace") or "{}")
            return {str(key): str(value) for key, value in payload.items()}, {}
        return {"message": body.decode("utf-8", errors="replace").strip()}, {}

    if "boundary=" not in content_type:
        raise ValueError("Expected multipart/form-data boundary.")

    boundary = content_type.split("boundary=", 1)[1].strip().strip('"').encode("utf-8")
    delimiter = b"--" + boundary
    fields: dict[str, str] = {}
    files: dict[str, Path] = {}

    for raw_part in body.split(delimiter):
        part = raw_part.strip(b"\r\n")
        if not part or part == b"--" or b"\r\n\r\n" not in part:
            continue
        raw_headers, content = part.split(b"\r\n\r\n", 1)
        headers = raw_headers.decode("utf-8", errors="ignore")
        name_match = re.search(r'name="?([^";\r\n]+)"?', headers, flags=re.IGNORECASE)
        if name_match is None:
            continue
        field_name = name_match.group(1)
        filename_match = re.search(r'filename="?([^";\r\n]*)"?', headers, flags=re.IGNORECASE)
        content = content.rstrip(b"\r\n")
        if filename_match and filename_match.group(1):
            safe_name = sanitize_filename(filename_match.group(1))
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            saved_path = UPLOAD_DIR / f"{time.time_ns()}_{safe_name}"
            saved_path.write_bytes(content)
            files[field_name] = saved_path
        else:
            fields[field_name] = content.decode("utf-8", errors="replace").strip()

    return fields, files


def handle_chat_request(state: dict[str, Any], fields: dict[str, str], files: dict[str, Path]) -> dict[str, Any]:
    message = (fields.get("message") or fields.get("text") or "").strip()
    uploaded_path = next(iter(files.values()), None)
    parts: list[str] = []
    channels: list[dict[str, Any]] = []

    if message:
        text_prediction = predict_text_emotion(message)
        channels.append({"kind": "text", "prediction": text_prediction["prediction_name"]})
        parts.append(
            "Text model: "
            f"I read this as {text_prediction['prediction_name']} "
            f"with a {text_prediction['polarity']} tone."
        )

    if uploaded_path is not None:
        file_result = analyze_uploaded_file(state, uploaded_path)
        channels.append(file_result["channel"])
        parts.append(file_result["message"])

    if not parts:
        parts.append("Type a message or attach an image, audio file, or video so I can analyze it.")

    return {
        "reply": "\n".join(parts),
        "channels": channels,
        "file": str(uploaded_path) if uploaded_path else None,
    }


def analyze_uploaded_file(state: dict[str, Any], path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        prediction = predict_fer_image_path(state, path)
        if not prediction.get("available", False):
            return {"channel": {"kind": "image", "prediction": None}, "message": prediction["message"]}
        return {
            "channel": {"kind": "image", "prediction": prediction["prediction_name"]},
            "message": f"Image model: I read the face as {prediction['prediction_name']}.",
        }

    if suffix in AUDIO_SUFFIXES:
        prediction = predict_savee_audio_path(state, path)
        if prediction is None:
            return {"channel": {"kind": "audio", "prediction": None}, "message": "Audio model: no audio was found."}
        if not prediction.get("available", False):
            return {"channel": {"kind": "audio", "prediction": None}, "message": prediction["message"]}
        return {
            "channel": {"kind": "audio", "prediction": prediction["prediction_name"]},
            "message": f"Audio model: I read the voice as {prediction['prediction_name']}.",
        }

    if suffix in VIDEO_SUFFIXES:
        processed = process_uploaded_video(path, state)
        details = [
            f"Video model: I processed {processed['frame_count']} sampled frame(s).",
        ]
        frame_prediction = processed.get("fer_prediction")
        if frame_prediction and frame_prediction.get("available", False):
            details.append(f"Frame model read the visible face emotion as {frame_prediction['prediction_name']}.")
        audio_prediction = processed.get("savee_prediction")
        if audio_prediction and audio_prediction.get("available", False):
            details.append(f"Audio track model read the voice as {audio_prediction['prediction_name']}.")
        elif processed.get("audio_error"):
            details.append("No usable audio track was extracted from this video.")
        return {
            "channel": {
                "kind": "video",
                "frames": processed["frame_count"],
                "face_prediction": frame_prediction.get("prediction_name") if frame_prediction else None,
                "audio_prediction": audio_prediction.get("prediction_name") if audio_prediction else None,
            },
            "message": " ".join(details),
        }

    supported = ", ".join(sorted(IMAGE_SUFFIXES | AUDIO_SUFFIXES | VIDEO_SUFFIXES))
    return {
        "channel": {"kind": "unknown", "prediction": None},
        "message": f"I saved the file, but this file type is not supported yet. Supported types: {supported}.",
    }


def predict_text_emotion(message: str) -> dict[str, str]:
    normalized = message.lower()
    lexicon = {
        "joy": ["happy", "joy", "great", "good", "love", "excited", "proud", "smile", "amazing", "thanks"],
        "sadness": ["sad", "cry", "hurt", "lonely", "tired", "depressed", "upset", "miss", "bad"],
        "anger": ["angry", "mad", "furious", "hate", "annoyed", "irritated", "rage", "stupid"],
        "fear": ["scared", "afraid", "fear", "worried", "anxious", "panic", "nervous"],
        "surprise": ["surprised", "wow", "shocked", "unexpected", "sudden"],
    }
    scores = {
        label: sum(len(re.findall(rf"\b{re.escape(word)}\b", normalized)) for word in words)
        for label, words in lexicon.items()
    }
    prediction = max(scores, key=scores.get)
    if scores[prediction] == 0:
        prediction = "neutral"
    polarity = "positive" if prediction in {"joy", "surprise"} else "negative" if prediction != "neutral" else "neutral"
    return {"prediction_name": prediction, "polarity": polarity}


def process_uploaded_video(saved_path: Path, state: dict[str, Any] | None = None) -> dict[str, Any]:
    processor = VideoProcessor(frame_fps=1.0, sample_rate=16_000)
    run_dir = UPLOAD_DIR / f"{saved_path.stem}_{int(time.time())}"
    frames_dir = run_dir / "frames"
    audio_path = run_dir / f"{saved_path.stem}.wav"
    run_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_paths, source_fps, duration_seconds = processor.extract_frames(saved_path, frames_dir)
    audio_error = None
    try:
        processor.extract_audio(saved_path, audio_path)
    except Exception as exc:
        audio_error = str(exc)
        audio_path = None

    return {
        "frames_dir": frames_dir,
        "audio_path": audio_path,
        "audio_error": audio_error,
        "frame_count": len(frame_paths),
        "source_fps": source_fps,
        "duration_seconds": duration_seconds,
        "sample_rate": processor.sample_rate,
        "fer_prediction": predict_fer_frames(state or {}, frame_paths),
        "savee_prediction": predict_savee_audio_path(state or {}, audio_path) if audio_path else None,
    }


def summarize_fer(fer: dict[str, Any] | None) -> dict[str, Any]:
    if fer is None:
        return {
            "available": False,
            "message": (
                "FER model not trained yet. Run: .\\.venv\\Scripts\\python.exe train_fer.py "
                "--data-path E:\\emotion_recognition_internship\\data\\raw\\fer2013.csv "
                "--task binary --classifier mlp --max-iter 40 "
                "--model-out artifacts\\fer_binary_mlp_classifier.joblib"
            ),
        }
    return {
        "available": True,
        "task": fer["task"],
        "class_names": fer["class_names"],
        "metrics": fer["metrics"],
        "model_path": fer["path"],
    }


def summarize_savee(savee: dict[str, Any] | None) -> dict[str, Any]:
    if savee is None:
        return {
            "available": False,
            "message": (
                "SAVEE model not trained yet. Run: .\\.venv\\Scripts\\python.exe "
                "train_savee.py --task binary --classifier rf --model-out artifacts\\savee_binary_rf_classifier.joblib"
            ),
        }
    return {
        "available": True,
        "class_names": savee["class_names"],
        "metrics": savee["metrics"],
        "model_path": savee["path"],
    }


def predict_fer_image_path(state: dict[str, Any], path: Path) -> dict[str, Any]:
    cv2 = _import_cv2()
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read uploaded image/video frame: {path}")
    return predict_fer_image(state, image)


def predict_fer_frames(state: dict[str, Any], frame_paths: list[Path], max_frames: int = 8) -> dict[str, Any] | None:
    fer = state.get("fer")
    if fer is None or not frame_paths:
        return None
    cv2 = _import_cv2()
    selected = evenly_spaced(frame_paths, max_frames)
    probabilities: list[np.ndarray] = []
    frames_used = 0
    for frame_path in selected:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        prediction = predict_fer_image(state, frame)
        probabilities.append(np.asarray([item["value"] for item in prediction["probabilities"]], dtype=float))
        frames_used += 1
    if not probabilities:
        return None
    mean_probabilities = np.mean(np.stack(probabilities, axis=0), axis=0)
    return format_fer_prediction(fer, mean_probabilities, frames_used=frames_used)


def predict_fer_image(state: dict[str, Any], frame: np.ndarray) -> dict[str, Any]:
    fer = state.get("fer")
    if fer is None:
        return {
            "available": False,
            "message": summarize_fer(None)["message"],
        }
    gray = frame_to_face_or_gray(frame, image_size=48)
    if fer.get("kind") == "torch_cnn":
        probabilities = predict_fer_cnn_probabilities(fer["model"], gray)
        return format_fer_prediction(fer, probabilities)
    features = extract_fer_features(np.stack([gray], axis=0), feature_size=fer["feature_size"])
    probabilities = predict_generic_probabilities(fer["model"], features, len(fer["class_names"]))
    return format_fer_prediction(fer, probabilities)


def predict_fer_cnn_probabilities(model: object, gray: np.ndarray) -> np.ndarray:
    torch = _import_torch()
    image = np.asarray(gray, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy().reshape(-1)
    return probabilities.astype(float)


def predict_savee_audio_path(state: dict[str, Any], path: Path | None) -> dict[str, Any] | None:
    savee = state.get("savee")
    if path is None:
        return None
    if savee is None:
        return {
            "available": False,
            "message": summarize_savee(None)["message"],
        }
    features = extract_audio_features(path, sample_rate=savee["sample_rate"]).reshape(1, -1)
    probabilities = predict_generic_probabilities(savee["model"], features, len(savee["class_names"]))
    return format_savee_prediction(savee, probabilities)


def format_fer_prediction(fer: dict[str, Any], probabilities: np.ndarray, frames_used: int | None = None) -> dict[str, Any]:
    class_names = fer["class_names"]
    prediction = int(np.argmax(probabilities))
    payload = {
        "available": True,
        "task": fer["task"],
        "prediction": prediction,
        "prediction_name": class_names[prediction],
        "confidence": float(probabilities[prediction]),
        "probabilities": [
            {"label": label, "value": float(value)}
            for label, value in zip(class_names, probabilities, strict=True)
        ],
    }
    if frames_used is not None:
        payload["frames_used"] = frames_used
    return payload


def format_savee_prediction(savee: dict[str, Any], probabilities: np.ndarray) -> dict[str, Any]:
    class_names = savee["class_names"]
    prediction = int(np.argmax(probabilities))
    return {
        "available": True,
        "prediction": prediction,
        "prediction_name": class_names[prediction],
        "confidence": float(probabilities[prediction]),
        "probabilities": [
            {"label": label, "value": float(value)}
            for label, value in zip(class_names, probabilities, strict=True)
        ],
    }


def predict_generic_probabilities(model: object, features: np.ndarray, class_count: int) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features)).reshape(-1).astype(float)
        if probabilities.shape[0] == class_count:
            return probabilities
    prediction = int(np.asarray(model.predict(features)).reshape(-1)[0])
    probabilities = np.zeros(class_count, dtype=float)
    probabilities[prediction] = 1.0
    return probabilities


def evenly_spaced(values: list[Path], limit: int) -> list[Path]:
    if len(values) <= limit:
        return values
    indices = np.linspace(0, len(values) - 1, num=limit, dtype=int)
    return [values[int(index)] for index in indices]


def _import_cv2() -> object:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for upload prediction. Install requirements.txt first.") from exc
    return cv2


def _import_torch() -> object:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required for CNN upload prediction. Install requirements.txt first.") from exc
    return torch


def sanitize_filename(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or "uploaded_video.mp4"


def render_chat_page() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Multimodal Emotion Chat</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f8fb;
      --panel: #ffffff;
      --ink: #18212f;
      --muted: #667085;
      --line: #d7dee8;
      --soft: #eef3f7;
      --accent: #0f766e;
      --accent-2: #b45309;
      --user: #e7f5f2;
      --assistant: #ffffff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: "Segoe UI", Arial, sans-serif;
      letter-spacing: 0;
    }
    .app {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
    }
    header {
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.92);
      padding: 14px clamp(14px, 4vw, 34px);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    h1 {
      margin: 0;
      font-size: clamp(19px, 2.4vw, 27px);
      line-height: 1.15;
      font-weight: 760;
    }
    .status {
      color: var(--accent);
      font-size: 14px;
      font-weight: 700;
      white-space: nowrap;
    }
    .chat {
      width: min(920px, calc(100vw - 24px));
      margin: 0 auto;
      padding: 20px 0 18px;
      overflow-y: auto;
    }
    .message {
      display: grid;
      grid-template-columns: 38px minmax(0, 1fr);
      gap: 10px;
      margin: 0 0 14px;
      align-items: start;
    }
    .avatar {
      width: 38px;
      height: 38px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      font-weight: 800;
      color: #fff;
      background: var(--accent);
    }
    .message.user .avatar {
      background: var(--accent-2);
    }
    .bubble {
      width: fit-content;
      max-width: min(720px, 100%);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      background: var(--assistant);
      line-height: 1.5;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    .message.user .bubble {
      background: var(--user);
      justify-self: end;
    }
    .file-note {
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
      white-space: normal;
    }
    .composer-wrap {
      border-top: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.95);
      padding: 12px clamp(12px, 4vw, 34px) 18px;
    }
    .composer {
      width: min(920px, 100%);
      margin: 0 auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 10px;
      display: grid;
      grid-template-columns: auto minmax(0, 1fr) auto;
      gap: 10px;
      align-items: end;
    }
    textarea {
      width: 100%;
      min-height: 44px;
      max-height: 160px;
      resize: vertical;
      border: 0;
      outline: 0;
      padding: 11px 2px;
      font: inherit;
      line-height: 1.35;
      color: var(--ink);
    }
    button,
    .file-button {
      min-height: 42px;
      border: 0;
      border-radius: 8px;
      padding: 0 14px;
      font: inherit;
      font-weight: 750;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      white-space: nowrap;
    }
    button {
      background: var(--accent);
      color: #fff;
    }
    button:disabled {
      opacity: 0.58;
      cursor: wait;
    }
    .file-button {
      background: var(--soft);
      color: var(--ink);
      border: 1px solid var(--line);
    }
    input[type="file"] {
      position: absolute;
      width: 1px;
      height: 1px;
      opacity: 0;
      pointer-events: none;
    }
    .attachment {
      width: min(920px, 100%);
      margin: 8px auto 0;
      color: var(--muted);
      font-size: 13px;
      min-height: 18px;
    }
    .dots {
      display: inline-flex;
      gap: 5px;
      align-items: center;
      min-width: 44px;
    }
    .dots span {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--accent);
      animation: pulse 900ms infinite ease-in-out;
    }
    .dots span:nth-child(2) { animation-delay: 120ms; }
    .dots span:nth-child(3) { animation-delay: 240ms; }
    @keyframes pulse {
      0%, 80%, 100% { opacity: 0.35; transform: translateY(0); }
      40% { opacity: 1; transform: translateY(-4px); }
    }
    @media (max-width: 640px) {
      header {
        align-items: flex-start;
        flex-direction: column;
      }
      .composer {
        grid-template-columns: 1fr auto;
      }
      textarea {
        grid-column: 1 / -1;
        order: -1;
      }
      button,
      .file-button {
        width: 100%;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <h1>Multimodal Emotion Chat</h1>
      <div class="status" id="status">Ready</div>
    </header>

    <main class="chat" id="chat">
      <section class="message assistant">
        <div class="avatar">AI</div>
        <div class="bubble">Send a message, image, audio file, or video. I will route it to the matching text, face, audio, or video model and show the answer here.</div>
      </section>
    </main>

    <form class="composer-wrap" id="chatForm">
      <div class="composer">
        <label class="file-button" for="fileInput" title="Attach image, audio, or video">
          <span aria-hidden="true">+</span>
          <span>Attach</span>
        </label>
        <input id="fileInput" name="file" type="file" accept="image/*,audio/*,video/*,.wav,.mp3,.m4a,.flac,.mp4,.mov,.mkv,.webm">
        <textarea id="messageInput" name="message" placeholder="Type your message..." aria-label="Message"></textarea>
        <button id="sendButton" type="submit">Send</button>
      </div>
      <div class="attachment" id="attachment"></div>
    </form>
  </div>

  <script>
    const chat = document.getElementById("chat");
    const form = document.getElementById("chatForm");
    const input = document.getElementById("messageInput");
    const fileInput = document.getElementById("fileInput");
    const attachment = document.getElementById("attachment");
    const sendButton = document.getElementById("sendButton");
    const statusText = document.getElementById("status");
    let loadingNode = null;

    function addMessage(role, text, fileName = "") {
      const section = document.createElement("section");
      section.className = `message ${role}`;
      const avatar = role === "user" ? "You" : "AI";
      const fileMarkup = fileName ? `<div class="file-note">Attached: ${escapeHtml(fileName)}</div>` : "";
      section.innerHTML = `
        <div class="avatar">${avatar}</div>
        <div class="bubble">${escapeHtml(text || "Attached file").replace(/\\n/g, "<br>")}${fileMarkup}</div>
      `;
      chat.appendChild(section);
      chat.scrollTop = chat.scrollHeight;
      return section;
    }

    function showLoading() {
      loadingNode = document.createElement("section");
      loadingNode.className = "message assistant";
      loadingNode.innerHTML = `
        <div class="avatar">AI</div>
        <div class="bubble"><span class="dots"><span></span><span></span><span></span></span></div>
      `;
      chat.appendChild(loadingNode);
      chat.scrollTop = chat.scrollHeight;
    }

    function hideLoading() {
      if (loadingNode) {
        loadingNode.remove();
        loadingNode = null;
      }
    }

    function escapeHtml(value) {
      return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }

    fileInput.addEventListener("change", () => {
      attachment.textContent = fileInput.files.length ? `Attached: ${fileInput.files[0].name}` : "";
    });

    input.addEventListener("keydown", event => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
      }
    });

    form.addEventListener("submit", async event => {
      event.preventDefault();
      const message = input.value.trim();
      const file = fileInput.files[0];
      if (!message && !file) return;

      addMessage("user", message, file ? file.name : "");
      input.value = "";
      fileInput.value = "";
      attachment.textContent = "";
      sendButton.disabled = true;
      statusText.textContent = "Analyzing";
      showLoading();

      const body = new FormData();
      body.append("message", message);
      if (file) body.append("file", file);

      try {
        const response = await fetch("/api/chat", { method: "POST", body });
        const result = await response.json();
        if (!response.ok || result.error) throw new Error(result.error || "Request failed");
        hideLoading();
        addMessage("assistant", result.reply);
      } catch (error) {
        hideLoading();
        addMessage("assistant", error.message);
      } finally {
        sendButton.disabled = false;
        statusText.textContent = "Ready";
        input.focus();
      }
    });
  </script>
</body>
</html>
"""


def render_page() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MMER Demo</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fa;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #5f6b7a;
      --line: #d9e0e8;
      --accent: #1769aa;
      --accent-2: #0a8f6a;
      --warn: #b85000;
      --bad: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: "Segoe UI", Arial, sans-serif;
      letter-spacing: 0;
    }
    main {
      width: min(1120px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 24px 0;
    }
    header {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 24px;
      padding: 4px 0 18px;
      border-bottom: 1px solid var(--line);
    }
    h1 {
      margin: 0;
      font-size: clamp(24px, 3vw, 34px);
      line-height: 1.12;
      font-weight: 750;
    }
    .subtitle {
      margin: 8px 0 0;
      color: var(--muted);
      max-width: 720px;
      line-height: 1.5;
    }
    .status {
      min-width: 142px;
      text-align: center;
      border: 1px solid var(--line);
      background: var(--panel);
      padding: 10px 12px;
      border-radius: 8px;
      font-weight: 700;
      color: var(--accent-2);
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin: 18px 0;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    .metric-title {
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
      text-transform: uppercase;
    }
    .metric-value {
      margin-top: 8px;
      font-size: clamp(28px, 4vw, 44px);
      font-weight: 800;
      color: var(--accent);
    }
    .metric-caption {
      margin-top: 6px;
      color: var(--muted);
      line-height: 1.45;
    }
    .workspace {
      display: grid;
      grid-template-columns: minmax(300px, 0.88fr) minmax(320px, 1.12fr);
      gap: 12px;
      align-items: stretch;
    }
    h2 {
      margin: 0 0 14px;
      font-size: 18px;
    }
    .control-row {
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 10px;
      align-items: center;
    }
    input {
      height: 40px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 12px;
      font: inherit;
      min-width: 0;
    }
    button {
      height: 40px;
      border: 0;
      border-radius: 6px;
      padding: 0 14px;
      background: var(--accent);
      color: white;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      white-space: nowrap;
    }
    button.secondary {
      background: #27384a;
    }
    .prediction {
      margin-top: 20px;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f9fbfd;
    }
    .prediction strong {
      display: block;
      font-size: 26px;
      margin-top: 4px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      padding: 4px 8px;
      border-radius: 999px;
      background: #e8f4ef;
      color: var(--accent-2);
      font-size: 13px;
      font-weight: 750;
    }
    .badge.bad {
      background: #fdebea;
      color: var(--bad);
    }
    .bars {
      display: grid;
      gap: 12px;
    }
    .bar-row {
      display: grid;
      grid-template-columns: 118px minmax(0, 1fr) 62px;
      gap: 10px;
      align-items: center;
    }
    .bar-label,
    .bar-value {
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
    }
    .track {
      height: 14px;
      border: 1px solid var(--line);
      border-radius: 999px;
      overflow: hidden;
      background: #edf1f5;
    }
    .fill {
      width: 0%;
      height: 100%;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
      transition: width 180ms ease;
    }
    .footnote {
      margin-top: 18px;
      color: var(--muted);
      line-height: 1.5;
      font-size: 14px;
    }
    .upload {
      margin-top: 12px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
    }
    input[type="file"] {
      display: block;
      height: auto;
      padding: 9px;
      background: #fff;
    }
    .upload-result {
      margin-top: 12px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f9fbfd;
      color: var(--muted);
      line-height: 1.5;
      min-height: 56px;
    }
    code {
      background: #eef2f6;
      border-radius: 5px;
      padding: 2px 5px;
    }
    @media (max-width: 820px) {
      header,
      .workspace {
        grid-template-columns: 1fr;
        display: grid;
      }
      .grid {
        grid-template-columns: 1fr;
      }
      .control-row {
        grid-template-columns: 1fr;
      }
      button {
        width: 100%;
      }
      .upload {
        grid-template-columns: 1fr;
      }
      .bar-row {
        grid-template-columns: 104px minmax(0, 1fr) 58px;
      }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Multimodal Emotion Recognition Demo</h1>
        <p class="subtitle">CatBoost fusion over CMU-MOSEI visual, acoustic, and text features. The dashboard reports training accuracy for notebook-style comparison and test accuracy for the final project result.</p>
      </div>
      <div class="status" id="status">Loading</div>
    </header>

    <section class="grid" aria-label="metrics">
      <article class="panel">
        <div class="metric-title">Training Accuracy</div>
        <div class="metric-value" id="trainAccuracy">--</div>
        <div class="metric-caption">Colab-style fit score, not the final report metric.</div>
      </article>
      <article class="panel">
        <div class="metric-title">Validation Accuracy</div>
        <div class="metric-value" id="validationAccuracy">--</div>
        <div class="metric-caption">Used for model selection and tuning.</div>
      </article>
      <article class="panel">
        <div class="metric-title">Final Test Accuracy</div>
        <div class="metric-value" id="testAccuracy">--</div>
        <div class="metric-caption">Use this number in the project evaluation.</div>
      </article>
    </section>

    <section class="workspace">
      <article class="panel">
        <h2>Held-Out Test Prediction</h2>
        <div class="control-row">
          <input id="sampleIndex" type="number" min="0" value="0" aria-label="Sample index">
          <button type="button" onclick="predictFromInput()">Predict</button>
          <button class="secondary" type="button" onclick="predictRandom()">Random</button>
        </div>
        <div class="prediction">
          <span id="correctBadge" class="badge">Waiting</span>
          <strong id="prediction">Select a sample</strong>
          <p id="details" class="footnote"></p>
        </div>
        <p class="footnote">This demo uses held-out MOSEI test features from the saved cache. It is meant for project evaluation, not live webcam/video inference.</p>
      </article>
      <article class="panel">
        <h2>Class Confidence</h2>
        <div class="bars" id="bars"></div>
      </article>
    </section>

    <section class="panel" style="margin-top: 12px;">
      <h2>FER Face Emotion Upload</h2>
      <div class="upload">
        <input id="faceFile" type="file" accept="image/*">
        <button type="button" onclick="uploadFace()">Predict Face</button>
      </div>
      <div class="upload-result" id="faceResult">
        FER model status is loading. Train it with <code>train_fer.py</code> if this panel says no FER model is available.
      </div>
    </section>

    <section class="panel" style="margin-top: 12px;">
      <h2>SAVEE Audio Emotion Upload</h2>
      <div class="upload">
        <input id="audioFile" type="file" accept="audio/*,.wav">
        <button type="button" onclick="uploadAudio()">Predict Audio</button>
      </div>
      <div class="upload-result" id="audioResult">
        SAVEE model status is loading. Train it with <code>train_savee.py</code> if this panel says no SAVEE model is available.
      </div>
    </section>

    <section class="panel" style="margin-top: 12px;">
      <h2>Video Upload Preprocessing + FER Frames</h2>
      <div class="upload">
        <input id="videoFile" type="file" accept="video/*">
        <button type="button" onclick="uploadVideo()">Upload Video</button>
      </div>
      <div class="upload-result" id="uploadResult">
        Upload a video to extract 1 FPS frames and 16 kHz mono audio. If FER/SAVEE models are trained, the demo predicts face emotion from sampled frames and audio emotion from the extracted WAV.
      </div>
    </section>
  </main>

  <script>
    const labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"];
    const pct = value => `${(value * 100).toFixed(2)}%`;

    async function loadSummary() {
      const response = await fetch("/api/summary");
      const summary = await response.json();
      document.getElementById("trainAccuracy").textContent = pct(summary.metrics.train.accuracy);
      document.getElementById("validationAccuracy").textContent = pct(summary.metrics.validation.accuracy);
      document.getElementById("testAccuracy").textContent = pct(summary.metrics.test.accuracy);
      document.getElementById("sampleIndex").max = summary.test_count - 1;
      document.getElementById("status").textContent = "Ready";
      renderFerStatus(summary.fer);
      renderSaveeStatus(summary.savee);
      buildBars(labels.map(label => ({ label, value: 0 })));
      await predictIndex(0);
    }

    function buildBars(probabilities) {
      const root = document.getElementById("bars");
      root.innerHTML = "";
      for (const item of probabilities) {
        const row = document.createElement("div");
        row.className = "bar-row";
        row.innerHTML = `
          <div class="bar-label">${item.label}</div>
          <div class="track"><div class="fill" style="width: ${(item.value * 100).toFixed(2)}%"></div></div>
          <div class="bar-value">${pct(item.value)}</div>
        `;
        root.appendChild(row);
      }
    }

    async function predictIndex(index) {
      const response = await fetch(`/api/predict?index=${encodeURIComponent(index)}`);
      const result = await response.json();
      if (result.error) throw new Error(result.error);
      showPrediction(result);
    }

    async function predictFromInput() {
      try {
        await predictIndex(document.getElementById("sampleIndex").value);
      } catch (error) {
        alert(error.message);
      }
    }

    async function predictRandom() {
      try {
        const response = await fetch("/api/predict");
        const result = await response.json();
        if (result.error) throw new Error(result.error);
        document.getElementById("sampleIndex").value = result.index;
        showPrediction(result);
      } catch (error) {
        alert(error.message);
      }
    }

    function showPrediction(result) {
      document.getElementById("prediction").textContent = result.prediction_name;
      document.getElementById("details").textContent =
        `Sample ${result.index} | Ground truth: ${result.truth_name} | Confidence: ${pct(result.confidence)}`;
      const badge = document.getElementById("correctBadge");
      badge.textContent = result.correct ? "Correct" : "Different from label";
      badge.className = result.correct ? "badge" : "badge bad";
      buildBars(result.probabilities);
    }

    function renderFerStatus(fer) {
      const resultBox = document.getElementById("faceResult");
      if (!fer.available) {
        resultBox.innerHTML = `${fer.message}`;
        return;
      }
      const metrics = fer.metrics || {};
      const test = metrics.test ? ` | FER test accuracy: ${pct(metrics.test.accuracy)}` : "";
      resultBox.innerHTML = `
        FER model loaded: <code>${fer.model_path}</code><br>
        Task: ${fer.task}${test}<br>
        Upload a face image to predict with the FER classifier.
      `;
    }

    function renderFerPrediction(prediction) {
      if (!prediction) {
        return "FER prediction unavailable. Train the FER model first.";
      }
      if (!prediction.available) {
        return prediction.message;
      }
      const frameText = prediction.frames_used ? ` from ${prediction.frames_used} sampled frame(s)` : "";
      const bars = prediction.probabilities
        .map(item => `${item.label}: ${pct(item.value)}`)
        .join(" | ");
      return `
        FER prediction${frameText}: <strong>${prediction.prediction_name}</strong>
        (${pct(prediction.confidence)} confidence)<br>
        ${bars}
      `;
    }

    function renderSaveeStatus(savee) {
      const resultBox = document.getElementById("audioResult");
      if (!savee.available) {
        resultBox.innerHTML = `${savee.message}`;
        return;
      }
      const metrics = savee.metrics || {};
      const test = metrics.test ? ` | SAVEE test accuracy: ${pct(metrics.test.accuracy)}` : "";
      const task = metrics.task === "binary" ? "binary audio emotion" : "7-class audio emotion";
      resultBox.innerHTML = `
        SAVEE model loaded: <code>${savee.model_path}</code><br>
        Task: ${task}${test}<br>
        Upload a WAV/audio file to predict with the SAVEE classifier.
      `;
    }

    function renderSaveePrediction(prediction) {
      if (!prediction) {
        return "SAVEE prediction unavailable. Upload/extract an audio track first.";
      }
      if (!prediction.available) {
        return prediction.message;
      }
      const bars = prediction.probabilities
        .map(item => `${item.label}: ${pct(item.value)}`)
        .join(" | ");
      return `
        SAVEE audio prediction: <strong>${prediction.prediction_name}</strong>
        (${pct(prediction.confidence)} confidence)<br>
        ${bars}
      `;
    }

    async function uploadFace() {
      const fileInput = document.getElementById("faceFile");
      const resultBox = document.getElementById("faceResult");
      if (!fileInput.files.length) {
        alert("Choose an image first.");
        return;
      }
      const form = new FormData();
      form.append("video", fileInput.files[0]);
      resultBox.textContent = "Uploading and predicting face emotion...";
      try {
        const response = await fetch("/api/upload-face", { method: "POST", body: form });
        const result = await response.json();
        if (result.error) throw new Error(result.error);
        resultBox.innerHTML = `
          <strong>${result.filename}</strong><br>
          ${renderFerPrediction(result.fer_prediction)}
        `;
      } catch (error) {
        resultBox.textContent = error.message;
      }
    }

    async function uploadAudio() {
      const fileInput = document.getElementById("audioFile");
      const resultBox = document.getElementById("audioResult");
      if (!fileInput.files.length) {
        alert("Choose an audio file first.");
        return;
      }
      const form = new FormData();
      form.append("video", fileInput.files[0]);
      resultBox.textContent = "Uploading and predicting SAVEE audio emotion...";
      try {
        const response = await fetch("/api/upload-audio", { method: "POST", body: form });
        const result = await response.json();
        if (result.error) throw new Error(result.error);
        resultBox.innerHTML = `
          <strong>${result.filename}</strong><br>
          ${renderSaveePrediction(result.savee_prediction)}
        `;
      } catch (error) {
        resultBox.textContent = error.message;
      }
    }

    async function uploadVideo() {
      const fileInput = document.getElementById("videoFile");
      const resultBox = document.getElementById("uploadResult");
      if (!fileInput.files.length) {
        alert("Choose a video first.");
        return;
      }
      const form = new FormData();
      form.append("video", fileInput.files[0]);
      resultBox.textContent = "Uploading and preprocessing video...";
      try {
        const response = await fetch("/api/upload-video", { method: "POST", body: form });
        const result = await response.json();
        if (result.error) throw new Error(result.error);
        resultBox.innerHTML = `
          <strong>${result.filename}</strong><br>
          Frames extracted: ${result.frame_count}<br>
          Duration: ${result.duration_seconds.toFixed(2)} seconds | Source FPS: ${result.source_fps.toFixed(2)}<br>
          Audio WAV: ${result.audio_path ? `<code>${result.audio_path}</code>` : `not extracted (${result.audio_error})`}<br>
          ${renderFerPrediction(result.fer_prediction)}<br>
          ${renderSaveePrediction(result.savee_prediction)}<br>
          ${result.message}
        `;
      } catch (error) {
        resultBox.textContent = error.message;
      }
    }

    loadSummary().catch(error => {
      document.getElementById("status").textContent = "Error";
      alert(error.message);
    });
  </script>
</body>
</html>
"""


def run_smoke_test(state: dict[str, Any]) -> None:
    prediction = predict_sample(state, 0)
    metrics = state["metrics"]
    print(f"Train accuracy: {percent(metrics['train']['accuracy'])}")
    print(f"Validation accuracy: {percent(metrics['validation']['accuracy'])}")
    print(f"Final test accuracy: {percent(metrics['test']['accuracy'])}")
    print(
        "Sample 0 prediction: "
        f"{prediction['prediction_name']} | truth: {prediction['truth_name']}"
    )
    fer = summarize_fer(state.get("fer"))
    print(f"FER model available: {fer['available']}")
    if fer["available"] and fer.get("metrics", {}).get("test"):
        print(f"FER test accuracy: {percent(fer['metrics']['test']['accuracy'])}")
    savee = summarize_savee(state.get("savee"))
    print(f"SAVEE model available: {savee['available']}")
    if savee["available"] and savee.get("metrics", {}).get("test"):
        print(f"SAVEE test accuracy: {percent(savee['metrics']['test']['accuracy'])}")


def main() -> None:
    args = parse_args()
    state = load_artifacts(args.model, args.cache, args.fer_model, args.savee_model)
    if args.smoke_test:
        run_smoke_test(state)
        return

    DemoHandler.state = state
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    host, port = server.server_address
    print(f"Serving MMER demo at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
