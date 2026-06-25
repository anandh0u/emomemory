# Multimodal Emotion Recognition

This project is a modular multi-agent scaffold for video-based emotion and
sentiment prediction using facial, speech, text, and auxiliary audio-event
signals.

## First Milestone

Implemented now:

- CMU-MOSEI `.csd` loading with `h5py`
- 5-class sentiment discretization
- explicit, official `mmsdk`, or deterministic fallback splits
- video frame extraction at 1 FPS by default
- mono WAV audio extraction at 16 kHz
- package stubs for agents, fusion, training, inference, and evaluation

## Install

For the current data-loading and video-processing milestone, install the base
requirements. If your network is slow, start with the small core file:

```powershell
.\.venv\Scripts\python.exe -m pip install --timeout 120 --retries 10 -r requirements-core.txt
```

Install the larger model stack when the connection is steadier:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The SER agent will later need `funasr`. It is kept in
`requirements-agents.txt` because on Windows/Python 3.13 its `editdistance`
dependency may try to compile native code and fail without Microsoft C++ Build
Tools:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-agents.txt
```

If that optional install fails on `editdistance`, use Python 3.11/3.12 for the
venv or install Microsoft C++ Build Tools.

## CMU-MOSEI Data

The dataset files are not stored directly in Git. The official CMU Multimodal
SDK has been cloned into `third_party/CMU-MultimodalSDK`. Install it with:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -e .\third_party\CMU-MultimodalSDK
```

Download the MOSEI CSD files needed by this project:

```powershell
.\.venv\Scripts\python.exe scripts\download_mosei_csd.py --output-dir datasets\mosei_csd
```

If the official CMU server times out, use the mirror/resume downloader:

```powershell
.\.venv\Scripts\python.exe scripts\download_mosei_csd.py --output-dir datasets\mosei_csd --source huggingface --timeout 120 --retries 10
```

Then verify loading:

```powershell
.\.venv\Scripts\python.exe train.py --data-dir datasets\mosei_csd --feature-mode aligned
```

`aligned` is the default and keeps memory usage reasonable by mean-pooling each
modality inside the matching label segment interval.

Before the large acoustic and visual CSD files are downloaded, verify the
completed text vectors and labels only:

```powershell
.\.venv\Scripts\python.exe train.py --data-dir datasets\mosei_csd --modalities text
```

## Train Best Current Baseline

Build the aligned native-concat feature cache once:

```powershell
.\.venv\Scripts\python.exe train.py --data-dir datasets\mosei_csd --feature-mode aligned --feature-layout native --build-cache-only --cache-path artifacts\mosei_aligned_native_concat_features.joblib
```

Train the strongest currently measured 5-class baseline:

```powershell
.\.venv\Scripts\python.exe train.py --data-dir datasets\mosei_csd --train-model --classifier catboost --feature-mode aligned --feature-layout native --cache-path artifacts\mosei_aligned_native_concat_features.joblib --model-out artifacts\mosei_aligned_native_concat_catboost_depth8.joblib --iterations 1500 --depth 8 --learning-rate 0.03 --early-stopping-rounds 150
```

This command has reached 47.53% test accuracy on the official 5-class
CMU-MOSEI split in this workspace. That is the honest benchmark path; a much
higher percentage would require changing the task, for example to binary
sentiment, or leaking labels into the inputs.

For comparison, the default `--classifier auto` tries several CatBoost, MLP,
and logistic-regression candidates and keeps the one with the best validation
accuracy.

## Demo Window

Open the local demo window after training:

```powershell
.\.venv\Scripts\python.exe demo_window.py
```

The demo shows the notebook-style training accuracy and the real final test
accuracy, then lets you run predictions on held-out MOSEI test samples from the
saved feature cache.

## Browser Demo

Start the local chat-style browser demo:

```powershell
.\.venv\Scripts\python.exe web_demo.py --host 127.0.0.1 --port 7860
```

Then open:

```text
http://127.0.0.1:7860
```

The browser demo is a ChatGPT-style multimodal window. You can type text or
attach an image, audio file, or video. Text is routed to the local text-tone
analyzer, images are routed to the FER face-emotion model, audio is routed to
the SAVEE audio model, and videos are routed through frame extraction plus
audio extraction. The interface shows answers only; accuracy is kept in the
separate report.

Full metric table:

```text
reports/final_accuracy_report.md
```

Important accuracy note:

- Notebook-style training accuracy: 90.49%
- Official 5-class validation accuracy: 46.23%
- Official 5-class test accuracy: 47.53%

The current trained classifier predicts from CMU-MOSEI OpenFace/COVAREP/GloVe
feature vectors. A raw uploaded `.mp4` cannot honestly be scored by that same
model until matching raw-video feature extractors are implemented. The upload
panel is included to demonstrate the video ingestion/preprocessing stage of
the full pipeline.

## Higher-Accuracy MOSEI Binary Benchmarks

The original MOSEI result is a harder 5-class task. For a higher honest
sentiment score, train binary negative-vs-positive models from the same saved
feature cache:

```powershell
.\.venv\Scripts\python.exe train_mosei_binary.py --threshold 0.3 --model-out artifacts\mosei_binary_nonneutral_logreg.joblib
.\.venv\Scripts\python.exe train_mosei_binary.py --threshold 1.0 --model-out artifacts\mosei_binary_strong_logreg.joblib
.\.venv\Scripts\python.exe train_mosei_binary.py --threshold 2.5 --model-out artifacts\mosei_binary_extreme_logreg.joblib
```

Current binary MOSEI results:

```text
Non-neutral, abs(score) > 0.3: 80.36% test accuracy on 3,636 test samples
Strong, abs(score) > 1.0:      87.75% test accuracy on 1,420 test samples
Extreme, abs(score) > 2.5:     90.09% test accuracy on 111 test samples
```

The 90.09% score is honest, but it applies only to the narrow extreme-sentiment
subset. Use the non-neutral result when you need a broader binary benchmark.

## Animated Dataset Track

The animated dataset at `E:\emotion_recognition_internship\data\raw\animated`
has been added as a binary optimized/not-optimized track using the saved
multimodal embeddings at:

```text
E:\emotion_recognition_internship\features\animated_embeddings.pt
```

Train the selected animated model:

```powershell
.\.venv\Scripts\python.exe train_animated.py --classifier logreg --model-out artifacts\animated_classifier.joblib
```

The selected animated model reaches 99.89% training accuracy, 51.85%
validation accuracy, and 49.21% test accuracy. This means it memorizes the
training split but does not honestly reach 90%+ on held-out validation/test.

## FER Face-Emotion Track

This project now also supports the saved FER2013 dataset at:

```text
E:\emotion_recognition_internship\data\raw\fer2013.csv
```

Train the stronger CPU-friendly binary FER model:

```powershell
.\.venv\Scripts\python.exe train_fer.py --task binary --classifier mlp --feature-size 24 --max-iter 40 --model-out artifacts\fer_binary_mlp_classifier.joblib
```

Previous FER binary MLP result:

```text
Training accuracy:   96.86%
Validation accuracy: 77.57%
Test accuracy:       78.13%
```

Train the stronger CNN FER model now used by the browser demo:

```powershell
.\.venv\Scripts\python.exe -u train_fer_cnn.py --task binary --epochs 25 --batch-size 512 --patience 6 --model-out artifacts\fer_binary_cnn.pt
```

Current FER binary CNN result:

```text
Training accuracy:   91.93%
Validation accuracy: 87.83%
Test accuracy:       86.50%
```

This is the higher-accuracy demo track. It is a different task from 5-class
CMU-MOSEI: FER binary classifies face images as `negative` or `positive`, with
neutral FER samples removed.

The 7-class FER baseline is also available:

```powershell
.\.venv\Scripts\python.exe train_fer.py --task fer7 --classifier sgd --feature-size 24 --max-iter 100 --model-out artifacts\fer_classifier.joblib
```

Current 7-class FER test accuracy is 33.02% with the fast linear baseline.

## SAVEE Audio-Emotion Track

The browser demo now also uses the saved SAVEE manifest:

```text
E:\emotion_recognition_data\agents\multimodal\manifests\labels_savee_fer_paired.csv
```

and WAV files under:

```text
E:\emotion_recognition_internship\data\raw\ALL
```

Train the default SAVEE binary audio model:

```powershell
.\.venv\Scripts\python.exe train_savee.py --task binary --classifier rf --model-out artifacts\savee_binary_rf_classifier.joblib
```

Current SAVEE binary result:

```text
Training accuracy:   100.00%
Validation accuracy: 74.07%
Test accuracy:       66.67%
```

The 7-class SAVEE model is also saved:

```powershell
.\.venv\Scripts\python.exe train_savee.py --task savee7 --classifier rf --model-out artifacts\savee_audio_rf_classifier.joblib
```

Current 7-class SAVEE test accuracy is 29.17% with the handcrafted-audio
random forest baseline. The browser demo uses the stronger binary SAVEE model
by default for audio uploads and video audio tracks.

## Layout

```text
agents/
  fed_agent.py
  ser_agent.py
  ted_agent.py
  aed_agent.py
data/
  mosei_loader.py
  video_processor.py
fusion/
  aggregator.py
  adapter.py
  classifier.py
train.py
inference.py
evaluate.py
requirements.txt
```

## Verify MOSEI Loading

```bash
python train.py --data-dir /path/to/mosei_csd
```

If you have split files, provide a directory containing `train.txt`, `val.txt`
or `valid.txt`, and `test.txt`:

```bash
python train.py --data-dir /path/to/mosei_csd --split-dir /path/to/splits
```

## Verify Video Processing

```bash
python inference.py path/to/input.mp4 --output-dir outputs/sample_video
```

The command extracts frames and audio first, then stops until the model agents
are implemented.

## FED Agent

`FEDAgent` expects YOLOv8-Face weights as a local `.pt` file. Put the file at
`models/yolov8n-face.pt` or pass `face_model_path` when creating the agent:

```python
from agents import FEDAgent

agent = FEDAgent(face_model_path="models/yolov8n-face.pt")
embeddings = agent.extract(["outputs/sample_video/frames/frame_000000.jpg"])
print(embeddings.shape)  # (num_faces, 512)
```
