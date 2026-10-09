# Callwatch — Real-Time Emotion Detection in Voice

Deep Learning project: trains a **deep multi-layer perceptron (MLP)** on
**MFCC-based audio features** to classify a caller's emotion, then converts
that into **frustration** and **urgency** scores for a call-center pipeline
— with a Flask backend and a browser frontend to test it live from your
microphone or an uploaded audio file.

Matches the assigned problem statement: *"Train a deep multi-layer
perceptron on MFCC audio features to accurately gauge caller frustration
and urgency in automated pipelines."*

## How it works

```
voice clip → librosa (MFCC + delta + delta2 + pitch + energy, 248 dims)
           → StandardScaler
           → Deep MLP (256 → 128 → 64 → 8 emotions)
           → softmax probabilities
           → weighted frustration/urgency score (emotions.py)
```

8 emotion classes: `neutral, calm, happy, sad, angry, fearful, disgust, surprised`
(the standard RAVDESS/CREMA-D label set).

## Project structure

```
backend/
  features.py      # audio -> 248-dim feature vector (MFCC/delta/pitch/energy)
  emotions.py       # label list + frustration/urgency scoring logic
  dataset.py        # loads RAVDESS / a folder-per-emotion dataset / synthetic demo data
  train.py           # trains the MLP, saves model+scaler+labels to backend/artifacts/
  app.py             # Flask API: /api/health, /api/predict, /api/history
  requirements.txt
frontend/
  index.html         # single-file web UI: record mic / upload file, live results
```

## 1. Setup

```bash
cd backend
python3 -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

## 2. Train a model

You have three options. **Start with option A** to confirm the whole
pipeline works, then swap in real data with option B for a model that
actually recognizes emotion well.

**A — Quick smoke test (synthetic data, no download needed)**
Generates small synthetic audio clips with distinct pitch/energy/noise
per class purely so you can verify every part of the stack runs. It will
train to high accuracy on itself but **will not generalize to real
voices** — it's a plumbing test, not a real model.
```bash
python train.py --mode synthetic
```

**B — Real training on RAVDESS (recommended for your actual submission)**
1. Download the RAVDESS Speech dataset (Audio_Speech_Actors_01-24.zip) from
   Zenodo: https://zenodo.org/record/1188976 — free, ~24k files, no login.
2. Unzip it anywhere, e.g. `~/data/RAVDESS/Actor_01/...`.
3. Train:
   ```bash
   python train.py --mode ravdess --data_dir ~/data/RAVDESS
   ```
   This parses RAVDESS's filename convention automatically (the 3rd
   number in the filename is the emotion code).

**C — Your own dataset**
Sort audio files into folders named exactly after the 8 emotion labels:
```
mydata/
  angry/*.wav
  happy/*.wav
  sad/*.wav
  ...
```
```bash
python train.py --mode folder --data_dir mydata
```

Training prints a held-out accuracy and a per-class precision/recall/F1
report, and saves `backend/artifacts/{model,scaler,labels}.joblib`.

Other datasets that work well here: **CREMA-D**, **TESS**, **SAVEE**,
**IEMOCAP** — reshape any of them into the folder-per-emotion layout
above and use `--mode folder`.

## 3. Run the backend

```bash
cd backend
python app.py
```
Starts a Flask API on `http://localhost:5000`.
- `GET  /api/health`  → is the model loaded?
- `POST /api/predict` → multipart form, field `audio` = a wav/webm/mp3 clip.
  Returns emotion, per-class probabilities, frustration/urgency scores+bands.
- `GET  /api/history`  → last 50 predictions this session (in-memory demo).

## 4. Open the frontend

Just open `frontend/index.html` directly in a browser (Chrome/Edge/Firefox)
while `app.py` is running. It will:
- show a green dot once it can reach the backend,
- let you record ~2–4 seconds from your mic (with a live waveform) or
  upload an audio file,
- show the predicted emotion, a probability bar per class, and the
  frustration/urgency gauges with LOW/MEDIUM/HIGH bands.

No build step, no npm — it's a single static HTML file.


