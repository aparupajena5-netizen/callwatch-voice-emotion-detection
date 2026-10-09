"""
app.py
------
Flask API that:
  1. Accepts an uploaded/recorded audio clip.
  2. Extracts MFCC-based features (features.py).
  3. Runs them through the trained deep MLP (artifacts/model.joblib).
  4. Returns the predicted emotion, per-class probabilities, and a
     derived frustration/urgency score+band for the caller pipeline.

Run:
    python app.py
Then open frontend/index.html in a browser (it calls this API at
http://localhost:5000).
"""

import os
import io
import tempfile
import time
from collections import deque

import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

from features import extract_features
from emotions import EMOTIONS, score_from_probabilities, band

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

app = Flask(__name__)
CORS(app)  # allow the static frontend (different origin/file://) to call this API

_model = None
_scaler = None
_labels = None

# Keep a short in-memory history per session for a simple "trend" view
# in the UI (not persisted -- resets on server restart).
_history = deque(maxlen=50)


def _load_artifacts():
    global _model, _scaler, _labels
    if _model is None:
        model_path = os.path.join(ARTIFACT_DIR, "model.joblib")
        scaler_path = os.path.join(ARTIFACT_DIR, "scaler.joblib")
        labels_path = os.path.join(ARTIFACT_DIR, "labels.joblib")
        if not all(os.path.exists(p) for p in (model_path, scaler_path, labels_path)):
            raise RuntimeError(
                "No trained model found. Run `python train.py --mode synthetic` "
                "(or with real data) before starting the server."
            )
        _model = joblib.load(model_path)
        _scaler = joblib.load(scaler_path)
        _labels = joblib.load(labels_path)
    return _model, _scaler, _labels


@app.route("/api/health", methods=["GET"])
def health():
    try:
        _load_artifacts()
        return jsonify({"status": "ok", "model_loaded": True})
    except Exception as e:
        return jsonify({"status": "ok", "model_loaded": False, "error": str(e)})


@app.route("/api/predict", methods=["POST"])
def predict():
    model, scaler, labels = _load_artifacts()

    if "audio" not in request.files:
        return jsonify({"error": "No 'audio' file in request"}), 400

    audio_file = request.files["audio"]
    suffix = os.path.splitext(audio_file.filename or "clip.webm")[1] or ".webm"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        t0 = time.time()
        feats = extract_features(tmp_path)
        feats_scaled = scaler.transform(feats.reshape(1, -1))
        proba = model.predict_proba(feats_scaled)[0]
        infer_ms = round((time.time() - t0) * 1000, 1)
    except Exception as e:
        return jsonify({"error": f"Could not process audio: {e}"}), 422
    finally:
        os.unlink(tmp_path)

    prob_by_label = {labels[i]: float(proba[i]) for i in range(len(labels))}
    top_emotion = max(prob_by_label, key=prob_by_label.get)
    frustration, urgency = score_from_probabilities(prob_by_label)

    result = {
        "emotion": top_emotion,
        "confidence": round(prob_by_label[top_emotion], 3),
        "probabilities": {k: round(v, 3) for k, v in prob_by_label.items()},
        "frustration_score": frustration,
        "frustration_band": band(frustration),
        "urgency_score": urgency,
        "urgency_band": band(urgency),
        "inference_ms": infer_ms,
        "timestamp": time.time(),
    }
    _history.append(result)
    return jsonify(result)


@app.route("/api/history", methods=["GET"])
def history():
    return jsonify(list(_history))


if __name__ == "__main__":
    print("Loading model...")
    try:
        _load_artifacts()
        print("Model loaded OK.")
    except Exception as e:
        print(f"WARNING: {e}")
    app.run(host="0.0.0.0", port=5000, debug=True)
