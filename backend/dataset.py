"""
dataset.py
----------
Three ways to get training data into the pipeline:

1. load_folder_dataset(root)   -> root/<emotion_name>/*.wav  (simplest,
   works with any dataset once you sort files into folders by label)

2. load_ravdess(root)          -> parses the official RAVDESS filename
   code so you can point straight at an unzipped RAVDESS download,
   e.g. Actor_01/03-01-05-01-01-01-01.wav  (3rd number = emotion)

3. build_synthetic_demo(out_dir) -> synthesizes a small toy dataset
   (varying pitch/energy/noise per "emotion") so the WHOLE pipeline
   (train -> save model -> backend -> frontend) can be run and
   sanity-checked immediately, without a real dataset. It will NOT
   give you a model that recognizes real human emotion -- swap in
   RAVDESS/CREMA-D/TESS/IEMOCAP for that (see README).
"""

import os
import glob
import numpy as np
import soundfile as sf

from emotions import EMOTIONS
from features import extract_features, SAMPLE_RATE

RAVDESS_CODE_TO_EMOTION = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}

AUDIO_EXTS = (".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm")


def load_folder_dataset(root):
    X, y = [], []
    for emotion in EMOTIONS:
        folder = os.path.join(root, emotion)
        if not os.path.isdir(folder):
            continue
        files = [f for f in glob.glob(os.path.join(folder, "*"))
                  if f.lower().endswith(AUDIO_EXTS)]
        for f in files:
            try:
                X.append(extract_features(f))
                y.append(emotion)
            except Exception as e:
                print(f"  skip {f}: {e}")
    return np.array(X), np.array(y)


def load_ravdess(root):
    X, y = [], []
    files = glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)
    for f in files:
        name = os.path.basename(f)
        parts = name.split("-")
        if len(parts) < 3:
            continue
        code = parts[2]
        emotion = RAVDESS_CODE_TO_EMOTION.get(code)
        if emotion is None:
            continue
        try:
            X.append(extract_features(f))
            y.append(emotion)
        except Exception as e:
            print(f"  skip {f}: {e}")
    return np.array(X), np.array(y)


def build_synthetic_demo(out_dir, n_per_class=25, seconds=2.0, sr=SAMPLE_RATE):
    """
    Writes synthetic wav files to out_dir/<emotion>/*.wav so the full
    pipeline is runnable end-to-end without any external dataset.
    Each "emotion" gets a distinct pitch range / amplitude / noise
    profile so the classifier has *something* separable to learn on.
    """
    rng = np.random.default_rng(42)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)

    # (base_freq_range, amplitude_range, noise_level, tremolo_rate)
    profile = {
        "neutral":   ((150, 180), (0.15, 0.25), 0.01, 0),
        "calm":      ((120, 150), (0.10, 0.18), 0.005, 0),
        "happy":     ((220, 280), (0.30, 0.45), 0.02, 6),
        "sad":       ((100, 130), (0.08, 0.15), 0.01, 1),
        "angry":     ((250, 320), (0.55, 0.80), 0.06, 9),
        "fearful":   ((280, 340), (0.25, 0.40), 0.08, 12),
        "disgust":   ((160, 200), (0.20, 0.35), 0.05, 3),
        "surprised": ((300, 380), (0.40, 0.60), 0.03, 4),
    }

    for emotion in EMOTIONS:
        folder = os.path.join(out_dir, emotion)
        os.makedirs(folder, exist_ok=True)
        (f_lo, f_hi), (a_lo, a_hi), noise_lvl, trem = profile[emotion]
        for i in range(n_per_class):
            f0 = rng.uniform(f_lo, f_hi)
            amp = rng.uniform(a_lo, a_hi)
            harmonics = sum(
                (1.0 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.28))
                for k in (1, 2, 3)
            )
            if trem > 0:
                harmonics *= (1 + 0.3 * np.sin(2 * np.pi * trem * t))
            noise = rng.normal(0, noise_lvl, size=t.shape)
            wave = amp * harmonics / np.max(np.abs(harmonics) + 1e-8) + noise
            wave = np.clip(wave, -1, 1).astype(np.float32)
            sf.write(os.path.join(folder, f"{emotion}_{i:03d}.wav"), wave, sr)

    print(f"Synthetic demo dataset written to: {out_dir}")
    return out_dir
