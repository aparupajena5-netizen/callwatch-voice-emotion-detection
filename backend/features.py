"""
features.py
------------
Extracts a fixed-length feature vector from a raw audio file that a
multi-layer perceptron can consume.

Feature set (per clip):
    - 40 MFCCs                -> mean + std over time   (80 values)
    - 40 Delta-MFCCs          -> mean + std over time    (80 values)
    - 40 Delta2-MFCCs         -> mean + std over time    (80 values)
    - Zero crossing rate      -> mean + std               (2 values)
    - Root-mean-square energy -> mean + std               (2 values)
    - Spectral centroid       -> mean + std               (2 values)
    - Pitch (f0, via YIN)     -> mean + std               (2 values)
Total: 248 features.

These are the standard hand-crafted features used in speech-emotion-
recognition literature (RAVDESS / CREMA-D / TESS baselines), which is
why a plain MLP on top of them works well without needing a CNN/RNN.
"""

import numpy as np
import librosa

SAMPLE_RATE = 22050
N_MFCC = 40
FEATURE_DIM = 248


def _stat_pair(x):
    """Return [mean, std] for each row of a 2D array, flattened."""
    return np.concatenate([x.mean(axis=1), x.std(axis=1)])


def extract_features(file_path_or_buffer, sr=SAMPLE_RATE):
    """
    Load an audio file (path, or file-like object) and return a
    248-dim numpy feature vector. Works for wav/mp3/ogg/webm/m4a
    (anything librosa/soundfile can decode).
    """
    y, sr = librosa.load(file_path_or_buffer, sr=sr, mono=True)

    # Guard against silence / near-empty clips
    if y is None or len(y) < sr // 10:
        y = np.pad(y if y is not None else np.array([]), (0, sr // 10))

    # Trim leading/trailing silence so padding doesn't dilute features
    y, _ = librosa.effects.trim(y, top_db=25)
    if len(y) < sr // 10:
        y = np.pad(y, (0, sr // 10 - len(y)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)

    zcr = librosa.feature.zero_crossing_rate(y)
    rmse = librosa.feature.rms(y=y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'),
                      fmax=librosa.note_to_hz('C7'), sr=sr)
    f0 = f0[np.isfinite(f0)]
    if f0.size == 0:
        f0 = np.array([0.0, 0.0])

    feats = np.concatenate([
        _stat_pair(mfcc),
        _stat_pair(delta1),
        _stat_pair(delta2),
        _stat_pair(zcr),
        _stat_pair(rmse),
        _stat_pair(centroid),
        [f0.mean(), f0.std()],
    ]).astype(np.float32)

    assert feats.shape[0] == FEATURE_DIM, f"got {feats.shape[0]}"
    return feats
