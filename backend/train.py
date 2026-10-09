"""
train.py
--------
Trains a deep multi-layer perceptron on MFCC-based features to
classify emotion from a short voice clip, then saves the model,
feature scaler and label list to backend/artifacts/ for the Flask
API to load.

Usage
-----
# 1) Quick end-to-end smoke test with a synthetic toy dataset:
    python train.py --mode synthetic

# 2) Real training once you have RAVDESS unzipped somewhere:
    python train.py --mode ravdess --data_dir /path/to/RAVDESS

# 3) Real training with your own data sorted into folders
#    (data_dir/angry/*.wav, data_dir/happy/*.wav, ...):
    python train.py --mode folder --data_dir /path/to/data
"""

import os
import argparse
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from dataset import load_folder_dataset, load_ravdess, build_synthetic_demo
from emotions import EMOTIONS

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synthetic", "ravdess", "folder"],
                     default="synthetic")
    ap.add_argument("--data_dir", default=None,
                     help="Required for --mode ravdess / folder")
    ap.add_argument("--epochs", type=int, default=400,
                     help="max_iter for the MLP")
    args = ap.parse_args()

    if args.mode == "synthetic":
        data_dir = os.path.join(os.path.dirname(__file__), "data", "synthetic")
        build_synthetic_demo(data_dir)
        X, y = load_folder_dataset(data_dir)
    elif args.mode == "ravdess":
        assert args.data_dir, "--data_dir is required for --mode ravdess"
        X, y = load_ravdess(args.data_dir)
    else:
        assert args.data_dir, "--data_dir is required for --mode folder"
        X, y = load_folder_dataset(args.data_dir)

    print(f"Loaded {len(X)} samples across {len(set(y))} classes.")
    if len(X) < 20:
        raise SystemExit("Not enough samples to train on. Check --data_dir.")

    le = LabelEncoder()
    le.fit(EMOTIONS)          # fixed label order, even if a class is missing
    y_enc = le.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # "Deep" MLP: 3 hidden layers. This is the model called for in the
    # problem statement (a deep multi-layer perceptron on MFCC features).
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=32,
        learning_rate_init=1e-3,
        max_iter=args.epochs,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.15,
        random_state=42,
        verbose=False,
    )

    print("Training...")
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.3f}\n")
    present_labels = sorted(set(y_test) | set(y_pred))
    print(classification_report(
        y_test, y_pred,
        labels=present_labels,
        target_names=[le.classes_[i] for i in present_labels],
        zero_division=0,
    ))

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(clf, os.path.join(ARTIFACT_DIR, "model.joblib"))
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.joblib"))
    joblib.dump(list(le.classes_), os.path.join(ARTIFACT_DIR, "labels.joblib"))
    print(f"Saved model + scaler + labels to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
