"""
emotions.py
-----------
Canonical emotion label set + a business-logic mapping from raw
emotion -> (frustration, urgency) scores in [0, 1], which is the
part your problem statement actually needs for a call-center pipeline
(the emotion label alone isn't actionable; frustration/urgency is).
"""

EMOTIONS = [
    "neutral", "calm", "happy", "sad",
    "angry", "fearful", "disgust", "surprised",
]

# Hand-set priors (0=low, 1=high). These are combined with the model's
# per-class probabilities at inference time, so the score isn't just a
# lookup table -- it reflects prediction confidence too.
FRUSTRATION_MAP = {
    "neutral": 0.10, "calm": 0.05, "happy": 0.05, "sad": 0.35,
    "angry": 0.95, "fearful": 0.55, "disgust": 0.70, "surprised": 0.30,
}
URGENCY_MAP = {
    "neutral": 0.15, "calm": 0.05, "happy": 0.10, "sad": 0.30,
    "angry": 0.85, "fearful": 0.80, "disgust": 0.40, "surprised": 0.60,
}


def score_from_probabilities(prob_by_label: dict):
    """
    Turn a {emotion: probability} dict (summing to ~1) into a single
    weighted frustration score and urgency score in [0, 1].
    """
    frustration = sum(p * FRUSTRATION_MAP[e] for e, p in prob_by_label.items())
    urgency = sum(p * URGENCY_MAP[e] for e, p in prob_by_label.items())
    return round(float(frustration), 3), round(float(urgency), 3)


def band(score: float) -> str:
    if score >= 0.66:
        return "HIGH"
    if score >= 0.33:
        return "MEDIUM"
    return "LOW"
