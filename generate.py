import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# --- CONFIG --- #
PASSWORD = "MyNameIsRochak!"
USERS = ["rochak", "rajat", "rohan"]
MAX_KEYS = 20          # must match MAX_KEYS in app.py
SAMPLES_PER_USER = 15  # 15 samples per user (5 clean, 5 medium drift, 5 high drift)

DATA_FILE = "data.csv"
MODEL_FILE = "model.pkl"


def simulate_user_dwell_pattern(user_name, base_mean_range=(120, 220)):
    """
    Create a base dwell-time pattern for a user:
    - base mean dwell
    - per-position bias to make each user unique
    """
    base_mean = random.uniform(*base_mean_range)  # ms
    # slight per-character biases for this user
    position_bias = [random.uniform(-20, 20) for _ in range(len(PASSWORD))]
    return base_mean, position_bias


def synthesize_sample(base_mean, position_bias, noise_sigma, drift=0.0):
    """
    Generate one sample (one password typing attempt) for a user.
    - base_mean: base dwell
    - position_bias: list per position
    - noise_sigma: randomness
    - drift: systematic shift (fatigue, session change)
    Returns a list of dwell times (len <= MAX_KEYS).
    """
    dwell = []
    for i, ch in enumerate(PASSWORD):
        mean_here = base_mean + position_bias[i] + drift
        # ensure positive
        mean_here = max(40, mean_here)
        value = random.gauss(mean_here, noise_sigma)
        value = max(20, value)  # at least 20ms
        dwell.append(value)

    # pad/truncate to MAX_KEYS (same as app.py logic)
    if len(dwell) < MAX_KEYS:
        dwell = dwell + [0.0] * (MAX_KEYS - len(dwell))
    else:
        dwell = dwell[:MAX_KEYS]

    return dwell


def compute_features(dwell):
    """Same idea as app.py: dwell_0..dwell_19 + mean_dwell + std_dwell."""
    dwell = [float(x) for x in dwell]
    mean_dwell = sum(dwell) / len(dwell)
    if len(dwell) > 1:
        var = sum((x - mean_dwell) ** 2 for x in dwell) / (len(dwell) - 1)
        std_dwell = var ** 0.5
    else:
        std_dwell = 0.0

    feats = {}
    for i in range(MAX_KEYS):
        feats[f"dwell_{i}"] = dwell[i]
    feats["mean_dwell"] = mean_dwell
    feats["std_dwell"] = std_dwell
    return feats


def main():
    rows = []

    for user in USERS:
        base_mean, position_bias = simulate_user_dwell_pattern(user)

        # 5 "clean" samples (low noise, no drift)
        for _ in range(5):
            dwell = synthesize_sample(base_mean, position_bias,
                                      noise_sigma=10, drift=0)
            feats = compute_features(dwell)
            feats["user"] = user
            feats["password"] = PASSWORD
            rows.append(feats)

        # 5 "medium drift" samples (more variance)
        for _ in range(5):
            dwell = synthesize_sample(base_mean, position_bias,
                                      noise_sigma=20, drift=random.uniform(-10, 10))
            feats = compute_features(dwell)
            feats["user"] = user
            feats["password"] = PASSWORD
            rows.append(feats)

        # 5 "high drift" samples (simulate tiredness / different mood)
        for _ in range(5):
            dwell = synthesize_sample(base_mean, position_bias,
                                      noise_sigma=30, drift=random.uniform(-25, 25))
            feats = compute_features(dwell)
            feats["user"] = user
            feats["password"] = PASSWORD
            rows.append(feats)

    df = pd.DataFrame(rows)
    df.to_csv(DATA_FILE, index=False)
    print(f"Saved synthetic dataset to {DATA_FILE} with shape {df.shape}")

    # Train model just like app.py does
    feature_cols = [col for col in df.columns if col.startswith("dwell_")] + ["mean_dwell", "std_dwell"]
    X = df[feature_cols]
    y = df["user"]

    model = RandomForestClassifier()
    model.fit(X, y)
    model.feature_names_in_ = feature_cols

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    print(f"Trained RandomForest model and saved to {MODEL_FILE}")
    print("Users in dataset:", df["user"].unique())


if __name__ == "__main__":
    main()
