import random
import math
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

DATA_FILE = "data.csv"
MODEL_FILE = "model.pkl"

FEATURE_COLS = [
    "total_length", "duration", "avg_speed",
    "bbox_width", "bbox_height",
    "mean_radius", "std_radius", "min_radius", "max_radius",
    "direction_changes", "pen_lifts",
    "start_angle", "end_angle",
    "curv_mean", "curv_std",
    "speed_p25", "speed_p50", "speed_p75"
]

NUM_USERS = 30
SAMPLES_PER_USER = 12


# ---------------------------------------------------------
# Synthetic signature stroke generator (not circles)
# ---------------------------------------------------------
def generate_signature_points(cx, cy, scale=1.0):
    """
    Generates a fake signature-like stroke:
    - Mix of curves and sharp angles
    - Variation per user
    - Mixture of loops and slants
    """
    points = []
    t = random.randint(0, 50)

    # Random number of segments (signature-like scribble)
    segs = random.randint(3, 6)

    x, y = cx, cy

    for s in range(segs):
        # each segment length and curvature style varies
        steps = random.randint(25, 45)

        angle = random.uniform(-2.5, 2.5)
        curv = random.uniform(-0.3, 0.3)
        speed = random.uniform(3, 8)

        for i in range(steps):
            x += math.cos(angle) * speed * scale
            y += math.sin(angle) * speed * scale

            # small curvature drift
            angle += curv * random.uniform(0.5, 1.4)

            # jitter for natural signature noise
            x += random.uniform(-1.5, 1.5)
            y += random.uniform(-1.5, 1.5)

            points.append({"x": x, "y": y, "t": t})
            t += random.randint(8, 18)

        # simulate pen lift between segments
        if random.random() < 0.5:
            t += random.randint(150, 300)

    return points


# ---------------------------------------------------------
# Extract signature features (same as app.py)
# ---------------------------------------------------------
def compute_features(points):
    xs = np.array([p["x"] for p in points])
    ys = np.array([p["y"] for p in points])
    ts = np.array([p["t"] for p in points])

    # distance and angles
    dx = np.diff(xs)
    dy = np.diff(ys)
    d = np.hypot(dx, dy)
    total_length = float(np.sum(d))

    duration = float(ts[-1] - ts[0]) if ts[-1] > ts[0] else 1.0
    speeds = np.divide(d, np.diff(ts) + 1e-9)
    avg_speed = float(np.mean(speeds)) if len(speeds) else 0.0

    # bbox
    bbox_w = float(xs.max() - xs.min())
    bbox_h = float(ys.max() - ys.min())

    # radius from centroid
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    radii = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    mean_r = float(np.mean(radii))
    std_r = float(np.std(radii))
    min_r = float(np.min(radii))
    max_r = float(np.max(radii))

    # angles & direction changes
    angles = np.arctan2(dy, dx)
    direction_changes = 0
    prev = None

    for a in angles:
        if prev is not None:
            diff = abs(a - prev)
            diff = min(diff, 2*math.pi - diff)
            if diff > 0.4:
                direction_changes += 1
        prev = a

    # pen lifts
    pen_lifts = int(np.sum(np.diff(ts) > 150))

    # curvature
    curvs = []
    for i in range(1, len(angles)):
        diff = angles[i] - angles[i-1]
        diff = (diff + math.pi) % (2 * math.pi) - math.pi
        curvs.append(abs(diff))

    curv_mean = float(np.mean(curvs)) if curvs else 0.0
    curv_std = float(np.std(curvs)) if len(curvs) > 1 else 0.0

    # angle endpoints
    start_angle = float(angles[0]) if len(angles) else 0.0
    end_angle = float(angles[-1]) if len(angles) else 0.0

    # speed quartiles
    speed_p25 = float(np.percentile(speeds, 25)) if len(speeds) else 0.0
    speed_p50 = float(np.percentile(speeds, 50)) if len(speeds) else 0.0
    speed_p75 = float(np.percentile(speeds, 75)) if len(speeds) else 0.0

    return {
        "total_length": total_length,
        "duration": duration,
        "avg_speed": avg_speed,
        "bbox_width": bbox_w,
        "bbox_height": bbox_h,
        "mean_radius": mean_r,
        "std_radius": std_r,
        "min_radius": min_r,
        "max_radius": max_r,
        "direction_changes": direction_changes,
        "pen_lifts": pen_lifts,
        "start_angle": start_angle,
        "end_angle": end_angle,
        "curv_mean": curv_mean,
        "curv_std": curv_std,
        "speed_p25": speed_p25,
        "speed_p50": speed_p50,
        "speed_p75": speed_p75
    }


# ---------------------------------------------------------
# Main generator
# ---------------------------------------------------------
def main():
    rows = []
    users = [f"user{i+1}" for i in range(NUM_USERS)]

    for u in users:
        scale = random.uniform(0.8, 1.3)  # unique signature style per user

        for _ in range(SAMPLES_PER_USER):
            pts = generate_signature_points(200, 150, scale)
            feats = compute_features(pts)
            feats["user"] = u
            feats["password"] = "1234"
            rows.append(feats)

    df = pd.DataFrame(rows)
    df.to_csv(DATA_FILE, index=False)
    print(f"Generated {len(df)} samples for {NUM_USERS} users → {DATA_FILE}")

    # Train SVM model
    X = df[FEATURE_COLS]
    y = df["user"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True))
    ])

    model.fit(X, y)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    print("Trained signature model → model.pkl")
    print("Users:", users)


if __name__ == "__main__":
    main()
