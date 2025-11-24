import random
import math
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

DATA_FILE = "data.csv"
MODEL_FILE = "model.pkl"

FEATURE_COLS = [
    "total_length",
    "duration",
    "avg_speed",
    "bbox_width",
    "bbox_height",
    "mean_radius",
    "std_radius",
    "min_radius",
    "max_radius",
    "direction_changes",
]

NUM_USERS = 30
SAMPLES_PER_USER = 12  # per user


# ------------ Generate synthetic circle-drawing points ------------ #

def generate_circle_points(cx, cy, radius, variation=0.10, noise=2.0):
    """Generates a noisy circle-like stroke."""
    points = []
    t_start = random.randint(0, 100)
    t = t_start

    for angle in range(0, 360, 5):
        rad = math.radians(angle)
        r = radius * (1 + random.uniform(-variation, variation))

        x = cx + r * math.cos(rad) + random.uniform(-noise, noise)
        y = cy + r * math.sin(rad) + random.uniform(-noise, noise)

        points.append({"x": x, "y": y, "t": t})
        t += random.randint(10, 20)

    return points


# ------------ Feature extraction (same as app.py) ------------ #

def compute_features(points):
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    ts = [p["t"] for p in points]

    total_length = 0
    direction_changes = 0
    prev_angle = None

    for i in range(1, len(points)):
        dx = xs[i] - xs[i-1]
        dy = ys[i] - ys[i-1]
        dist = math.sqrt(dx*dx + dy*dy)
        total_length += dist

        angle = math.atan2(dy, dx)
        if prev_angle is not None:
            diff = abs(angle - prev_angle)
            diff = min(diff, 2*math.pi - diff)
            if diff > 0.5:
                direction_changes += 1
        prev_angle = angle

    duration = ts[-1] - ts[0]
    if duration <= 0:
        duration = 1
    avg_speed = total_length / duration

    bbox_width = max(xs) - min(xs)
    bbox_height = max(ys) - min(ys)

    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    radii = [math.sqrt((x - cx)**2 + (y - cy)**2) for x, y in zip(xs, ys)]

    mean_radius = sum(radii) / len(radii)
    if len(radii) > 1:
        var = sum((r - mean_radius)**2 for r in radii) / (len(radii) - 1)
        std_radius = math.sqrt(var)
    else:
        std_radius = 0

    min_radius = min(radii)
    max_radius = max(radii)

    return {
        "total_length": total_length,
        "duration": duration,
        "avg_speed": avg_speed,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "mean_radius": mean_radius,
        "std_radius": std_radius,
        "min_radius": min_radius,
        "max_radius": max_radius,
        "direction_changes": direction_changes,
    }


# ------------ Main generator ------------ #

def main():
    rows = []
    users = [f"user{i+1}" for i in range(NUM_USERS)]

    for user in users:
        # Each user gets slightly different base radius & style
        base_radius = random.uniform(60, 120)
        variation = random.uniform(0.05, 0.20)
        noise = random.uniform(1.0, 3.0)

        for _ in range(SAMPLES_PER_USER):
            pts = generate_circle_points(200, 150, base_radius, variation, noise)
            feats = compute_features(pts)
            feats["user"] = user
            feats["password"] = "1234"  # same pwd for dummy dataset
            rows.append(feats)

    df = pd.DataFrame(rows)
    df.to_csv(DATA_FILE, index=False)
    print(f"Saved {len(df)} samples for {NUM_USERS} users → {DATA_FILE}")

    # Train model
    X = df[FEATURE_COLS]
    y = df["user"]
    model = RandomForestClassifier()
    model.fit(X, y)
    model.feature_names_in_ = FEATURE_COLS

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    print("Model trained → model.pkl")
    print("Users:", users)


if __name__ == "__main__":
    main()
