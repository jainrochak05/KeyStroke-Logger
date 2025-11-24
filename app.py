from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import math

app = Flask(__name__)

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


# ----------------- Feature extraction from raw points ----------------- #

def compute_features_from_points(points):
    """
    points: list of dicts: [{x:..., y:..., t:...}, ...]
    Returns dict of features or None if not enough points.
    """
    if not points or len(points) < 5:
        return None

    # sort by time
    pts = sorted(points, key=lambda p: p["t"])
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    ts = [p["t"] for p in pts]

    # total path length
    total_length = 0.0
    direction_changes = 0
    prev_angle = None

    for i in range(1, len(pts)):
        dx = xs[i] - xs[i - 1]
        dy = ys[i] - ys[i - 1]
        dist = math.sqrt(dx * dx + dy * dy)
        total_length += dist

        angle = math.atan2(dy, dx)
        if prev_angle is not None:
            diff = abs(angle - prev_angle)
            # normalize angle diff to [0, pi]
            diff = min(diff, 2 * math.pi - diff)
            if diff > 0.5:  # significant direction change
                direction_changes += 1
        prev_angle = angle

    duration = ts[-1] - ts[0]  # ms
    if duration <= 0:
        duration = 1.0
    avg_speed = total_length / duration

    # bounding box
    bbox_width = max(xs) - min(xs)
    bbox_height = max(ys) - min(ys)

    # radius around centroid (for circle-ness and personal signature)
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    radii = [math.sqrt((x - cx) ** 2 + (y - cy) ** 2) for x, y in zip(xs, ys)]

    mean_radius = sum(radii) / len(radii)
    if len(radii) > 1:
        var = sum((r - mean_radius) ** 2 for r in radii) / (len(radii) - 1)
        std_radius = math.sqrt(var)
    else:
        std_radius = 0.0

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


# ----------------- Data utilities ----------------- #

def empty_df():
    return pd.DataFrame(columns=FEATURE_COLS + ["user", "password"])


def load_data():
    if os.path.exists(DATA_FILE):
        try:
            return pd.read_csv(DATA_FILE)
        except pd.errors.EmptyDataError:
            return empty_df()
    return empty_df()


def save_data(df):
    df.to_csv(DATA_FILE, index=False)


def train_model(df):
    if df.empty:
        return None
    X = df[FEATURE_COLS]
    y = df["user"]
    model = RandomForestClassifier()
    model.fit(X, y)
    model.feature_names_in_ = FEATURE_COLS
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    return model


def load_model():
    if not os.path.exists(MODEL_FILE):
        return None
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


# ----------------- API: Register samples ----------------- #

@app.route("/register_sample", methods=["POST"])
def register_sample():
    """
    Body JSON:
    {
      "username": "...",
      "password": "...",
      "points": [{x:..., y:..., t:...}, ...]
    }
    """
    data = request.json
    username = data.get("username", "").strip()
    password = data.get("password", "")
    points = data.get("points", [])

    if not username or not password:
        return jsonify({"status": "error", "message": "username and password required"}), 400

    feats = compute_features_from_points(points)
    if feats is None:
        return jsonify({"status": "error", "message": "not enough drawing data"}), 400

    df = load_data()
    feats["user"] = username
    feats["password"] = password

    df_new = pd.concat([df, pd.DataFrame([feats])], ignore_index=True)
    save_data(df_new)

    # Count samples for that user
    count = df_new[df_new["user"] == username].shape[0]

    return jsonify({"status": "saved", "user_samples": count})


# ----------------- API: Train model ----------------- #

@app.route("/train", methods=["GET"])
def train_endpoint():
    df = load_data()
    if df.empty:
        return jsonify({"status": "error", "message": "no data to train"}), 400
    model = train_model(df)
    if model is None:
        return jsonify({"status": "error", "message": "training failed"}), 500
    users = sorted(df["user"].unique().tolist())
    return jsonify({"status": "trained", "users": users})


# ----------------- API: Verify login ----------------- #

@app.route("/verify", methods=["POST"])
def verify():
    """
    Body JSON:
    {
      "username": "...",
      "password": "...",
      "points": [{x:..., y:..., t:...}, ...]
    }
    """
    data = request.json
    username = data.get("username", "").strip()
    password = data.get("password", "")
    points = data.get("points", [])

    if not username or not password:
        return jsonify({"result": "fail", "reason": "missing_credentials"}), 400

    feats = compute_features_from_points(points)
    if feats is None:
        return jsonify({"result": "fail", "reason": "no_drawing_data"}), 400

    df = load_data()
    if df.empty:
        return jsonify({"result": "fail", "reason": "no_registered_users"}), 200

    df_user = df[df["user"] == username]
    if df_user.empty:
        return jsonify({"result": "fail", "reason": "unknown_user"}), 200

    stored_password = df_user["password"].iloc[0]
    if password != stored_password:
        return jsonify({"result": "fail", "reason": "wrong_password"}), 200

    model = load_model()
    if model is None:
        model = train_model(df)
        if model is None:
            return jsonify({"result": "fail", "reason": "model_not_ready"}), 200

    X = pd.DataFrame([feats])
    # ensure columns align
    for col in FEATURE_COLS:
        if col not in X.columns:
            X[col] = 0.0
    X = X[FEATURE_COLS]

    predicted_user = model.predict(X)[0]

    if predicted_user == username:
        # optional: enhance dataset with this new successful sample
        feats["user"] = username
        feats["password"] = stored_password
        df_new = pd.concat([df, pd.DataFrame([feats])], ignore_index=True)
        save_data(df_new)
        train_model(df_new)
        return jsonify({"result": "success", "predicted": predicted_user}), 200
    else:
        return jsonify({"result": "fail", "predicted": predicted_user}), 200


# ----------------- Pages ----------------- #

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/register")
def register_page():
    return render_template("register.html")


@app.route("/login")
def login_page():
    return render_template("login.html")


if __name__ == "__main__":
    app.run(debug=True)
