# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import math
import os
import pickle

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

app = Flask(__name__)

DATA_FILE = "data.csv"
MODEL_FILE = "model.pkl"

# Features we store (summary numeric features) - used for model training
FEATURE_COLS = [
    "total_length", "duration", "avg_speed",
    "bbox_width", "bbox_height",
    "mean_radius", "std_radius", "min_radius", "max_radius",
    "direction_changes", "pen_lifts",
    "start_angle", "end_angle",
    "curv_mean", "curv_std",
    "speed_p25", "speed_p50", "speed_p75"
]

# ------------------ Utilities / Feature Extraction ------------------ #
def resample_points(points, n=100):
    """Resample stroke points (list of dicts) by path distance to n points.
    Returns list of (x,y,t)."""
    if not points:
        return []
    # flatten strokes into single sequence preserving order (points already ordered by t)
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    ts = [p["t"] for p in points]
    # compute cumulative distances
    dists = [0.0]
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i-1]
        dy = ys[i] - ys[i-1]
        dists.append(dists[-1] + math.hypot(dx, dy))
    total = dists[-1] if dists[-1] > 0 else 1.0
    target = np.linspace(0, total, n)
    res_x, res_y, res_t = [], [], []
    j = 0
    for tt in target:
        while j < len(dists)-1 and dists[j+1] < tt:
            j += 1
        if j == len(dists)-1:
            res_x.append(xs[-1]); res_y.append(ys[-1]); res_t.append(ts[-1])
            continue
        # linear interp between j and j+1
        frac = (tt - dists[j]) / (dists[j+1] - dists[j] + 1e-9)
        rx = xs[j] + frac * (xs[j+1] - xs[j])
        ry = ys[j] + frac * (ys[j+1] - ys[j])
        rt = ts[j] + frac * (ts[j+1] - ts[j])
        res_x.append(rx); res_y.append(ry); res_t.append(rt)
    return list(zip(res_x, res_y, res_t))


def compute_signature_features(points):
    """Compute numeric features for a signature given a list of points (ordered by time)."""
    if not points or len(points) < 5:
        return None
    # ensure sorted by time
    pts = sorted(points, key=lambda p: p["t"])
    xs = np.array([p["x"] for p in pts], dtype=float)
    ys = np.array([p["y"] for p in pts], dtype=float)
    ts = np.array([p["t"] for p in pts], dtype=float)
    # distances & speeds
    dx = np.diff(xs)
    dy = np.diff(ys)
    d = np.hypot(dx, dy)
    total_length = float(np.sum(d))
    duration = float(ts[-1] - ts[0]) if ts[-1] > ts[0] else 1.0
    speeds = np.divide(d, np.diff(ts) + 1e-9)  # pixels per ms
    avg_speed = float(np.mean(speeds)) if len(speeds) > 0 else 0.0
    # bbox
    bbox_w = float(xs.max() - xs.min())
    bbox_h = float(ys.max() - ys.min())
    # centroid radius stats
    cx = float(np.mean(xs)); cy = float(np.mean(ys))
    radii = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    mean_r = float(np.mean(radii)); std_r = float(np.std(radii, ddof=1)) if len(radii)>1 else 0.0
    min_r = float(np.min(radii)); max_r = float(np.max(radii))
    # direction changes and curvature
    angles = np.arctan2(dy, dx)
    direction_changes = 0
    curvatures = []
    prev_angle = None
    for a_idx, a in enumerate(angles):
        if prev_angle is not None:
            diff = abs(a - prev_angle)
            diff = min(diff, 2*math.pi - diff)
            if diff > 0.4:
                direction_changes += 1
        prev_angle = a
    # curvature approx: angle differences over segment lengths
    if len(angles) >= 2:
        for i in range(1, len(angles)):
            diff = angles[i] - angles[i-1]
            diff = (diff + math.pi) % (2*math.pi) - math.pi
            curvatures.append(abs(diff))
    curv_mean = float(np.mean(curvatures)) if curvatures else 0.0
    curv_std = float(np.std(curvatures, ddof=1)) if len(curvatures)>1 else 0.0
    # pen lifts count: detect large time gaps or zero distances
    time_gaps = np.diff(ts)
    pen_lifts = int(np.sum(time_gaps > 150))  # ms gap >150 considered a pen lift
    # angles start/end
    start_angle = float(angles[0]) if len(angles)>0 else 0.0
    end_angle = float(angles[-1]) if len(angles)>0 else 0.0
    # speed percentiles
    speed_p25 = float(np.percentile(speeds, 25)) if len(speeds)>0 else 0.0
    speed_p50 = float(np.percentile(speeds, 50)) if len(speeds)>0 else 0.0
    speed_p75 = float(np.percentile(speeds, 75)) if len(speeds)>0 else 0.0

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

# ------------------ Data utilities ------------------ #
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

def train_model(df, algo="svm"):
    if df.empty:
        return None
    X = df[FEATURE_COLS].astype(float)
    y = df["user"]
    scaler = StandardScaler()
    if algo == "svm":
        clf = SVC(kernel="rbf", probability=True)
    else:
        clf = RandomForestClassifier(n_estimators=150)
    pipe = Pipeline([("scaler", scaler), ("clf", clf)])
    pipe.fit(X, y)
    # save
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(pipe, f)
    return pipe

def load_model():
    if not os.path.exists(MODEL_FILE):
        return None
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)

# ------------------ API endpoints ------------------ #
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/register")
def register_page():
    return render_template("register.html")

@app.route("/login")
def login_page():
    return render_template("login.html")

@app.route("/register_sample", methods=["POST"])
def register_sample():
    data = request.json
    username = data.get("username","").strip()
    password = data.get("password","")
    points = data.get("points",[])
    if not username or not password:
        return jsonify({"status":"error","message":"username and password required"}),400
    feats = compute_signature_features(points)
    if feats is None:
        return jsonify({"status":"error","message":"insufficient drawing data"}),400
    feats["user"] = username
    feats["password"] = password
    df = load_data()
    df_new = pd.concat([df, pd.DataFrame([feats])], ignore_index=True)
    save_data(df_new)
    count = df_new[df_new["user"]==username].shape[0]
    return jsonify({"status":"saved","user_samples":count})

@app.route("/train", methods=["GET"])
def train_endpoint():
    df = load_data()
    if df.empty:
        return jsonify({"status":"error","message":"no data to train"}),400
    model = train_model(df, algo="svm")
    users = sorted(df["user"].unique().tolist())
    return jsonify({"status":"trained","users":users})

@app.route("/verify", methods=["POST"])
def verify():
    data = request.json
    username = data.get("username","").strip()
    password = data.get("password","")
    points = data.get("points",[])
    if not username or not password:
        return jsonify({"result":"fail","reason":"missing_credentials"}),400
    feats = compute_signature_features(points)
    if feats is None:
        return jsonify({"result":"fail","reason":"no_signature"}),400
    df = load_data()
    if df.empty:
        return jsonify({"result":"fail","reason":"no_registered"}),200
    df_user = df[df["user"]==username]
    if df_user.empty:
        return jsonify({"result":"fail","reason":"unknown_user"}),200
    stored_password = df_user["password"].iloc[0]
    if password != stored_password:
        return jsonify({"result":"fail","reason":"wrong_password"}),200
    model = load_model()
    if model is None:
        model = train_model(df, algo="svm")
        if model is None:
            return jsonify({"result":"fail","reason":"model_not_ready"}),200
    X = pd.DataFrame([feats])
    for c in FEATURE_COLS:
        if c not in X.columns:
            X[c] = 0.0
    X = X[FEATURE_COLS].astype(float)
    pred = model.predict(X)[0]
    prob = None
    try:
        prob = float(max(model.predict_proba(X)[0]))
    except Exception:
        prob = None
    if pred == username:
        # optionally save and retrain to improve
        feats["user"]=username; feats["password"]=stored_password
        df_new = pd.concat([df, pd.DataFrame([feats])], ignore_index=True)
        save_data(df_new)
        train_model(df_new, algo="svm")
        return jsonify({"result":"success","predicted":pred,"confidence":prob}),200
    else:
        return jsonify({"result":"fail","predicted":pred,"confidence":prob}),200

if __name__ == "__main__":
    app.run(debug=True)
