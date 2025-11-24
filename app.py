from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

app = Flask(__name__)

DATA_FILE = "data.csv"
MODEL_FILE = "model.pkl"

# max number of keys we consider per attempt (password length upper bound)
MAX_KEYS = 20

# ----------------- Helpers -----------------

def compute_features(dwell):
    """
    From a list of dwell times (per key), build a fixed-length feature vector:
    - dwell_0 ... dwell_(MAX_KEYS-1) (padded with 0 if shorter)
    - mean_dwell, std_dwell
    """
    if not dwell:
        return None

    dwell = [float(x) for x in dwell]

    # pad / truncate to MAX_KEYS
    if len(dwell) < MAX_KEYS:
        dwell = dwell + [0.0] * (MAX_KEYS - len(dwell))
    else:
        dwell = dwell[:MAX_KEYS]

    mean_dwell = sum(dwell) / len(dwell)

    # std dev
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


def load_data():
    """Safely load dataset; handle empty or missing file."""
    if os.path.exists(DATA_FILE):
        try:
            return pd.read_csv(DATA_FILE)
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=feature_columns() + ["user", "password"])
    else:
        return pd.DataFrame(columns=feature_columns() + ["user", "password"])


def save_data(df):
    df.to_csv(DATA_FILE, index=False)


def feature_columns():
    """List of all feature columns used by the model."""
    cols = [f"dwell_{i}" for i in range(MAX_KEYS)]
    cols += ["mean_dwell", "std_dwell"]
    return cols


def train_global_model(df):
    """Train multi-user model on all samples."""
    if df.empty:
        return None

    X = df[feature_columns()]
    y = df["user"]

    model = RandomForestClassifier()
    model.fit(X, y)

    # store feature names explicitly
    model.feature_names_in_ = list(X.columns)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    return model


def load_model():
    if not os.path.exists(MODEL_FILE):
        return None
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


# ----------------- API: Register user -----------------

@app.route("/register_data", methods=["POST"])
def register_data():
    """
    Body JSON:
    {
      "username": "...",
      "password": "...",
      "samples": [
         [dwell1, dwell2, ...],  # attempt 1
         ...
      ]
    }
    """
    data = request.json
    username = data.get("username", "").strip()
    password = data.get("password", "")
    samples = data.get("samples", [])

    if not username or not password:
        return jsonify({"status": "error", "message": "username and password required"}), 400
    if len(samples) < 1:
        return jsonify({"status": "error", "message": "no samples provided"}), 400

    df = load_data()
    rows = []

    for s in samples:
        feats = compute_features(s)
        if feats is None:
            continue
        feats["user"] = username
        feats["password"] = password  # stored plain (for project simplicity)
        rows.append(feats)

    if not rows:
        return jsonify({"status": "error", "message": "no valid samples"}), 400

    df_new = pd.DataFrame(rows)
    df_all = pd.concat([df, df_new], ignore_index=True)
    save_data(df_all)

    # train/update global model
    model = train_global_model(df_all)

    users = sorted(df_all["user"].unique().tolist())
    return jsonify({"status": "registered", "users": users, "samples_added": len(rows)})


# ----------------- API: Verify login -----------------

@app.route("/verify", methods=["POST"])
def verify():
    """
    Body JSON:
    {
      "username": "...",
      "password": "...",
      "dwell_times": [ ... ]
    }
    """
    data = request.json
    username = data.get("username", "").strip()
    password = data.get("password", "")
    dwell = data.get("dwell_times", [])

    if not username or not password:
        return jsonify({"result": "fail", "reason": "missing_credentials"}), 400

    feats = compute_features(dwell)
    if feats is None:
        return jsonify({"result": "fail", "reason": "no_keystroke_data"}), 400

    df = load_data()
    if df.empty:
        return jsonify({"result": "fail", "reason": "no_registered_users"}), 200

    # Check if user exists
    df_user = df[df["user"] == username]
    if df_user.empty:
        return jsonify({"result": "fail", "reason": "unknown_user"}), 200

    # Check password match (using first stored password for user)
    stored_password = df_user["password"].iloc[0]
    if password != stored_password:
        return jsonify({"result": "fail", "reason": "wrong_password"}), 200

    # Load or train model
    model = load_model()
    if model is None:
        model = train_global_model(df)
        if model is None:
            return jsonify({"result": "fail", "reason": "model_not_ready"}), 200

    # Prepare feature row
    X = pd.DataFrame([feats])

    needed = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else feature_columns()
    for col in needed:
        if col not in X.columns:
            X[col] = 0.0
    X = X[needed]

    predicted_user = model.predict(X)[0]

    if predicted_user == username:
        # success → add this sample to dataset and retrain model
        feats["user"] = username
        feats["password"] = stored_password
        df_new = pd.concat([df, pd.DataFrame([feats])], ignore_index=True)
        save_data(df_new)
        train_global_model(df_new)
        return jsonify({"result": "success", "predicted": predicted_user}), 200
    else:
        return jsonify({"result": "fail", "predicted": predicted_user}), 200


# ----------------- Pages -----------------

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
