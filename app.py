# app.py
from flask import (
    Flask, request, jsonify, render_template,
    redirect, url_for, flash, session
)
import os
import joblib
import json
import numpy as np
import datetime
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import warnings
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

from config import (
    MODEL_PATH, FEATURE_META_PATH, TRAIN_STATS_PATH,
    CLASS_NAMES_PATH, DB_PATH, MODEL_VERSION, SECRET_KEY
)

# -------------------------------
# Optional SHAP support
# -------------------------------
try:
    from xai.shap_waterfall import FusionWaterfallExplainer, SHAP_AVAILABLE
except Exception as e:
    FusionWaterfallExplainer = None
    SHAP_AVAILABLE = False
    print(f"[WARN] SHAP import failed: {e}. To enable SHAP, run: pip install shap matplotlib")


app = Flask(__name__)
app.secret_key = SECRET_KEY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 1) Load model and metadata
# ============================================================
print(f"[INFO] scikit-learn version: {sklearn.__version__}")
try:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", InconsistentVersionWarning)
        baseline_model = joblib.load(MODEL_PATH)
        for warn in w:
            if issubclass(warn.category, InconsistentVersionWarning):
                print(f"[WARN] InconsistentVersionWarning while unpickling model: {warn.message}")
except Exception as e:
    print(f"[ERROR] Failed to load model from {MODEL_PATH}: {e}")
    raise

with open(FEATURE_META_PATH, "r") as f:
    feature_meta = json.load(f)

with open(TRAIN_STATS_PATH, "r") as f:
    training_stats = json.load(f)

with open(CLASS_NAMES_PATH, "r") as f:
    class_info = json.load(f)

FEATURE_NAMES = feature_meta["feature_names"]
FEATURE_MEANS = training_stats.get("feature_means", {})
CLASS_NAMES = class_info["classes"]

# ---- IMPORTANT: your pkl is a dict, not a Pipeline ----
if not isinstance(baseline_model, dict):
    raise TypeError(
        f"Expected baseline_model to be a dict, got {type(baseline_model)}. "
        "Your code assumes {'scaler','lr_cal','rf_cal'}."
    )

required_keys = {"scaler", "lr_cal", "rf_cal"}
missing_keys = required_keys - set(baseline_model.keys())
if missing_keys:
    raise KeyError(
        f"baseline_model is missing keys: {missing_keys}. Found keys: {list(baseline_model.keys())}"
    )

scaler = baseline_model["scaler"]
lr_cal = baseline_model["lr_cal"]
rf_cal = baseline_model["rf_cal"]

MODEL_CLASSES = getattr(lr_cal, "classes_", None)
if MODEL_CLASSES is None:
    raise AttributeError("lr_cal has no attribute classes_. Is it a classifier?")

rf_classes = getattr(rf_cal, "classes_", None)
if rf_classes is None or not np.array_equal(np.array(MODEL_CLASSES), np.array(rf_classes)):
    raise ValueError(
        "lr_cal.classes_ and rf_cal.classes_ do not match. "
        f"lr_cal.classes_={MODEL_CLASSES}, rf_cal.classes_={rf_classes}"
    )

MODEL_CLASSES = np.array(MODEL_CLASSES)

# ============================================================
# ✅ FIX: Correct label mapping by POSITION (proba index), not by class-id
# probs[i] corresponds to MODEL_CLASSES[i], so label must be CLASS_NAMES[i]
# ============================================================
if len(CLASS_NAMES) != len(MODEL_CLASSES):
    raise ValueError(
        f"CLASS_NAMES length ({len(CLASS_NAMES)}) != MODEL_CLASSES length ({len(MODEL_CLASSES)}). "
        "Your class_names.json order must match model predict_proba columns."
    )

# class-id -> label (safe mapping)
IDX_TO_LABEL = {int(MODEL_CLASSES[pos]): str(CLASS_NAMES[pos]) for pos in range(len(MODEL_CLASSES))}

print("[DEBUG] MODEL_CLASSES:", MODEL_CLASSES)
print("[DEBUG] CLASS_NAMES:", CLASS_NAMES)
print("[DEBUG] IDX_TO_LABEL:", IDX_TO_LABEL)

#def class_idx_to_label(class_idx: int) -> str:
   # try:
        #i = int(class_idx)
    #except Exception:
        #return str(class_idx)

   # if 0 <= i < len(CLASS_NAMES):
      #  return CLASS_NAMES[i]
#    return f"Class {i}"


#IDX_TO_LABEL = {int(c): class_idx_to_label(int(c)) for c in MODEL_CLASSES}


# ============================================================
# 1.1) SHAP background + Fusion Explainer (optional)
# ============================================================
FUSION_SHAP = None
SHAP_BG_PATH = os.path.join(BASE_DIR, "models", "shap_background.npy")

if SHAP_AVAILABLE and FusionWaterfallExplainer is not None:
    if os.path.exists(SHAP_BG_PATH):
        try:
            SHAP_BG = np.load(SHAP_BG_PATH)  # scaled background (n_bg, n_features)
            FUSION_SHAP = FusionWaterfallExplainer(
                scaler=scaler,
                lr_cal=lr_cal,
                rf_cal=rf_cal,
                feature_names=FEATURE_NAMES,
                background_scaled=SHAP_BG,
                w_lr=0.5,
                w_rf=0.5
            )
        except Exception as e:
            print(f"[WARN] SHAP explainer init failed: {e}")
            FUSION_SHAP = None
    else:
        print(f"[WARN] SHAP background not found: {SHAP_BG_PATH}. Waterfall will be unavailable.")
else:
    print("[WARN] SHAP not available. Waterfall will be unavailable.")


# ============================================================
# 2) DB helpers
# ============================================================
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, username, email FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return row


def login_required(view_func):
    from functools import wraps

    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login", next=request.path))
        return view_func(*args, **kwargs)

    return wrapper


def save_prediction(user_id, features_dict, pred_idx, pred_label, prob):
    conn = get_db_connection()
    cursor = conn.cursor()
    timestamp = datetime.datetime.utcnow().isoformat()
    features_json = json.dumps(features_dict)

    cursor.execute("""
        INSERT INTO predictions (
            user_id, timestamp, features_json,
            prediction_index, prediction_label,
            probability, model_version, true_label
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        timestamp,
        features_json,
        int(pred_idx),
        pred_label,
        float(prob),
        MODEL_VERSION,
        None
    ))

    pred_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return pred_id


def get_user_history(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, timestamp, features_json,
               prediction_index, prediction_label,
               probability, model_version, true_label
        FROM predictions
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT 50
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()

    history = []
    for row in rows:
        history.append({
            "id": row["id"],
            "timestamp": row["timestamp"],
            "features": json.loads(row["features_json"]),
            "prediction_index": row["prediction_index"],
            "prediction_label": row["prediction_label"],
            "probability": row["probability"],
            "model_version": row["model_version"],
            "true_label": row["true_label"]
        })
    return history


# ============================================================
# 3) Prediction + lightweight explanation
# ============================================================
def predict_fusion_proba(x_array: np.ndarray) -> np.ndarray:
    """
    x_array: shape (1, n_features) in RAW feature space (not scaled).
    Returns fused probabilities in the order of MODEL_CLASSES.
    """
    x_scaled = scaler.transform(x_array)

    probs_lr = lr_cal.predict_proba(x_scaled)[0]
    probs_rf = rf_cal.predict_proba(x_scaled)[0]

    probs = (probs_lr + probs_rf) / 2.0

    # normalize for safety
    s = float(np.sum(probs))
    if s > 0:
        probs = probs / s

    return probs


def explain_for_class(x_array, target_class_index):
    """
    Lightweight explanation (fast) — ranks features by absolute deviation from population mean.
    NOTE: "contribution" here is an importance score, not true SHAP.
    """
    explanation = []
    for i, feat in enumerate(FEATURE_NAMES):
        val = float(x_array[0][i])
        mean = FEATURE_MEANS.get(feat, None)

        if mean is None:
            pop_mean = None
            score = 0.0
        else:
            pop_mean = float(mean)
            score = abs(val - pop_mean)

        explanation.append({
            "feature": feat,
            "value": val,
            "population_mean": pop_mean,
            "contribution": float(score)
        })

    explanation.sort(key=lambda d: d["contribution"], reverse=True)
    return explanation


# ============================================================
# 4) Auth routes
# ============================================================
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")

        if not username or not email or not password:
            flash("All fields are required.", "danger")
            return render_template("signup.html")

        if password != confirm:
            flash("Passwords do not match.", "danger")
            return render_template("signup.html")

        password_hash = generate_password_hash(password)
        created_at = datetime.datetime.utcnow().isoformat()

        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO users (username, email, password_hash, created_at)
                VALUES (?, ?, ?, ?)
            """, (username, email, password_hash, created_at))
            conn.commit()
        except sqlite3.IntegrityError:
            flash("Username or email already exists.", "danger")
            conn.close()
            return render_template("signup.html")

        conn.close()
        flash("Account created successfully. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, username, password_hash FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        conn.close()

        if not row or not check_password_hash(row["password_hash"], password):
            flash("Invalid username or password.", "danger")
            return render_template("login.html")

        session["user_id"] = row["id"]
        session["username"] = row["username"]
        flash("Logged in successfully.", "success")
        next_url = request.args.get("next") or url_for("index")
        return redirect(next_url)

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))


# ============================================================
# 5) Frontend routes
# ============================================================
@app.route("/")
def index():
    user = get_current_user()
    return render_template("index.html", user=user)


@app.route("/predict-page")
@login_required
def predict_page():
    user = get_current_user()
    return render_template("predict.html", user=user, feature_names=FEATURE_NAMES)


@app.route("/result")
@login_required
def result_page():
    user = get_current_user()
    return render_template("result.html", user=user)


@app.route("/history")
@login_required
def history_page():
    user = get_current_user()
    history = get_user_history(user["id"])
    return render_template("history.html", user=user, history=history)


@app.route("/monitoring")
@login_required
def monitoring_page():
    user = get_current_user()
    return render_template("monitoring.html", user=user)


# ============================================================
# 6) API: prediction + explanation
# ============================================================
@app.route("/api/explain", methods=["POST"])
@login_required
def api_explain():
    user = get_current_user()
    if not user:
        return jsonify({"error": "Not authenticated"}), 401

    data = request.get_json() or {}

    def get_feature_value(feat):
        if feat == "binaryClass" and feat not in data:
            return 0.0
        return float(data[feat])

    # collect features in the correct order
    try:
        x = [get_feature_value(feat) for feat in FEATURE_NAMES]
    except Exception as e:
        return jsonify({
            "error": "Missing or invalid feature values.",
            "details": str(e),
            "expected_features": FEATURE_NAMES
        }), 400

    x_array = np.array(x, dtype=float).reshape(1, -1)

    # fused prediction (order = MODEL_CLASSES)
    probs = predict_fusion_proba(x_array)

    # ✅ FIX: use POSITION for label (no class_idx_to_label)
    pred_pos = int(np.argmax(probs))
    pred_idx = int(MODEL_CLASSES[pred_pos])      # numeric class id (0..4) for DB + SHAP
    pred_label = str(CLASS_NAMES[pred_pos])      # ✅ correct label
    pred_prob = float(probs[pred_pos])

    # ✅ FIX: label probabilities by position too
    class_probabilities = {
        str(CLASS_NAMES[i]): float(probs[i]) for i in range(len(probs))
    }

    # lightweight explanation (fast)
    explanation = explain_for_class(x_array, pred_idx)

    # store numeric features in DB
    features_dict = {feat: get_feature_value(feat) for feat in FEATURE_NAMES}
    pred_id = save_prediction(
        user_id=user["id"],
        features_dict=features_dict,
        pred_idx=pred_idx,
        pred_label=pred_label,
        prob=pred_prob
    )

    return jsonify({
        "prediction_id": int(pred_id),
        "prediction_index": pred_idx,
        "prediction_label": pred_label,
        "prediction_probability": pred_prob,
        "class_probabilities": class_probabilities,
        "explanation": explanation,

        # IMPORTANT: send inputs so result.html can call SHAP
        "inputs": features_dict,

        "model_version": MODEL_VERSION,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }), 200


# ============================================================
# 6.1) API: SHAP Fusion Waterfall
# ============================================================
@app.route("/api/shap-waterfall", methods=["POST"])
@login_required
def api_shap_waterfall():
    if FUSION_SHAP is None:
        return jsonify({
            "error": "SHAP waterfall is not available on this server.",
            "hint": "Install shap+matplotlib and add models/shap_background.npy"
        }), 400

    data = request.get_json() or {}
    inputs = data.get("inputs", data)  # allow nested or direct

    # Build input vector in correct order
    try:
        x = [float(inputs[feat]) for feat in FEATURE_NAMES]
    except Exception as e:
        return jsonify({
            "error": "Invalid or missing inputs for SHAP.",
            "details": str(e),
            "expected_features": FEATURE_NAMES
        }), 400

    x_raw = np.array(x, dtype=float)

    # Which class to explain?
    class_index = data.get("class_index", None)
    try:
        if class_index is None:
            probs = FUSION_SHAP.fusion_predict_proba(x_raw)
            class_pos = int(np.argmax(probs))
            class_index = int(lr_cal.classes_[class_pos])
        else:
            class_index = int(class_index)
    except Exception as e:
        return jsonify({"error": "Failed to select class for explanation.", "details": str(e)}), 400

    # Generate waterfall image
    try:
        img_uri = FUSION_SHAP.waterfall_png_data_uri(
            x_raw_row=x_raw,
            class_index=class_index,
            max_display=12
        )
        return jsonify({
            "image": img_uri,
            "class_index": int(class_index)
        }), 200
    except Exception as e:
        return jsonify({
            "error": "Failed to generate SHAP waterfall.",
            "details": str(e)
        }), 500


# ============================================================
# 7) Monitoring API
# ============================================================
@app.route("/monitor-summary", methods=["GET"])
@login_required
def monitor_summary():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), AVG(probability) FROM predictions")
    total, avg_prob = cursor.fetchone()
    conn.close()
    return jsonify({
        "total_predictions": int(total or 0),
        "average_probability": float(avg_prob) if avg_prob is not None else None
    })

@app.errorhandler(500)
def handle_500(e):
    # Return JSON instead of HTML for internal server errors
    return jsonify({
        "error": "Internal server error",
        "details": str(e)
    }), 500

# ============================================================
# 8) Run
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
