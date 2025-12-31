from flask import Blueprint, request, jsonify
import numpy as np
import json
import datetime


def create_api_blueprint(
    baseline_model,
    scaler,
    logreg,
    FEATURE_NAMES,
    FEATURE_MEANS,
    IDX_TO_LABEL,
    MODEL_VERSION,
    get_history,
    save_prediction,
    get_db_connection,
):
    """
    Factory that creates the API blueprint.

    You register it in app.py like:

        from api import create_api_blueprint
        api_bp = create_api_blueprint(
            baseline_model=baseline_model,
            scaler=scaler,
            logreg=logreg,
            FEATURE_NAMES=FEATURE_NAMES,
            FEATURE_MEANS=FEATURE_MEANS,
            IDX_TO_LABEL=IDX_TO_LABEL,
            MODEL_VERSION=MODEL_VERSION,
            get_history=get_history,
            save_prediction=save_prediction,
            get_db_connection=get_db_connection,
        )
        app.register_blueprint(api_bp)
    """

    api_bp = Blueprint("api", __name__, url_prefix="/api")

    # ---------- Helper: personalized explanation ----------
    def explain_for_class(x_array, target_class_index):
        """
        Personalized explanation for a multiclass logistic regression.

        - x_array: shape (1, n_features) in original scale
        - target_class_index: numeric class index (e.g. 0,1,2,3,...)
        """
        # Scale features using same scaler as training
        x_scaled = scaler.transform(x_array)  # shape (1, n_features)
        coefs = logreg.coef_                  # shape (n_classes, n_features)

        # Find row of coefficients corresponding to this class
        class_indices = logreg.classes_
        class_pos = int(np.where(class_indices == target_class_index)[0][0])

        # Per-feature contributions = coef_k * scaled_feature
        contributions = coefs[class_pos] * x_scaled[0]

        explanation = []
        for feat, val, contrib in zip(FEATURE_NAMES, x_array[0], contributions):
            population_mean = FEATURE_MEANS.get(feat, None)
            explanation.append(
                {
                    "feature": feat,
                    "value": float(val),
                    "population_mean": (
                        float(population_mean)
                        if population_mean is not None
                        else None
                    ),
                    "contribution": float(contrib),
                }
            )

        # Sort by absolute contribution (most important first)
        explanation = sorted(
            explanation, key=lambda d: abs(d["contribution"]), reverse=True
        )
        return explanation

    # ---------- /api/explain ----------
    @api_bp.route("/explain", methods=["POST"])
    def api_explain():
        """
        Expected JSON payload must provide ALL feature values used in the model.

        Required features (keys must match exactly):

          age
          sex
          on thyroxine
          query on thyroxine
          on antithyroid medication
          sick
          pregnant
          thyroid surgery
          I131 treatment
          query hypothyroid
          query hyperthyroid
          lithium
          goitre
          tumor
          hypopituitary
          psych
          TSH measured
          TSH
          T3 measured
          TT4 measured
          TT4
          T4U measured
          T4U
          FTI measured
          FTI

        Optional:
          patient_id
        """

        data = request.get_json() or {}
        patient_id = data.get("patient_id") or "anonymous"

        x = []
        try:
            for feat in FEATURE_NAMES:
                value = data[feat]  # must exist
                x_val = float(value)
                x.append(x_val)
        except (KeyError, ValueError, TypeError) as e:
            return (
                jsonify(
                    {
                        "error": (
                            "Missing or invalid feature values. "
                            "You must send all features exactly as in FEATURE_NAMES."
                        ),
                        "details": str(e),
                        "expected_features": FEATURE_NAMES,
                    }
                ),
                400,
            )

        x_array = np.array(x).reshape(1, -1)

        # Predict class probabilities
        probs = baseline_model.predict_proba(x_array)[0]  # aligned with logreg.classes_

        # Numeric class index of prediction
        pred_idx = int(np.argmax(probs))
        pred_label = IDX_TO_LABEL[pred_idx]
        pred_prob = float(probs[np.where(logreg.classes_ == pred_idx)][0])

        class_probabilities = {
            IDX_TO_LABEL[int(idx)]: float(prob)
            for idx, prob in zip(logreg.classes_, probs)
        }

        explanation = explain_for_class(x_array, pred_idx)

        features_dict = {feat: data.get(feat) for feat in FEATURE_NAMES}

        # Persist prediction
        save_prediction(
            patient_id=patient_id,
            features_dict=features_dict,
            pred_idx=pred_idx,
            pred_label=pred_label,
            prob=pred_prob,
        )

        return (
            jsonify(
                {
                    "patient_id": patient_id,
                    "prediction_index": pred_idx,
                    "prediction_label": pred_label,
                    "prediction_probability": pred_prob,
                    "class_probabilities": class_probabilities,
                    "explanation": explanation,
                    "model_version": MODEL_VERSION,
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                }
            ),
            200,
        )

    # ---------- /api/history/<patient_id> ----------
    @api_bp.route("/history/<patient_id>", methods=["GET"])
    def api_history(patient_id):
        history = get_history(patient_id)
        return jsonify(
            {
                "patient_id": patient_id,
                "count": len(history),
                "records": history,
            }
        )

    # ---------- Optional: /api/monitor-summary (JSON) ----------
    @api_bp.route("/monitor-summary", methods=["GET"])
    def api_monitor_summary():
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), AVG(probability) FROM predictions")
        total, avg_prob = cursor.fetchone()
        conn.close()

        return jsonify(
            {
                "total_predictions": total or 0,
                "average_probability": (
                    float(avg_prob) if avg_prob is not None else None
                ),
            }
        )

    return api_bp
