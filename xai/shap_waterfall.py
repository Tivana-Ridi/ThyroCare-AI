# xai/shap_waterfall.py
import io
import base64
import numpy as np

# SHAP and plotting are optional — we handle failures gracefully in the API
try:
    import shap
    import matplotlib.pyplot as plt
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    plt = None
    SHAP_AVAILABLE = False


def _to_data_uri_png(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=160)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return "data:image/png;base64," + b64


def _get_wrapped_estimator(calibrated_model):
    """
    CalibratedClassifierCV stores the base model in .base_estimator (common),
    but sometimes in .estimator. This tries both.
    """
    if hasattr(calibrated_model, "base_estimator") and calibrated_model.base_estimator is not None:
        return calibrated_model.base_estimator
    if hasattr(calibrated_model, "estimator") and calibrated_model.estimator is not None:
        return calibrated_model.estimator
    return calibrated_model  # fallback


class FusionWaterfallExplainer:
    """
    Produces a Fusion Waterfall image for a target class by:
    - explaining LR (linear) in scaled space
    - explaining RF (tree) in scaled space
    - fusing: values = 0.5*LR + 0.5*RF (same for base_value)
    """

    def __init__(self, scaler, lr_cal, rf_cal, feature_names, background_scaled, w_lr=0.5, w_rf=0.5):
        self.scaler = scaler
        self.lr_cal = lr_cal
        self.rf_cal = rf_cal
        self.feature_names = list(feature_names)
        self.background_scaled = np.array(background_scaled, dtype=float)
        self.w_lr = float(w_lr)
        self.w_rf = float(w_rf)

        if not SHAP_AVAILABLE:
            return

        # unwrap calibrated models for SHAP explainers
        self.lr_est = _get_wrapped_estimator(lr_cal)
        self.rf_est = _get_wrapped_estimator(rf_cal)

        # Create explainers once (fast reuse)
        # LR: LinearExplainer
        self.lr_explainer = shap.LinearExplainer(
            self.lr_est,
            self.background_scaled,
            feature_perturbation="interventional"
        )

        # RF: TreeExplainer
        # model_output="probability" works for many tree models; if it fails, SHAP will throw and we handle in API
        self.rf_explainer = shap.TreeExplainer(
            self.rf_est,
            self.background_scaled,
            feature_perturbation="interventional",
            model_output="probability"
        )

    def fusion_predict_proba(self, x_raw_row: np.ndarray) -> np.ndarray:
        x_scaled = self.scaler.transform(x_raw_row.reshape(1, -1))
        p_lr = self.lr_cal.predict_proba(x_scaled)[0]
        p_rf = self.rf_cal.predict_proba(x_scaled)[0]
        p = (self.w_lr * p_lr) + (self.w_rf * p_rf)
        # normalize (safe)
        s = float(np.sum(p))
        if s > 0:
            p = p / s
        return p

    def _pick_class_pos(self, class_index: int) -> int:
        classes = getattr(self.lr_cal, "classes_", None)
        if classes is None:
            raise ValueError("lr_cal has no classes_ attribute.")
        classes = np.array(classes)
        pos = np.where(classes == class_index)[0]
        if len(pos) == 0:
            raise ValueError(f"class_index {class_index} not found in model classes {classes}.")
        return int(pos[0])

    def _get_shap_for_model(self, explainer, x_scaled_row: np.ndarray, class_pos: int):
        """
        Returns (values, base_value) for given class position.
        Works with SHAP outputs that are list (per class) or ndarray.
        """
        sv = explainer.shap_values(x_scaled_row.reshape(1, -1))

        # expected value might be list/array per class
        ev = explainer.expected_value

        if isinstance(sv, list):
            values = np.array(sv[class_pos][0], dtype=float)
        else:
            # sometimes shape is (1, n_features) for binary or (1, n_features, n_classes)
            sv = np.array(sv)
            if sv.ndim == 2:
                values = sv[0]
            elif sv.ndim == 3:
                values = sv[0, :, class_pos]
            else:
                raise ValueError(f"Unexpected SHAP shape: {sv.shape}")

        if isinstance(ev, (list, np.ndarray)):
            base_value = float(np.array(ev)[class_pos])
        else:
            base_value = float(ev)

        return values, base_value

    def waterfall_png_data_uri(self, x_raw_row: np.ndarray, class_index: int, max_display: int = 12) -> str:
        if not SHAP_AVAILABLE:
            raise RuntimeError("SHAP is not available in this environment.")

        x_raw_row = np.array(x_raw_row, dtype=float)
        x_scaled_row = self.scaler.transform(x_raw_row.reshape(1, -1))[0]

        class_pos = self._pick_class_pos(class_index)

        # LR SHAP
        lr_vals, lr_base = self._get_shap_for_model(self.lr_explainer, x_scaled_row, class_pos)

        # RF SHAP
        rf_vals, rf_base = self._get_shap_for_model(self.rf_explainer, x_scaled_row, class_pos)

        # Fuse
        fusion_vals = (self.w_lr * lr_vals) + (self.w_rf * rf_vals)
        fusion_base = (self.w_lr * lr_base) + (self.w_rf * rf_base)

        # Build Explanation object for waterfall plot
        exp = shap.Explanation(
            values=fusion_vals,
            base_values=fusion_base,
            data=x_raw_row,
            feature_names=self.feature_names
        )

        fig = plt.figure()
        shap.plots.waterfall(exp, max_display=max_display, show=False)

        title = f"Fusion Waterfall (LR + RF) — Class {class_index}"
        plt.title(title)

        return _to_data_uri_png(fig)
