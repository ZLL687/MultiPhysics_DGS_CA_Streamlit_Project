# -*- coding: utf-8 -*-
"""
Streamlit GUI for CFST eccentric load–strain curve prediction using the new
MultiPhysics-DGS-CA model package.

This file replaces the old GUI that used:
    voting_regressor.pkl + scaler_y.pkl

The new GUI uses:
    ea_pgcc_dgs_ca_model_package.pkl

Prediction logic:
    1. Build raw features:
       D/t, L/D, Helix Angle, fcuAc, fyAs, h/l, Concrete Types, e/r,
       Confinement Factor, Strain
    2. Apply the saved categorical mapping from the model package.
    3. Reconstruct physics-guided features:
       N0_Enhanced, PhysicsReduction, N_phy, BoundaryFactor
    4. Predict correction ratio R by Dynamic Gated Stacking.
    5. Blend DGS with CatBoost anchor:
       R_final = anchor_alpha * R_DGS + (1-anchor_alpha) * R_CatBoost
    6. Recover capacity:
       Nu_pred = BoundaryFactor * N_phy * R_final
"""

import io
import math
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.signal import savgol_filter

from sklearn.pipeline import make_pipeline as sklearn_make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# Optional imports needed when joblib deserializes the saved model package.
try:
    import xgboost as xgb  # noqa: F401
except Exception:
    pass

try:
    from lightgbm import LGBMRegressor  # noqa: F401
except Exception:
    pass

try:
    from catboost import CatBoostRegressor  # noqa: F401
except Exception:
    pass

try:
    import gdown
except Exception:
    gdown = None


warnings.filterwarnings("ignore")

# ============================================================
# 0. App configuration
# ============================================================

APP_TITLE = "CFST Eccentric Load–Strain Curve Prediction System"
ALGORITHM_TITLE = "MultiPhysics-DGS-CA"

MODEL_DIR = Path("models")
MODEL_PACKAGE_PATH = MODEL_DIR / "ea_pgcc_dgs_ca_model_package.pkl"

# Optional: paste your Google Drive file ID here if you want automatic download.
# Leave empty if you prefer local file or manual upload.
MODEL_PACKAGE_GDRIVE_ID = "1TVYByfRL6Nt2WMQ2MEG9XBNKj_ZXxUC0"
MODEL_PACKAGE_GDRIVE_URL = "https://drive.google.com/file/d/1TVYByfRL6Nt2WMQ2MEG9XBNKj_ZXxUC0/view?usp=drive_link"

DEFAULT_RAW_FEATURE_COLS = [
    "D/t",
    "L/D",
    "Helix Angle",
    "fcuAc",
    "fyAs",
    "h/l",
    "Concrete Types",
    "e/r",
    "Confinement Factor",
    "Strain",
]

DEFAULT_TARGET_COL = "Nu"
DEFAULT_ID_COL = "Specimen ID"
DEFAULT_STRAIN_COL = "Strain"

FIG_DPI = 300

plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]
plt.rcParams["font.size"] = 10
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 1. Functions required for joblib deserialization
# ============================================================

def stable_softmax(z, axis=1):
    z = np.asarray(z, dtype=float)
    z = z - np.max(z, axis=axis, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)


class DynamicGatedStackingRegressor:
    """
    Keep this class name unchanged.

    The model package was saved with a custom DynamicGatedStackingRegressor
    object. Streamlit must define a compatible class before joblib.load().
    """

    def __init__(self, base_estimators, gate_model=None, gate_temperature=None, random_state=42):
        self.base_estimators = base_estimators
        self.gate_temperature = gate_temperature
        self.random_state = random_state
        self.gate_model = gate_model if gate_model else sklearn_make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=1.0),
        )

    def _predict_weights(self, X):
        return stable_softmax(
            np.column_stack([gm.predict(X) for gm in self.gate_models_]),
            axis=1,
        )

    def predict(self, X, return_weights=False, return_base_pred=False):
        base_pred = np.column_stack([model.predict(X) for _, model in self.final_estimators_])
        weights = self._predict_weights(X)
        dgs_pred = np.sum(base_pred * weights, axis=1)

        outputs = [dgs_pred]
        if return_weights:
            outputs.append(weights)
        if return_base_pred:
            outputs.append(base_pred)

        return outputs[0] if len(outputs) == 1 else tuple(outputs)


# ============================================================
# 2. Model loading utilities
# ============================================================

def extract_google_drive_file_id(url_or_id):
    """
    Accept either a Google Drive sharing URL or a raw file ID.
    """
    text = str(url_or_id or "").strip()
    if not text:
        return ""

    match = re.search(r"/d/([a-zA-Z0-9_-]+)", text)
    if match:
        return match.group(1)

    match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", text)
    if match:
        return match.group(1)

    return text


def patch_simple_imputer_for_sklearn_compatibility(imputer):
    """
    Compatibility patch for sklearn SimpleImputer objects loaded from older
    joblib/pickle files. This avoids errors such as:
        'SimpleImputer' object has no attribute '_fill_dtype'
    """
    if not isinstance(imputer, SimpleImputer):
        return imputer

    if hasattr(imputer, "statistics_"):
        stats = np.asarray(imputer.statistics_)
        dtype = stats.dtype if stats.dtype != object else np.dtype(float)
    else:
        dtype = np.dtype(float)

    if not hasattr(imputer, "_fit_dtype"):
        imputer._fit_dtype = dtype
    if not hasattr(imputer, "_fill_dtype"):
        imputer._fill_dtype = dtype
    if not hasattr(imputer, "keep_empty_features"):
        imputer.keep_empty_features = False

    return imputer


def patch_pipeline_imputers_for_sklearn_compatibility(estimator):
    """Patch SimpleImputer instances inside sklearn Pipelines."""
    if estimator is None:
        return estimator

    if isinstance(estimator, SimpleImputer):
        return patch_simple_imputer_for_sklearn_compatibility(estimator)

    if hasattr(estimator, "named_steps"):
        for _, step in estimator.named_steps.items():
            patch_pipeline_imputers_for_sklearn_compatibility(step)

    return estimator


def patch_model_package_for_sklearn_compatibility(model_package):
    """
    Patch SimpleImputer objects inside the loaded MultiPhysics-DGS-CA package.
    """
    try:
        model = model_package.get("model", None) if isinstance(model_package, dict) else None
        if model is None:
            return model_package

        for _, estimator in getattr(model, "final_estimators_", []):
            patch_pipeline_imputers_for_sklearn_compatibility(estimator)

        for gate_model in getattr(model, "gate_models_", []):
            patch_pipeline_imputers_for_sklearn_compatibility(gate_model)

        patch_pipeline_imputers_for_sklearn_compatibility(getattr(model, "gate_model", None))

        for _, estimator in getattr(model, "base_estimators", []):
            patch_pipeline_imputers_for_sklearn_compatibility(estimator)

    except Exception as exc:
        st.warning(f"Model compatibility patch issued a warning: {exc}")

    return model_package



def download_from_gdrive(file_id_or_url, output_path):
    """Download model package from Google Drive if gdown is available."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_id = extract_google_drive_file_id(file_id_or_url)
    if not file_id:
        return False

    if gdown is None:
        st.error("gdown is not installed. Please add `gdown` to requirements.txt.")
        return False

    if output_path.exists() and output_path.stat().st_size > 1024:
        return True

    with st.spinner(f"Downloading {output_path.name} from Google Drive..."):
        try:
            try:
                downloaded_path = gdown.download(
                    id=file_id,
                    output=str(output_path),
                    quiet=False,
                    fuzzy=True,
                )
            except TypeError:
                url = f"https://drive.google.com/uc?id={file_id}&export=download"
                downloaded_path = gdown.download(url, str(output_path), quiet=False)

            ok = (
                downloaded_path is not None
                and output_path.exists()
                and output_path.stat().st_size > 1024
            )

            if not ok:
                st.error(
                    "Download failed or downloaded file is invalid. "
                    "Please make sure the Google Drive file permission is set to "
                    "'Anyone with the link can view'."
                )
            return ok

        except Exception as exc:
            st.error(f"Download failed: {exc}")
            st.info(
                "Please check that the Google Drive model file is shared as "
                "'Anyone with the link can view'."
            )
            return False

def load_model_package_from_path(model_path):
    try:
        model_package = joblib.load(model_path)
        model_package = patch_model_package_for_sklearn_compatibility(model_package)
        return model_package
    except Exception as exc:
        st.error(f"Model package loading failed: {exc}")
        return None


def load_model_package_from_upload(uploaded_file):
    try:
        model_package = joblib.load(io.BytesIO(uploaded_file.read()))
        model_package = patch_model_package_for_sklearn_compatibility(model_package)
        return model_package
    except Exception as exc:
        st.error(f"Uploaded model package loading failed: {exc}")
        return None


# ============================================================
# 3. New model inference utilities
# ============================================================

def zero_boundary_factor(strain_values, eps0, mode="hard", power=2.0, zero_eps=1e-12):
    strain_values = pd.to_numeric(pd.Series(strain_values), errors="coerce").to_numpy(dtype=float)
    abs_strain = np.abs(strain_values)

    if mode == "hard":
        factor = np.ones_like(abs_strain, dtype=float)
        factor[np.isclose(abs_strain, 0.0, atol=zero_eps, rtol=0.0)] = 0.0
        return factor

    if mode == "smooth":
        eps0 = max(float(eps0), 1e-12)
        factor = 1.0 - np.exp(-np.power(abs_strain / eps0, power))
        factor[np.isclose(abs_strain, 0.0, atol=zero_eps, rtol=0.0)] = 0.0
        return factor

    raise ValueError(f"Unknown BoundaryFactor mode: {mode}")


def _categorical_candidates(value):
    """Return robust string candidates for categorical mapping."""
    if pd.isna(value):
        return ["Missing"]

    candidates = []
    text = str(value).strip()
    candidates.append(text)

    try:
        v = float(value)
        candidates.append(f"{v:g}")
        candidates.append(str(v))
        if float(v).is_integer():
            candidates.append(str(int(v)))
            candidates.append(f"{int(v)}.0")
    except Exception:
        pass

    # Preserve order while removing duplicates
    unique = []
    for item in candidates:
        if item not in unique:
            unique.append(item)
    return unique


def apply_saved_categorical_mappings(X_base, model_package):
    """
    Apply categorical mappings saved during training.
    Unknown categories are set to NaN and handled by the imputer inside the model pipeline.
    """
    X_base = X_base.copy()
    mappings = model_package.get("categorical_mappings", {}) or {}

    for col, mapping in mappings.items():
        if col not in X_base.columns:
            continue

        if mapping is None:
            X_base[col] = pd.to_numeric(X_base[col], errors="coerce")
            continue

        def map_one(value):
            for cand in _categorical_candidates(value):
                if cand in mapping:
                    return mapping[cand]
            return np.nan

        X_base[col] = X_base[col].apply(map_one).astype(float)

    return X_base


def add_physics_feature_for_new_model(X_base, model_package):
    """Reconstruct N0_Enhanced, PhysicsReduction, and N_phy."""
    X = X_base.copy()

    e_over_r_col = model_package.get("e_over_r_col", "e/r")
    l_over_d_col = model_package.get("l_over_d_col", "L/D")
    xi_col = model_package.get("xi_col", "Confinement Factor")

    n0_enhanced_col = model_package.get("n0_enhanced_col", "N0_Enhanced")
    physics_reduction_col = model_package.get("physics_reduction_col", "PhysicsReduction")
    physics_col = model_package.get("physics_col", "N_phy")

    alpha = float(model_package.get("physics_alpha", 1.0))
    beta = float(model_package.get("physics_beta", 0.0))
    gamma = float(model_package.get("physics_gamma", 0.0))
    n0_unit_scale = float(model_package.get("n0_unit_scale", 1000.0))

    required_cols = ["fcuAc", "fyAs", e_over_r_col, l_over_d_col, xi_col]
    missing = [col for col in required_cols if col not in X.columns]
    if missing:
        raise ValueError(f"Missing columns required by physics features: {missing}")

    for col in required_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    fcuAc = X["fcuAc"].fillna(0).to_numpy(dtype=float)
    fyAs = X["fyAs"].fillna(0).to_numpy(dtype=float)
    xi = np.maximum(X[xi_col].fillna(0).to_numpy(dtype=float), 0.0)
    e_over_r = np.maximum(X[e_over_r_col].fillna(0).to_numpy(dtype=float), 0.0)
    l_over_d = np.maximum(X[l_over_d_col].fillna(0).to_numpy(dtype=float), 0.0)

    X[n0_enhanced_col] = (fcuAc * (1.0 + alpha * xi) + fyAs) / n0_unit_scale
    X[physics_reduction_col] = 1.0 / (1.0 + beta * e_over_r + gamma * l_over_d)
    X[physics_col] = X[n0_enhanced_col] * X[physics_reduction_col]

    return X


def build_model_matrix_for_new_model(X_raw, model_package):
    X = X_raw.copy()

    strain_col = model_package.get("strain_col", DEFAULT_STRAIN_COL)
    boundary_col = model_package.get("boundary_col", "BoundaryFactor")
    boundary_mode = model_package.get("boundary_mode", "hard")
    boundary_power = float(model_package.get("boundary_power", 2.0))
    boundary_eps0 = float(model_package.get("boundary_eps0", 1.0))

    if strain_col not in X.columns:
        raise ValueError(f"Missing strain column: {strain_col}")

    X[boundary_col] = zero_boundary_factor(
        X[strain_col],
        eps0=boundary_eps0,
        mode=boundary_mode,
        power=boundary_power,
    )

    model_feature_cols = list(model_package.get("model_feature_cols", []))
    if not model_feature_cols:
        raise ValueError("model_package does not contain model_feature_cols.")

    for col in model_feature_cols:
        if col not in X.columns:
            X[col] = np.nan

    return X[model_feature_cols]


def prepare_X_model(df_raw_input, model_package):
    raw_feature_cols = list(model_package.get("raw_feature_cols", DEFAULT_RAW_FEATURE_COLS))
    missing = [col for col in raw_feature_cols if col not in df_raw_input.columns]
    if missing:
        raise ValueError(f"Input data are missing raw_feature_cols: {missing}")

    X_base = df_raw_input[raw_feature_cols].copy()

    categorical_cols = set((model_package.get("categorical_mappings", {}) or {}).keys())
    for col in raw_feature_cols:
        if col not in categorical_cols:
            X_base[col] = pd.to_numeric(X_base[col], errors="coerce")

    X_base = apply_saved_categorical_mappings(X_base, model_package)
    X_raw = add_physics_feature_for_new_model(X_base, model_package)
    X_model = build_model_matrix_for_new_model(X_raw, model_package)

    return X_model


def capacity_from_ratio_new_model(ratio_pred, X_model, model_package):
    ratio_pred = np.asarray(ratio_pred, dtype=float).copy()

    physics_col = model_package.get("physics_col", "N_phy")
    boundary_col = model_package.get("boundary_col", "BoundaryFactor")
    strain_col = model_package.get("strain_col", DEFAULT_STRAIN_COL)

    use_non_negative = bool(model_package.get("use_non_negative_constraint", True))
    if use_non_negative:
        ratio_pred = np.maximum(ratio_pred, 0.0)

    n_phy = pd.to_numeric(X_model[physics_col], errors="coerce").to_numpy(dtype=float)
    boundary = pd.to_numeric(X_model[boundary_col], errors="coerce").to_numpy(dtype=float)

    y_pred = boundary * n_phy * ratio_pred

    if use_non_negative:
        y_pred = np.maximum(y_pred, 0.0)

    strain_values = pd.to_numeric(X_model[strain_col], errors="coerce").to_numpy(dtype=float)
    zero_mask = np.isclose(strain_values, 0.0, atol=1e-12, rtol=0.0)
    y_pred[zero_mask] = 0.0

    return y_pred


def predict_capacity_with_new_model(df_raw_input, model_package):
    model = model_package["model"]
    X_model = prepare_X_model(df_raw_input, model_package)

    dgs_ratio_pred, weights, base_ratio_pred = model.predict(
        X_model,
        return_weights=True,
        return_base_pred=True,
    )

    use_anchor = bool(model_package.get("use_catboost_anchor", True))
    anchor_alpha = float(model_package.get("anchor_alpha", 1.0))
    anchor_model_index = int(model_package.get("anchor_model_index", 0))

    if use_anchor and base_ratio_pred.ndim == 2 and base_ratio_pred.shape[1] > anchor_model_index:
        anchor_ratio_pred = base_ratio_pred[:, anchor_model_index]
        final_ratio_pred = anchor_alpha * dgs_ratio_pred + (1.0 - anchor_alpha) * anchor_ratio_pred
    else:
        anchor_ratio_pred = np.full_like(dgs_ratio_pred, np.nan, dtype=float)
        final_ratio_pred = dgs_ratio_pred

    y_pred = capacity_from_ratio_new_model(final_ratio_pred, X_model, model_package)

    return y_pred, X_model, {
        "dgs_ratio_pred": dgs_ratio_pred,
        "anchor_ratio_pred": anchor_ratio_pred,
        "final_ratio_pred": final_ratio_pred,
        "weights": weights,
        "base_ratio_pred": base_ratio_pred,
    }


def build_conformal_interval(y_pred, X_model, model_package):
    """Use conformal_q saved in the model package if available."""
    q = model_package.get("conformal_q", None)
    if q is None:
        return None, None

    q = float(q)
    y_pred = np.asarray(y_pred, dtype=float)
    lower = y_pred - q
    upper = y_pred + q

    if bool(model_package.get("use_non_negative_constraint", True)):
        lower = np.maximum(lower, 0.0)

    strain_col = model_package.get("strain_col", DEFAULT_STRAIN_COL)
    strain = pd.to_numeric(X_model[strain_col], errors="coerce").to_numpy(dtype=float)
    zero_mask = np.isclose(strain, 0.0, atol=1e-12, rtol=0.0)
    lower[zero_mask] = 0.0
    upper[zero_mask] = 0.0

    return lower, upper


# ============================================================
# 4. Engineering calculation utilities
# ============================================================

def calculate_amplification_factor(h_over_l):
    h_l_tolerance = 0.01
    if abs(h_over_l - 13 / 68) < h_l_tolerance:
        return 1.084
    if abs(h_over_l - 25 / 75) < h_l_tolerance:
        return 1.247
    if abs(h_over_l - 25 / 125) < h_l_tolerance:
        return 1.070
    if abs(h_over_l - 50 / 150) < h_l_tolerance:
        return 1.179
    if abs(h_over_l - 55 / 200) < h_l_tolerance:
        return 1.142
    if abs(h_over_l) < 1e-12:
        return 1.000
    return 1.000


def build_raw_curve_input(
    strain_values,
    D,
    t,
    L,
    h,
    l,
    helix_angle,
    fcuAc,
    fyAs,
    concrete_type_value,
    e_over_r,
    confinement_factor,
    model_package,
):
    raw_feature_cols = list(model_package.get("raw_feature_cols", DEFAULT_RAW_FEATURE_COLS))

    D_over_t = D / t
    L_over_D = L / D
    h_over_l = h / l if abs(l) > 1e-12 else 0.0

    base_values = {
        "D/t": D_over_t,
        "D": D,
        "t": t,
        "L/D": L_over_D,
        "Helix Angle": helix_angle,
        "fcuAc": fcuAc,
        "fyAs": fyAs,
        "h/l": h_over_l,
        "Concrete Types": concrete_type_value,
        "e/r": e_over_r,
        "Confinement Factor": confinement_factor,
    }

    rows = []
    for strain in strain_values:
        row = {}
        for col in raw_feature_cols:
            if col == model_package.get("strain_col", DEFAULT_STRAIN_COL) or col == "Strain":
                row[col] = strain
            elif col in base_values:
                row[col] = base_values[col]
            else:
                row[col] = np.nan
        rows.append(row)

    return pd.DataFrame(rows, columns=raw_feature_cols)


def apply_optional_zero_constraint(strain_values, predictions, tolerance=1e-12):
    adjusted = np.asarray(predictions, dtype=float).copy()
    zero_mask = np.isclose(np.asarray(strain_values, dtype=float), 0.0, atol=tolerance, rtol=0.0)
    adjusted[zero_mask] = 0.0
    return adjusted, zero_mask


def safe_savgol(y, window_size, poly_order):
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return y

    # Savitzky-Golay requires odd window and window <= len(y).
    window_size = int(window_size)
    if window_size % 2 == 0:
        window_size += 1

    if window_size > len(y):
        window_size = len(y) if len(y) % 2 == 1 else len(y) - 1

    if window_size < 3:
        return y

    poly_order = min(int(poly_order), window_size - 1)
    return savgol_filter(y, window_size, poly_order)


# ============================================================
# 5. Streamlit layout
# ============================================================

st.set_page_config(
    page_title=f"{APP_TITLE} | {ALGORITHM_TITLE}",
    page_icon="🏗️",
    layout="wide",
)

st.markdown(
    f"""
    <h1 style="font-size: 28px; text-align: center; color: #2E86AB;
    border-bottom: 3px solid #2E86AB; margin-bottom: 1.8rem; padding-bottom: 10px;">
    🏗️ {APP_TITLE}
    </h1>
    <h3 style="text-align: center; color: #555; margin-top: -1.2rem;">
    {ALGORITHM_TITLE}: Multi-Physics-Guided Dynamic Gated Stacking with CatBoost Anchor
    </h3>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    .stMainBlockContainer {
        padding-top: 28px;
        padding-bottom: 28px;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        border-left: 4px solid #2E86AB;
    }
    .upload-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 2px dashed #2E86AB;
        margin-bottom: 20px;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .constraint-box {
        background-color: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .model-box {
        background-color: #eef6ff;
        border: 1px solid #9bc8f2;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 6. Load new model package
# ============================================================

st.markdown('<div class="upload-container">', unsafe_allow_html=True)
st.markdown("### 📁 Model Package")
st.info("Model package source: Google Drive auto-download is enabled.")

model_package = None

local_exists = MODEL_PACKAGE_PATH.exists() and MODEL_PACKAGE_PATH.stat().st_size > 1024
auto_download_available = bool(MODEL_PACKAGE_GDRIVE_ID or globals().get("MODEL_PACKAGE_GDRIVE_URL", ""))
default_manual_upload = (not local_exists) and (not auto_download_available)

st.caption(
    "The app will first try to use the local model package. "
    "If it is not available, it will automatically download the package from Google Drive."
)

use_manual_upload = st.checkbox(
    "Use manual model package upload",
    value=default_manual_upload,
    help="Only enable this if automatic Google Drive download fails or you want to upload another model package.",
)

if use_manual_upload:
    uploaded_package = st.file_uploader(
        "Upload MultiPhysics-DGS-CA model package",
        type=["pkl"],
        help="Please upload ea_pgcc_dgs_ca_model_package.pkl",
    )
    if uploaded_package is None:
        st.warning("⚠️ Please upload ea_pgcc_dgs_ca_model_package.pkl before prediction.")
        st.stop()

    with st.spinner("Loading uploaded model package..."):
        model_package = load_model_package_from_upload(uploaded_package)

else:
    if not local_exists:
        gdrive_source = globals().get("MODEL_PACKAGE_GDRIVE_URL", "") or MODEL_PACKAGE_GDRIVE_ID
        if gdrive_source:
            ok = download_from_gdrive(gdrive_source, MODEL_PACKAGE_PATH)
            if not ok:
                st.error("❌ Could not download model package. Please use manual upload.")
                st.stop()
        else:
            st.error(
                f"❌ Model package not found at {MODEL_PACKAGE_PATH}. "
                "Either place the package there or enable manual upload."
            )
            st.stop()

    with st.spinner("Loading local model package..."):
        model_package = load_model_package_from_path(MODEL_PACKAGE_PATH)

st.markdown("</div>", unsafe_allow_html=True)

if model_package is None:
    st.error("❌ Model package loading failed.")
    st.stop()

st.success("✅ MultiPhysics-DGS-CA model package loaded successfully!")

algorithm_name = model_package.get("algorithm_name", ALGORITHM_TITLE)
algorithm_full_name = model_package.get("algorithm_full_name", "")
raw_feature_cols = list(model_package.get("raw_feature_cols", DEFAULT_RAW_FEATURE_COLS))
model_feature_cols = list(model_package.get("model_feature_cols", []))
base_model_names = list(model_package.get("base_model_names", []))

st.markdown(
    f"""
    <div class="model-box">
    <strong>🤖 Loaded model:</strong> {algorithm_name}<br>
    <strong>Full name:</strong> {algorithm_full_name}<br>
    <strong>Raw input features:</strong> {", ".join(raw_feature_cols)}<br>
    <strong>Prediction formula:</strong> Nu_pred = BoundaryFactor × N_phy × R_final
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("🔍 View model package details", expanded=False):
    st.write("**Raw feature columns**")
    st.write(raw_feature_cols)
    st.write("**Model feature columns**")
    st.write(model_feature_cols)
    st.write("**Base model names**")
    st.write(base_model_names)
    st.write("**Anchor formula**")
    st.write(model_package.get("anchor_formula", "Not available"))
    st.write("**Physics parameters**")
    st.json({
        "alpha": model_package.get("physics_alpha"),
        "beta": model_package.get("physics_beta"),
        "gamma": model_package.get("physics_gamma"),
        "boundary_mode": model_package.get("boundary_mode"),
        "boundary_eps0": model_package.get("boundary_eps0"),
        "conformal_q": model_package.get("conformal_q"),
    })


# ============================================================
# 7. Parameter inputs
# ============================================================

st.markdown("### 📊 Model Parameter Input")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🔧 Geometric parameters**")
    D = st.number_input("D - Outer diameter (mm)", value=315.0, min_value=1.0, format="%.3f")
    t = st.number_input("t - Wall thickness (mm)", value=2.0, min_value=0.01, format="%.3f")
    L = st.number_input("L - Length (mm)", value=750.0, min_value=1.0, format="%.3f")
    h = st.number_input("h - Corrugation height (mm)", value=25.0, min_value=0.0, format="%.3f")
    l = st.number_input("l - Corrugation spacing (mm)", value=75.0, min_value=0.01, format="%.3f")

with col2:
    st.markdown("**🧱 Material and concrete type**")
    fcu = st.number_input("fcu - Concrete cube compressive strength (MPa)", value=40.0, min_value=0.01, format="%.3f")
    fy = st.number_input("fy - Steel yield strength (MPa)", value=235.0, min_value=0.01, format="%.3f")
    K = st.number_input("K - Wave number", value=10.0, min_value=0.0, format="%.3f")

    concrete_mapping = (model_package.get("categorical_mappings", {}) or {}).get("Concrete Types", None)
    if concrete_mapping is not None:
        concrete_options = list(concrete_mapping.keys())
        if not concrete_options:
            concrete_type_value = "Missing"
            st.warning("Concrete Types mapping is empty; using Missing.")
        else:
            default_index = 0
            for i, opt in enumerate(concrete_options):
                if str(opt).strip() in {"0", "0.0"}:
                    default_index = i
                    break
            concrete_type_value = st.selectbox(
                "Concrete Types",
                options=concrete_options,
                index=default_index,
                help="Categories are read from the saved training mapping.",
            )
    else:
        concrete_type_value = st.number_input(
            "Concrete Types / rubber replacement ratio",
            value=0.0,
            min_value=0.0,
            max_value=1.0,
            format="%.3f",
            help="The saved model package treats Concrete Types as numeric.",
        )

with col3:
    st.markdown("**📈 Eccentricity and strain range**")

    eccentricity_mode = st.radio(
        "Eccentricity input mode",
        ["Input e/r directly", "Input eccentricity e (mm)"],
        horizontal=False,
    )

    if eccentricity_mode == "Input e/r directly":
        e_over_r = st.number_input("e/r - Eccentricity ratio", value=0.20, min_value=0.0, format="%.4f")
        e_mm = e_over_r * (D / 2.0)
    else:
        e_mm = st.number_input("e - Eccentricity (mm)", value=30.0, min_value=0.0, format="%.3f")
        e_over_r = e_mm / (D / 2.0) if D > 0 else 0.0

    start_strain = st.number_input("Start strain", value=0.000, format="%.6f")
    end_strain = st.number_input("End strain", value=0.030, format="%.6f")
    N_points = st.number_input("Number of curve points", value=101, min_value=10, max_value=1000, step=1)

    include_conformal = st.checkbox(
        "Show conformal prediction interval if available",
        value=model_package.get("conformal_q", None) is not None,
    )


# ============================================================
# 8. Validation and derived parameters
# ============================================================

if start_strain >= end_strain:
    st.error("❌ Start strain must be less than end strain.")
    st.stop()

if t >= D / 2.0:
    st.error("❌ Wall thickness cannot be greater than or equal to half the diameter.")
    st.stop()

if any(val <= 0 for val in [D, t, L, fcu, fy]):
    st.error("❌ D, t, L, fcu and fy must be greater than 0.")
    st.stop()

st.markdown("### 🔧 Calculated Derived Parameters")

D_over_t = D / t
L_over_D = L / D
h_over_l = h / l if abs(l) > 1e-12 else 0.0

if K != 0 and l != 0:
    B = l * K
    helix_angle = math.degrees(math.atan(B / (math.pi * D)))
else:
    helix_angle = 0.0

amplification_factor = calculate_amplification_factor(h_over_l)

if amplification_factor == 1.0 and h > 0:
    st.markdown(
        f"""
        <div class="warning-box">
        ⚠️ <strong>Warning:</strong> h/l = {h_over_l:.3f} is outside the preset corrugation ranges.
        The steel-area amplification factor is set to 1.0.
        </div>
        """,
        unsafe_allow_html=True,
    )

area_method = st.radio(
    "Concrete core area calculation",
    [
        "Database-compatible: Ac = π(D - t)² / 4",
        "Physical inner diameter: Ac = π(D - 2t)² / 4",
    ],
    index=0,
    horizontal=True,
    help="Use the same formula as the training database whenever possible.",
)

if area_method.startswith("Database-compatible"):
    Ac = math.pi * (D - t) ** 2 / 4.0
else:
    Ac = math.pi * max(D - 2.0 * t, 0.0) ** 2 / 4.0

As = math.pi * D * t * amplification_factor

# MPa = N/mm², so fcuAc and fyAs are in N. The model divides by 1000 internally to obtain kN scale.
fcuAc = fcu * Ac
fyAs = fy * As

steel_ratio = As / Ac if Ac > 1e-12 else 0.0
confinement_factor = steel_ratio * fy / fcu if fcu > 1e-12 else 0.0

param_col1, param_col2, param_col3, param_col4 = st.columns(4)

with param_col1:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("D/t", f"{D_over_t:.3f}")
    st.metric("L/D", f"{L_over_D:.3f}")
    st.metric("Helix angle", f"{helix_angle:.2f}°")
    st.markdown("</div>", unsafe_allow_html=True)

with param_col2:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("Ac", f"{Ac:.0f} mm²")
    st.metric("As", f"{As:.0f} mm²")
    st.metric("Amplification factor", f"{amplification_factor:.3f}")
    st.markdown("</div>", unsafe_allow_html=True)

with param_col3:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("fcuAc", f"{fcuAc:.0f} N")
    st.metric("fyAs", f"{fyAs:.0f} N")
    st.metric("Confinement factor", f"{confinement_factor:.3f}")
    st.markdown("</div>", unsafe_allow_html=True)

with param_col4:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("h/l", f"{h_over_l:.3f}")
    st.metric("e", f"{e_mm:.2f} mm")
    st.metric("e/r", f"{e_over_r:.4f}")
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# 9. Prediction
# ============================================================

st.markdown("### 📈 Prediction Results")

strain_values = np.linspace(float(start_strain), float(end_strain), int(N_points))

df_raw_input = build_raw_curve_input(
    strain_values=strain_values,
    D=D,
    t=t,
    L=L,
    h=h,
    l=l,
    helix_angle=helix_angle,
    fcuAc=fcuAc,
    fyAs=fyAs,
    concrete_type_value=concrete_type_value,
    e_over_r=e_over_r,
    confinement_factor=confinement_factor,
    model_package=model_package,
)

with st.expander("🔍 View model input data and feature construction", expanded=False):
    st.write("**Raw input to model package**")
    st.dataframe(df_raw_input.head(10), use_container_width=True)
    st.write("**Required raw feature columns from model package**")
    st.write(raw_feature_cols)

try:
    with st.spinner("Predicting load–strain curve using MultiPhysics-DGS-CA..."):
        Nu_predicted_raw, X_model, aux = predict_capacity_with_new_model(df_raw_input, model_package)

        # The new model already applies BoundaryFactor and zero-strain constraint internally.
        # This extra checkbox is only a defensive option for GUI output.
        apply_extra_zero_constraint = st.checkbox(
            "Apply extra GUI zero-strain constraint",
            value=True,
            help="The new model already uses BoundaryFactor; this is an additional output-level safeguard.",
        )
        constraint_tolerance = st.number_input(
            "Zero-strain tolerance",
            value=1e-12,
            format="%.2e",
            disabled=not apply_extra_zero_constraint,
        )

        if apply_extra_zero_constraint:
            Nu_predicted, constraint_applied = apply_optional_zero_constraint(
                strain_values,
                Nu_predicted_raw,
                tolerance=constraint_tolerance,
            )
        else:
            Nu_predicted = Nu_predicted_raw.copy()
            constraint_applied = np.zeros_like(strain_values, dtype=bool)

        y_lower, y_upper = build_conformal_interval(Nu_predicted, X_model, model_package)
        if y_lower is None or y_upper is None:
            include_conformal = False

    st.success(f"✅ Prediction completed. Generated {len(Nu_predicted)} curve points.")

except Exception as exc:
    st.error(f"❌ Prediction failed: {exc}")
    st.write("**Debugging information**")
    st.write("Raw input shape:", df_raw_input.shape)
    st.write("Raw input columns:", list(df_raw_input.columns))
    st.write("Model raw_feature_cols:", raw_feature_cols)
    st.write("Model model_feature_cols:", model_feature_cols)
    st.stop()


# ============================================================
# 10. Result table and plot
# ============================================================

base_names = list(model_package.get("base_model_names", []))
weights = aux["weights"]
base_ratio_pred = aux["base_ratio_pred"]

result_df = pd.DataFrame({
    "Strain": strain_values,
    "Nu_pred_raw": Nu_predicted_raw,
    "Nu_pred": Nu_predicted,
    "Constraint_Applied": constraint_applied,
    "R_final": aux["final_ratio_pred"],
    "R_DGS": aux["dgs_ratio_pred"],
    "R_CatBoost_anchor": aux["anchor_ratio_pred"],
    "N0_Enhanced": X_model[model_package.get("n0_enhanced_col", "N0_Enhanced")].to_numpy(dtype=float),
    "PhysicsReduction": X_model[model_package.get("physics_reduction_col", "PhysicsReduction")].to_numpy(dtype=float),
    "N_phy": X_model[model_package.get("physics_col", "N_phy")].to_numpy(dtype=float),
    "BoundaryFactor": X_model[model_package.get("boundary_col", "BoundaryFactor")].to_numpy(dtype=float),
    "D/t": D_over_t,
    "L/D": L_over_D,
    "h/l": h_over_l,
    "e/r": e_over_r,
    "Confinement Factor": confinement_factor,
})

if include_conformal and y_lower is not None and y_upper is not None:
    result_df["Nu_lower_conformal"] = y_lower
    result_df["Nu_upper_conformal"] = y_upper
    result_df["Conformal_interval_width"] = y_upper - y_lower

for j, name in enumerate(base_names):
    if j < weights.shape[1]:
        result_df[f"{name}_weight"] = weights[:, j]
    if j < base_ratio_pred.shape[1]:
        result_df[f"{name}_ratio_pred"] = base_ratio_pred[:, j]
        result_df[f"{name}_capacity_pred"] = capacity_from_ratio_new_model(
            base_ratio_pred[:, j],
            X_model,
            model_package,
        )

result_col1, result_col2 = st.columns([1, 2])

with result_col1:
    st.markdown("**📋 Prediction Data**")
    st.dataframe(result_df, use_container_width=True, height=450)

    st.markdown("**📊 Statistical Information**")
    st.write(f"- Maximum capacity: {np.max(Nu_predicted):.2f} kN")
    st.write(f"- Minimum capacity: {np.min(Nu_predicted):.2f} kN")
    st.write(f"- Average capacity: {np.mean(Nu_predicted):.2f} kN")
    st.write(f"- Capacity range: {np.max(Nu_predicted) - np.min(Nu_predicted):.2f} kN")
    st.write(f"- Final ratio mean: {np.mean(aux['final_ratio_pred']):.4f}")
    st.write(f"- Physics baseline peak: {np.max(result_df['BoundaryFactor'] * result_df['N_phy']):.2f} kN")

    csv = result_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="📥 Download Prediction Data (CSV)",
        data=csv,
        file_name=f"multiphy_dgs_ca_curve_prediction_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

with result_col2:
    st.markdown("**📊 Load–Strain Curve**")

    smooth_col1, smooth_col2, smooth_col3 = st.columns(3)

    with smooth_col1:
        apply_smoothing = st.checkbox("Apply curve smoothing", value=True)
        window_size = st.number_input(
            "Window size",
            value=9,
            min_value=3,
            max_value=101,
            step=2,
            disabled=not apply_smoothing,
        )

    with smooth_col2:
        poly_order = st.number_input(
            "Polynomial order",
            value=2,
            min_value=1,
            max_value=5,
            disabled=not apply_smoothing,
        )
        show_original = st.checkbox("Show unsmoothed curve", value=True)

    with smooth_col3:
        show_physics_baseline = st.checkbox("Show physics baseline", value=False)
        show_conformal_band = st.checkbox(
            "Show conformal band",
            value=include_conformal,
            disabled=not include_conformal,
        )

    if apply_smoothing:
        try:
            smoothed_Nu = safe_savgol(Nu_predicted, window_size, poly_order)
            smoothed_Nu = np.maximum(smoothed_Nu, 0.0)
            zero_mask = np.isclose(strain_values, 0.0, atol=1e-12, rtol=0.0)
            smoothed_Nu[zero_mask] = 0.0
        except Exception as exc:
            st.warning(f"Smoothing failed: {exc}")
            smoothed_Nu = Nu_predicted
    else:
        smoothed_Nu = Nu_predicted

    fig, ax = plt.subplots(figsize=(12, 7.5))

    if show_conformal_band and y_lower is not None and y_upper is not None:
        ax.fill_between(
            strain_values,
            y_lower,
            y_upper,
            alpha=0.18,
            color="#4C78A8",
            label="Conformal interval",
        )

    if show_physics_baseline:
        physics_baseline = result_df["BoundaryFactor"].to_numpy() * result_df["N_phy"].to_numpy()
        ax.plot(
            strain_values,
            physics_baseline,
            linestyle=":",
            linewidth=2.0,
            color="gray",
            label="Physics baseline",
        )

    if show_original and apply_smoothing:
        ax.plot(
            strain_values,
            Nu_predicted,
            linestyle="--",
            linewidth=1.6,
            alpha=0.70,
            color="#8AB6D6",
            label="Original prediction",
        )

    ax.plot(
        strain_values,
        smoothed_Nu,
        linewidth=2.7,
        color="#C23B22",
        label="Smoothed prediction" if apply_smoothing else "Prediction",
    )

    peak_idx = int(np.nanargmax(smoothed_Nu))
    peak_strain = strain_values[peak_idx]
    peak_Nu = smoothed_Nu[peak_idx]

    ax.scatter(
        peak_strain,
        peak_Nu,
        color="black",
        s=110,
        zorder=5,
        label="Peak point",
        edgecolors="white",
        linewidth=1.5,
    )

    offset_x = max((end_strain - start_strain) * 0.12, 1e-6)
    y_span = max(np.max(smoothed_Nu) - np.min(smoothed_Nu), 1.0)
    offset_y = 0.10 * y_span

    ax.annotate(
        f"Peak: {peak_Nu:.2f} kN\nStrain: {peak_strain:.6f}",
        xy=(peak_strain, peak_Nu),
        xytext=(min(peak_strain + offset_x, end_strain), peak_Nu + offset_y),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.3),
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.92, edgecolor="black"),
    )

    ax.set_xlabel("Strain", fontsize=14, fontweight="bold")
    ax.set_ylabel(r"Predicted load $N_u$ (kN)", fontsize=14, fontweight="bold")
    ax.set_title(
        f"{ALGORITHM_TITLE} Load–Strain Curve Prediction",
        fontsize=16,
        fontweight="bold",
        pad=18,
    )

    ax.grid(True, alpha=0.28, linestyle="--")
    ax.legend(fontsize=11, frameon=True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_linewidth(1.4)

    y_min = min(0.0, float(np.min(smoothed_Nu)))
    y_max = float(np.max(smoothed_Nu))
    y_margin = max((y_max - y_min) * 0.10, 1.0)
    ax.set_ylim(y_min - 0.02 * y_margin, y_max + y_margin)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)


# ============================================================
# 11. Summary
# ============================================================

st.markdown("### 🎯 Prediction Summary")

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

with summary_col1:
    st.success(f"🏆 **Peak load**\n\n{peak_Nu:.2f} kN")

with summary_col2:
    st.info(f"📍 **Peak strain**\n\n{peak_strain:.6f}")

with summary_col3:
    ultimate_idx = int(len(strain_values) * 0.90)
    ultimate_idx = min(max(ultimate_idx, 0), len(strain_values) - 1)
    ultimate_Nu = smoothed_Nu[ultimate_idx]
    st.warning(f"📈 **90% curve-point load**\n\n{ultimate_Nu:.2f} kN")

with summary_col4:
    st.metric("e/r", f"{e_over_r:.4f}")
    st.metric("Confinement factor", f"{confinement_factor:.3f}")

st.markdown("### 🧩 Model Internal Outputs")

internal_col1, internal_col2, internal_col3 = st.columns(3)

with internal_col1:
    st.metric("Mean R_final", f"{np.mean(aux['final_ratio_pred']):.4f}")
    st.metric("Peak R_final", f"{aux['final_ratio_pred'][peak_idx]:.4f}")

with internal_col2:
    st.metric("Mean N_phy", f"{np.mean(result_df['N_phy']):.2f} kN")
    st.metric("Peak N_phy", f"{result_df['N_phy'].iloc[peak_idx]:.2f} kN")

with internal_col3:
    st.metric("Anchor alpha", f"{float(model_package.get('anchor_alpha', 1.0)):.4f}")
    if len(base_names) >= 3 and weights.shape[1] >= 3:
        st.metric("Peak weights", f"xgb={weights[peak_idx,0]:.2f}, lgbm={weights[peak_idx,1]:.2f}, cat={weights[peak_idx,2]:.2f}")

st.markdown(
    """
    ---
    **Note.** `N0_Enhanced`, `PhysicsReduction`, `N_phy`, and `BoundaryFactor`
    are internal physics-guided features reconstructed by the app from the
    saved model package. They are not manual GUI inputs.
    """
)
