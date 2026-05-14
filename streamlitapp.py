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
# 5. Streamlit layout: redesigned paper-style dashboard
# ============================================================

st.set_page_config(
    page_title=f"{APP_TITLE} | {ALGORITHM_TITLE}",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --primary: #1F5EFF;
        --primary-dark: #123A8C;
        --accent: #13A8A8;
        --danger: #C23B22;
        --bg-soft: #F6F8FC;
        --card-bg: #FFFFFF;
        --text-main: #1B2430;
        --text-muted: #5D6778;
        --border: #E1E7F0;
    }

    .block-container {
        padding-top: 1.35rem;
        padding-bottom: 2.5rem;
        max-width: 1500px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F7FAFF 0%, #FFFFFF 100%);
        border-right: 1px solid #E6ECF5;
    }

    .hero {
        padding: 1.25rem 1.45rem 1.15rem 1.45rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #0B1F4D 0%, #153E8A 48%, #0EA5A4 100%);
        color: white;
        box-shadow: 0 16px 35px rgba(15, 39, 95, 0.18);
        margin-bottom: 1.0rem;
    }

    .hero-title {
        font-size: 2.05rem;
        font-weight: 760;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        color: rgba(255, 255, 255, 0.86);
        font-size: 1.02rem;
        line-height: 1.55;
        margin-bottom: 0.85rem;
    }

    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 0.35rem;
    }

    .pill {
        display: inline-block;
        padding: 0.28rem 0.72rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.13);
        border: 1px solid rgba(255, 255, 255, 0.20);
        font-size: 0.82rem;
        color: rgba(255, 255, 255, 0.93);
    }

    .panel {
        border: 1px solid var(--border);
        background: var(--card-bg);
        border-radius: 18px;
        padding: 1.05rem 1.15rem;
        box-shadow: 0 10px 25px rgba(20, 37, 63, 0.055);
        margin-bottom: 1.0rem;
    }

    .panel-title {
        font-size: 1.03rem;
        font-weight: 720;
        color: var(--text-main);
        margin-bottom: 0.35rem;
    }

    .panel-caption {
        color: var(--text-muted);
        font-size: 0.88rem;
        line-height: 1.42;
    }

    .mini-card {
        border: 1px solid var(--border);
        background: linear-gradient(180deg, #FFFFFF 0%, #F9FBFF 100%);
        border-radius: 16px;
        padding: 0.82rem 0.95rem;
        min-height: 95px;
    }

    .mini-label {
        color: var(--text-muted);
        font-size: 0.78rem;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .mini-value {
        color: var(--text-main);
        font-size: 1.38rem;
        font-weight: 780;
        margin-top: 0.22rem;
    }

    .mini-unit {
        color: var(--text-muted);
        font-size: 0.82rem;
        margin-left: 0.15rem;
    }

    .formula-box {
        border-radius: 16px;
        padding: 0.85rem 1.0rem;
        background: #F2F7FF;
        border: 1px solid #D9E7FF;
        color: #173A73;
        font-size: 0.92rem;
        line-height: 1.55;
        margin-top: 0.55rem;
    }

    .status-ok {
        border-radius: 14px;
        padding: 0.72rem 0.85rem;
        background: #ECFDF3;
        border: 1px solid #BFECCF;
        color: #116A35;
        font-weight: 650;
        margin-bottom: 0.75rem;
    }

    .status-warn {
        border-radius: 14px;
        padding: 0.72rem 0.85rem;
        background: #FFF8E7;
        border: 1px solid #FFE1A3;
        color: #8A5A00;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }

    .footer-note {
        color: var(--text-muted);
        font-size: 0.86rem;
        line-height: 1.45;
        padding-top: 0.65rem;
        border-top: 1px solid var(--border);
        margin-top: 0.85rem;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E1E7F0;
        padding: 0.75rem 0.85rem;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(20, 37, 63, 0.045);
    }

    div[data-testid="stTabs"] button {
        font-weight: 650;
    }

    .stDownloadButton button {
        border-radius: 12px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 6. Header and model loading
# ============================================================

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-title">🏗️ {APP_TITLE}</div>
        <div class="hero-subtitle">
            A compact research dashboard for eccentric load–strain curve prediction of
            concrete-filled steel tubular members using <b>{ALGORITHM_TITLE}</b>.
        </div>
        <div class="pill-row">
            <span class="pill">Physics-guided features</span>
            <span class="pill">Dynamic gated stacking</span>
            <span class="pill">CatBoost anchor</span>
            <span class="pill">Full curve prediction</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.caption("Set inputs once, then inspect curve, data, and model diagnostics in the main dashboard.")

    st.markdown("### 📦 Model source")
    local_exists = MODEL_PACKAGE_PATH.exists() and MODEL_PACKAGE_PATH.stat().st_size > 1024
    auto_download_available = bool(MODEL_PACKAGE_GDRIVE_ID or globals().get("MODEL_PACKAGE_GDRIVE_URL", ""))
    default_manual_upload = (not local_exists) and (not auto_download_available)

    use_manual_upload = st.toggle(
        "Manual model upload",
        value=default_manual_upload,
        help="Disable this to use the Google Drive auto-download model package.",
    )

model_package = None

if use_manual_upload:
    with st.sidebar:
        uploaded_package = st.file_uploader(
            "Upload ea_pgcc_dgs_ca_model_package.pkl",
            type=["pkl"],
        )
    if uploaded_package is None:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Model package required</div>
                <div class="panel-caption">
                    Please upload <code>ea_pgcc_dgs_ca_model_package.pkl</code> in the sidebar,
                    or disable manual upload to use Google Drive auto-download.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()
    with st.spinner("Loading uploaded model package..."):
        model_package = load_model_package_from_upload(uploaded_package)
else:
    if not local_exists:
        gdrive_source = globals().get("MODEL_PACKAGE_GDRIVE_URL", "") or MODEL_PACKAGE_GDRIVE_ID
        if gdrive_source:
            ok = download_from_gdrive(gdrive_source, MODEL_PACKAGE_PATH)
            if not ok:
                st.error("❌ Could not download model package. Please enable manual upload in the sidebar.")
                st.stop()
        else:
            st.error(
                f"❌ Model package not found at {MODEL_PACKAGE_PATH}. "
                "Please enable manual upload or configure MODEL_PACKAGE_GDRIVE_ID."
            )
            st.stop()

    with st.spinner("Loading local model package..."):
        model_package = load_model_package_from_path(MODEL_PACKAGE_PATH)

if model_package is None:
    st.error("❌ Model package loading failed.")
    st.stop()

algorithm_name = model_package.get("algorithm_name", ALGORITHM_TITLE)
algorithm_full_name = model_package.get("algorithm_full_name", "")
raw_feature_cols = list(model_package.get("raw_feature_cols", DEFAULT_RAW_FEATURE_COLS))
model_feature_cols = list(model_package.get("model_feature_cols", []))
base_model_names = list(model_package.get("base_model_names", []))

st.markdown(
    f"""
    <div class="status-ok">
        ✅ Model loaded: {algorithm_name}
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 7. Sidebar inputs
# ============================================================

with st.sidebar:
    st.markdown("### ① Geometry")
    D = st.number_input("Outer diameter D (mm)", value=315.0, min_value=1.0, format="%.3f")
    t = st.number_input("Wall thickness t (mm)", value=2.0, min_value=0.01, format="%.3f")
    L = st.number_input("Length L (mm)", value=750.0, min_value=1.0, format="%.3f")
    h = st.number_input("Corrugation height h (mm)", value=25.0, min_value=0.0, format="%.3f")
    l = st.number_input("Corrugation spacing l (mm)", value=75.0, min_value=0.01, format="%.3f")

    st.markdown("### ② Materials")
    fcu = st.number_input("Concrete strength fcu (MPa)", value=40.0, min_value=0.01, format="%.3f")
    fy = st.number_input("Steel yield strength fy (MPa)", value=235.0, min_value=0.01, format="%.3f")
    K = st.number_input("Wave number K", value=10.0, min_value=0.0, format="%.3f")

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
                "Concrete type",
                options=concrete_options,
                index=default_index,
                help="Options are taken from the saved training categorical mapping.",
            )
    else:
        concrete_type_value = st.number_input(
            "Concrete type / rubber ratio",
            value=0.0,
            min_value=0.0,
            max_value=1.0,
            format="%.3f",
        )

    st.markdown("### ③ Loading path")
    eccentricity_mode = st.radio(
        "Eccentricity input",
        ["e/r", "e in mm"],
        horizontal=True,
    )
    if eccentricity_mode == "e/r":
        e_over_r = st.number_input("Eccentricity ratio e/r", value=0.20, min_value=0.0, format="%.4f")
        e_mm = e_over_r * (D / 2.0)
    else:
        e_mm = st.number_input("Eccentricity e (mm)", value=30.0, min_value=0.0, format="%.3f")
        e_over_r = e_mm / (D / 2.0) if D > 0 else 0.0

    st.markdown("### ④ Curve settings")
    start_strain = st.number_input("Start strain", value=0.000, format="%.6f")
    end_strain = st.number_input("End strain", value=0.030, format="%.6f")
    N_points = st.slider("Curve points", min_value=20, max_value=500, value=101, step=1)

    include_conformal = st.toggle(
        "Show conformal interval",
        value=model_package.get("conformal_q", None) is not None,
        disabled=model_package.get("conformal_q", None) is None,
    )
    apply_extra_zero_constraint = st.toggle(
        "Force zero load at zero strain",
        value=True,
        help="The model already applies BoundaryFactor; this is an extra GUI-level safeguard.",
    )

# ============================================================
# 8. Validation and derived parameters
# ============================================================

if start_strain >= end_strain:
    st.error("❌ Start strain must be smaller than end strain.")
    st.stop()

if t >= D / 2.0:
    st.error("❌ Wall thickness cannot be greater than or equal to half the diameter.")
    st.stop()

if any(val <= 0 for val in [D, t, L, fcu, fy]):
    st.error("❌ D, t, L, fcu and fy must be greater than 0.")
    st.stop()

D_over_t = D / t
L_over_D = L / D
h_over_l = h / l if abs(l) > 1e-12 else 0.0

if K != 0 and l != 0:
    B = l * K
    helix_angle = math.degrees(math.atan(B / (math.pi * D)))
else:
    helix_angle = 0.0

amplification_factor = calculate_amplification_factor(h_over_l)

area_method = st.sidebar.radio(
    "Concrete core area",
    [
        "Database-compatible: Ac = π(D - t)² / 4",
        "Physical inner diameter: Ac = π(D - 2t)² / 4",
    ],
    index=0,
    help="Use the database-compatible option if you want consistency with the training data.",
)

if area_method.startswith("Database-compatible"):
    Ac = math.pi * (D - t) ** 2 / 4.0
else:
    Ac = math.pi * max(D - 2.0 * t, 0.0) ** 2 / 4.0

As = math.pi * D * t * amplification_factor
fcuAc = fcu * Ac
fyAs = fy * As
steel_ratio = As / Ac if Ac > 1e-12 else 0.0
confinement_factor = steel_ratio * fy / fcu if fcu > 1e-12 else 0.0

if amplification_factor == 1.0 and h > 0:
    st.markdown(
        f"""
        <div class="status-warn">
        ⚠️ h/l = {h_over_l:.3f} is outside the preset corrugation ranges.
        The steel-area amplification factor is set to 1.0.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# 9. Prediction
# ============================================================

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

try:
    with st.spinner("Running MultiPhysics-DGS-CA inference..."):
        Nu_predicted_raw, X_model, aux = predict_capacity_with_new_model(df_raw_input, model_package)

        if apply_extra_zero_constraint:
            Nu_predicted, constraint_applied = apply_optional_zero_constraint(
                strain_values,
                Nu_predicted_raw,
                tolerance=1e-12,
            )
        else:
            Nu_predicted = Nu_predicted_raw.copy()
            constraint_applied = np.zeros_like(strain_values, dtype=bool)

        y_lower, y_upper = build_conformal_interval(Nu_predicted, X_model, model_package)
        if y_lower is None or y_upper is None:
            include_conformal = False

except Exception as exc:
    st.error(f"❌ Prediction failed: {exc}")
    with st.expander("Debugging information", expanded=True):
        st.write("Raw input shape:", df_raw_input.shape)
        st.write("Raw input columns:", list(df_raw_input.columns))
        st.write("Model raw_feature_cols:", raw_feature_cols)
        st.write("Model model_feature_cols:", model_feature_cols)
    st.stop()

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

# Smoothing controls in main page
controls_l, controls_m, controls_r = st.columns([1, 1, 1])
with controls_l:
    apply_smoothing = st.toggle("Smooth curve", value=True)
with controls_m:
    window_size = st.slider("Smoothing window", 3, 101, 9, step=2, disabled=not apply_smoothing)
with controls_r:
    show_internal_curves = st.toggle("Show model internals", value=False)

if apply_smoothing:
    try:
        smoothed_Nu = safe_savgol(Nu_predicted, window_size, 2)
        smoothed_Nu = np.maximum(smoothed_Nu, 0.0)
        smoothed_Nu[np.isclose(strain_values, 0.0, atol=1e-12, rtol=0.0)] = 0.0
    except Exception:
        smoothed_Nu = Nu_predicted
else:
    smoothed_Nu = Nu_predicted

peak_idx = int(np.nanargmax(smoothed_Nu))
peak_strain = strain_values[peak_idx]
peak_Nu = smoothed_Nu[peak_idx]
physics_baseline = result_df["BoundaryFactor"].to_numpy() * result_df["N_phy"].to_numpy()

# ============================================================
# 10. Dashboard cards
# ============================================================

card_cols = st.columns(5)
card_data = [
    ("Peak load", f"{peak_Nu:.2f}", "kN"),
    ("Peak strain", f"{peak_strain:.6f}", ""),
    ("e/r", f"{e_over_r:.4f}", ""),
    ("Confinement", f"{confinement_factor:.3f}", ""),
    ("Mean R_final", f"{np.mean(aux['final_ratio_pred']):.4f}", ""),
]

for col, (label, value, unit) in zip(card_cols, card_data):
    with col:
        st.markdown(
            f"""
            <div class="mini-card">
                <div class="mini-label">{label}</div>
                <div class="mini-value">{value}<span class="mini-unit">{unit}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

tab_curve, tab_parameters, tab_data, tab_diagnostics, tab_model = st.tabs(
    ["📈 Prediction curve", "🧾 Parameters", "📋 Data table", "🧩 Diagnostics", "⚙️ Model"]
)

with tab_curve:
    left, right = st.columns([2.2, 1.0])

    with left:
        fig, ax = plt.subplots(figsize=(11.8, 6.8))

        if include_conformal and y_lower is not None and y_upper is not None:
            ax.fill_between(
                strain_values,
                y_lower,
                y_upper,
                alpha=0.16,
                color="#1F5EFF",
                label="Conformal interval",
            )

        ax.plot(
            strain_values,
            Nu_predicted,
            linestyle="--",
            linewidth=1.4,
            alpha=0.62,
            color="#7FA8D8",
            label="Raw prediction",
        )

        ax.plot(
            strain_values,
            smoothed_Nu,
            linewidth=2.8,
            color="#C23B22",
            label="Smoothed prediction" if apply_smoothing else "Prediction",
        )

        if show_internal_curves:
            ax.plot(
                strain_values,
                physics_baseline,
                linestyle=":",
                linewidth=2.0,
                color="#5F6773",
                label="Physics baseline",
            )

        ax.scatter(
            peak_strain,
            peak_Nu,
            color="#0B1F4D",
            s=115,
            zorder=5,
            edgecolors="white",
            linewidth=1.5,
            label="Peak",
        )

        y_span = max(np.nanmax(smoothed_Nu) - np.nanmin(smoothed_Nu), 1.0)
        ax.annotate(
            f"Peak = {peak_Nu:.2f} kN\nε = {peak_strain:.6f}",
            xy=(peak_strain, peak_Nu),
            xytext=(min(peak_strain + 0.14 * (end_strain - start_strain), end_strain), peak_Nu + 0.11 * y_span),
            arrowprops=dict(arrowstyle="->", color="#0B1F4D", lw=1.35),
            fontsize=10.8,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#0B1F4D", alpha=0.96),
        )

        ax.set_xlabel("Strain", fontsize=13, fontweight="bold")
        ax.set_ylabel(r"Predicted load $N_u$ (kN)", fontsize=13, fontweight="bold")
        ax.set_title(f"{ALGORITHM_TITLE} load–strain curve", fontsize=15, fontweight="bold", pad=15)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend(fontsize=10.5, frameon=True, loc="best")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)

        y_min = min(0.0, float(np.nanmin(smoothed_Nu)))
        y_max = float(np.nanmax(smoothed_Nu))
        y_margin = max((y_max - y_min) * 0.10, 1.0)
        ax.set_ylim(y_min - 0.02 * y_margin, y_max + y_margin)

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with right:
        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-title">Prediction summary</div>
                <div class="panel-caption">
                    The reported peak is calculated from the displayed curve.
                </div>
                <div class="formula-box">
                    <b>Nu_pred</b> = BoundaryFactor × N_phy × R_final<br>
                    <b>R_final</b> = α_anchor R_DGS + (1 − α_anchor) R_CatBoost
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.metric("Maximum capacity", f"{np.max(Nu_predicted):.2f} kN")
        st.metric("Average capacity", f"{np.mean(Nu_predicted):.2f} kN")
        st.metric("Physics baseline peak", f"{np.max(physics_baseline):.2f} kN")
        st.metric("Anchor alpha", f"{float(model_package.get('anchor_alpha', 1.0)):.4f}")

with tab_parameters:
    st.markdown('<div class="panel"><div class="panel-title">Derived input parameters</div></div>', unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.metric("D/t", f"{D_over_t:.3f}")
        st.metric("L/D", f"{L_over_D:.3f}")
        st.metric("Helix angle", f"{helix_angle:.2f}°")
    with p2:
        st.metric("Ac", f"{Ac:.0f} mm²")
        st.metric("As", f"{As:.0f} mm²")
        st.metric("Amplification factor", f"{amplification_factor:.3f}")
    with p3:
        st.metric("fcuAc", f"{fcuAc:.0f} N")
        st.metric("fyAs", f"{fyAs:.0f} N")
        st.metric("Steel ratio", f"{steel_ratio:.4f}")
    with p4:
        st.metric("h/l", f"{h_over_l:.3f}")
        st.metric("e", f"{e_mm:.2f} mm")
        st.metric("Confinement factor", f"{confinement_factor:.3f}")

    with st.expander("View raw model input", expanded=False):
        st.dataframe(df_raw_input.head(20), use_container_width=True)
        st.write("Required raw feature columns:", raw_feature_cols)

with tab_data:
    st.dataframe(result_df, use_container_width=True, height=520)

    csv = result_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="📥 Download prediction data as CSV",
        data=csv,
        file_name=f"multiphy_dgs_ca_curve_prediction_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with tab_diagnostics:
    d1, d2 = st.columns([1.05, 1.0])

    with d1:
        st.markdown('<div class="panel"><div class="panel-title">Dynamic gating weights</div></div>', unsafe_allow_html=True)
        if len(base_names) > 0 and weights.ndim == 2:
            fig_w, ax_w = plt.subplots(figsize=(8.4, 4.6))
            for j, name in enumerate(base_names):
                if j < weights.shape[1]:
                    ax_w.plot(strain_values, weights[:, j], linewidth=2.0, label=name)
            ax_w.set_xlabel("Strain", fontsize=12)
            ax_w.set_ylabel("Gating weight", fontsize=12)
            ax_w.set_ylim(0, 1.05)
            ax_w.grid(True, alpha=0.25, linestyle="--")
            ax_w.legend(frameon=False, ncol=min(3, len(base_names)))
            fig_w.tight_layout()
            st.pyplot(fig_w, use_container_width=True)
        else:
            st.info("No gating weights available.")

    with d2:
        st.markdown('<div class="panel"><div class="panel-title">Correction ratios</div></div>', unsafe_allow_html=True)
        fig_r, ax_r = plt.subplots(figsize=(8.4, 4.6))
        ax_r.plot(strain_values, aux["final_ratio_pred"], linewidth=2.4, label="R_final", color="#C23B22")
        ax_r.plot(strain_values, aux["dgs_ratio_pred"], linewidth=1.8, linestyle="--", label="R_DGS", color="#1F5EFF")
        if np.isfinite(aux["anchor_ratio_pred"]).any():
            ax_r.plot(strain_values, aux["anchor_ratio_pred"], linewidth=1.8, linestyle=":", label="R_CatBoost", color="#13A8A8")
        ax_r.set_xlabel("Strain", fontsize=12)
        ax_r.set_ylabel("Correction ratio", fontsize=12)
        ax_r.grid(True, alpha=0.25, linestyle="--")
        ax_r.legend(frameon=False)
        fig_r.tight_layout()
        st.pyplot(fig_r, use_container_width=True)

    st.markdown('<div class="panel"><div class="panel-title">Point diagnostics at peak</div></div>', unsafe_allow_html=True)
    diag_cols = st.columns(4)
    with diag_cols[0]:
        st.metric("Peak R_final", f"{aux['final_ratio_pred'][peak_idx]:.4f}")
    with diag_cols[1]:
        st.metric("Peak N_phy", f"{result_df['N_phy'].iloc[peak_idx]:.2f} kN")
    with diag_cols[2]:
        st.metric("Peak BoundaryFactor", f"{result_df['BoundaryFactor'].iloc[peak_idx]:.3f}")
    with diag_cols[3]:
        if len(base_names) >= 3 and weights.shape[1] >= 3:
            st.metric("Peak weights", f"{weights[peak_idx,0]:.2f} / {weights[peak_idx,1]:.2f} / {weights[peak_idx,2]:.2f}")
        else:
            st.metric("Peak index", str(peak_idx))

with tab_model:
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">Model package</div>
            <div class="panel-caption">
                <b>Loaded model:</b> {algorithm_name}<br>
                <b>Full name:</b> {algorithm_full_name}<br>
                <b>Formula:</b> Nu_pred = BoundaryFactor × N_phy × R_final
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2 = st.columns(2)
    with m1:
        st.write("**Raw feature columns**")
        st.write(raw_feature_cols)
        st.write("**Base model names**")
        st.write(base_model_names)
        st.write("**Anchor formula**")
        st.write(model_package.get("anchor_formula", "Not available"))
    with m2:
        st.write("**Model feature columns**")
        st.write(model_feature_cols)
        st.write("**Physics parameters**")
        st.json({
            "alpha": model_package.get("physics_alpha"),
            "beta": model_package.get("physics_beta"),
            "gamma": model_package.get("physics_gamma"),
            "boundary_mode": model_package.get("boundary_mode"),
            "boundary_eps0": model_package.get("boundary_eps0"),
            "conformal_q": model_package.get("conformal_q"),
        })

st.markdown(
    """
    <div class="footer-note">
        <b>Note.</b> N0_Enhanced, PhysicsReduction, N_phy, and BoundaryFactor are
        reconstructed internally from the saved model package. They are not manual GUI inputs.
        Keep all input units consistent with the training database.
    </div>
    """,
    unsafe_allow_html=True,
)
