"""
ML Ensemble - XGBoost + LightGBM ensemble predictor.

Drop-in *alternative* to ml_engine.py's single-model predictor. This module
trains both XGBoost and LightGBM on the same trade history, averages their
probabilities, and applies a joint isotonic calibration fit on a held-out
temporal slice (oldest 80% -> predict newest 20% -> fit calibration).

Designed to live alongside ml_engine.py without colliding:
    - Feature extraction is imported from ml_engine (no duplication).
    - Writes its own model files (ml_xgb.ubj, ml_lgbm.txt) and its own
      calibration inside ml_ensemble_meta.json -- never touches ml_model.ubj
      or ml_calibration.json.

Public API:
    train_ensemble(force: bool = False) -> dict
    predict_win_probability_ensemble(features: dict) -> float
    get_ensemble_status() -> dict
"""

from __future__ import annotations

import json
import logging
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Silence harmless sklearn/LightGBM warning: LGBM fit with feature names but
# predicted with a raw numpy array. We feed a plain list-of-floats to the
# predict path on purpose (no pandas round-trip needed), so the warning is noise.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)

import numpy as np

import ml_engine  # for extract_features + ALL_FEATURE_NAMES + trade history loader

logger = logging.getLogger(__name__)

# ---- File locations (kept separate from the single-model files) -------------
XGB_MODEL_FILE = Path(__file__).parent / "ml_xgb.ubj"
LGBM_MODEL_FILE = Path(__file__).parent / "ml_lgbm.txt"
ENSEMBLE_META_FILE = Path(__file__).parent / "ml_ensemble_meta.json"
TRADE_HISTORY_FILE = Path(__file__).parent / "trade_history.json"

# ---- In-memory caches (thread-safe) ----------------------------------------
_lock = threading.Lock()
_xgb_model: Optional[Any] = None
_lgbm_model: Optional[Any] = None
_calibration: Optional[dict] = None  # {"x": [...], "y": [...]}
_models_loaded: bool = False
_last_reload_ts: float = 0.0
_RELOAD_SECONDS: int = 300  # 5-minute hot-reload window

_training_in_progress: bool = False

# ---- Minimum data requirement ----------------------------------------------
MIN_TRADES = 100


# =============================================================================
# TRAINING
# =============================================================================

def train_ensemble(force: bool = False) -> dict:
    """Train both XGBoost and LightGBM on the full trade history.

    Pipeline:
        1. Load trade history, filter to trades with `rsi is not None`.
        2. Sort by timestamp (walk-forward honesty -- no future leak).
        3. TimeSeriesSplit CV for per-model AUC (xgb_auc, lgbm_auc).
        4. Train final models on all data.
        5. For the joint ensemble calibration: fit fresh xgb+lgbm on oldest 80%,
           predict newest 20%, average probs, fit IsotonicRegression on that.
        6. Compute walk-forward ensemble AUC from the 20% holdout (uncalibrated
           avg -- calibration is monotonic so AUC is preserved).
        7. Persist ml_xgb.ubj, ml_lgbm.txt, ml_ensemble_meta.json.

    Args:
        force: If True, train even when another training call is in progress.

    Returns:
        dict with keys: status, trades, xgb_auc, lgbm_auc, ensemble_auc,
        calibration. `status` is one of: "trained", "insufficient_data",
        "already_training", "error".
    """
    global _training_in_progress

    if _training_in_progress and not force:
        return {"status": "already_training"}
    _training_in_progress = True

    try:
        logger.info("ML-Ensemble: training started")

        # Lazy imports so import-time failures surface here with context
        import xgboost as xgb
        import lightgbm as lgb
        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import TimeSeriesSplit

        # --- Load + filter history exactly like ml_engine does ---------------
        history = _load_trade_history()
        ml_trades = [t for t in history if t.get("rsi") is not None]

        if len(ml_trades) < MIN_TRADES:
            logger.info(
                "ML-Ensemble: insufficient_data (%d trades, need %d)",
                len(ml_trades), MIN_TRADES,
            )
            return {
                "status": "insufficient_data",
                "trades": len(ml_trades),
                "needed": MIN_TRADES,
            }

        # Walk-forward honesty: sort by timestamp.
        ml_trades.sort(
            key=lambda t: t.get("exit_time") or t.get("entry_time") or ""
        )

        # --- Build X, y ------------------------------------------------------
        X_raw: list[list[float]] = []
        y_raw: list[int] = []
        for t in ml_trades:
            X_raw.append(ml_engine.extract_features(t))
            y_raw.append(1 if t.get("won", False) else 0)

        X = np.array(X_raw, dtype=np.float32)
        y = np.array(y_raw, dtype=np.float32)
        n = len(ml_trades)

        # --- Parameters ------------------------------------------------------
        xgb_params = _xgb_params(n)
        lgbm_params = _lgbm_params()

        # --- Walk-forward per-model AUC (TimeSeriesSplit) --------------------
        n_splits = max(2, min(5, n // 40))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        xgb_cv_aucs: list[float] = []
        lgbm_cv_aucs: list[float] = []
        for train_idx, test_idx in tscv.split(X):
            # XGB fold
            try:
                fold_xgb = _build_xgb(xgb_params, use_gpu=True)
                fold_xgb.fit(X[train_idx], y[train_idx])
            except Exception as e:
                logger.warning("ML-Ensemble: xgb GPU fold failed (%s); falling back to CPU", e)
                fold_xgb = _build_xgb(xgb_params, use_gpu=False)
                fold_xgb.fit(X[train_idx], y[train_idx])
            try:
                xgb_probs = fold_xgb.predict_proba(X[test_idx])[:, 1]
                xgb_cv_aucs.append(float(roc_auc_score(y[test_idx], xgb_probs)))
            except Exception:
                pass

            # LGBM fold
            fold_lgbm = lgb.LGBMClassifier(**lgbm_params)
            fold_lgbm.fit(X[train_idx], y[train_idx])
            try:
                lgbm_probs = fold_lgbm.predict_proba(X[test_idx])[:, 1]
                lgbm_cv_aucs.append(float(roc_auc_score(y[test_idx], lgbm_probs)))
            except Exception:
                pass

        xgb_auc = float(np.mean(xgb_cv_aucs)) if xgb_cv_aucs else 0.0
        lgbm_auc = float(np.mean(lgbm_cv_aucs)) if lgbm_cv_aucs else 0.0

        # --- Calibration on the 80/20 temporal holdout -----------------------
        split_idx = int(n * 0.8)
        calibration_points: Optional[dict] = None
        ensemble_auc = 0.0

        if split_idx >= 50 and (n - split_idx) >= 20:
            try:
                cal_xgb = _build_xgb(xgb_params, use_gpu=True)
                try:
                    cal_xgb.fit(X[:split_idx], y[:split_idx])
                except Exception as e:
                    logger.warning("ML-Ensemble: GPU calibration xgb failed (%s); CPU", e)
                    cal_xgb = _build_xgb(xgb_params, use_gpu=False)
                    cal_xgb.fit(X[:split_idx], y[:split_idx])

                cal_lgbm = lgb.LGBMClassifier(**lgbm_params)
                cal_lgbm.fit(X[:split_idx], y[:split_idx])

                xgb_hold = cal_xgb.predict_proba(X[split_idx:])[:, 1]
                lgbm_hold = cal_lgbm.predict_proba(X[split_idx:])[:, 1]
                ensemble_hold = 0.5 * xgb_hold + 0.5 * lgbm_hold

                try:
                    ensemble_auc = float(roc_auc_score(y[split_idx:], ensemble_hold))
                except Exception:
                    ensemble_auc = 0.0

                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(ensemble_hold, y[split_idx:])
                calibration_points = {
                    "x": [float(v) for v in iso.X_thresholds_.tolist()],
                    "y": [float(v) for v in iso.y_thresholds_.tolist()],
                }
                logger.info(
                    "ML-Ensemble: joint isotonic calibration fit on %d holdout trades",
                    n - split_idx,
                )
            except Exception as e:
                logger.warning("ML-Ensemble: calibration step failed (non-fatal): %s", e)
                calibration_points = None
        else:
            logger.info(
                "ML-Ensemble: not enough data for holdout calibration "
                "(split_idx=%d, holdout=%d)", split_idx, n - split_idx,
            )

        # --- Final models trained on ALL data --------------------------------
        try:
            final_xgb = _build_xgb(xgb_params, use_gpu=True)
            final_xgb.fit(X, y)
        except Exception as e:
            logger.warning("ML-Ensemble: final GPU xgb failed (%s); CPU", e)
            final_xgb = _build_xgb(xgb_params, use_gpu=False)
            final_xgb.fit(X, y)

        final_lgbm = lgb.LGBMClassifier(**lgbm_params)
        final_lgbm.fit(X, y)

        # --- Persist ---------------------------------------------------------
        final_xgb.save_model(str(XGB_MODEL_FILE))
        final_lgbm.booster_.save_model(str(LGBM_MODEL_FILE))

        meta = {
            "trained_at": datetime.now().isoformat(),
            "trades": n,
            "feature_count": len(ml_engine.ALL_FEATURE_NAMES),
            "xgb_auc": round(xgb_auc, 4),
            "lgbm_auc": round(lgbm_auc, 4),
            "ensemble_auc": round(ensemble_auc, 4),
            "cv_splits": n_splits,
            "cv_method": "TimeSeriesSplit",
            "calibration": "isotonic" if calibration_points else "none",
            "calibration_points": calibration_points,  # inline -- no extra file
        }
        with open(ENSEMBLE_META_FILE, "w") as f:
            json.dump(meta, f, indent=2)

        # Reload in memory (picks up brand-new models + calibration)
        _reload_models(force=True)

        logger.info(
            "ML-Ensemble: training succeeded | trades=%d | xgb_auc=%.3f "
            "lgbm_auc=%.3f ensemble_auc=%.3f | calibration=%s",
            n, xgb_auc, lgbm_auc, ensemble_auc,
            "isotonic" if calibration_points else "none",
        )

        return {
            "status": "trained",
            "trades": n,
            "xgb_auc": round(xgb_auc, 4),
            "lgbm_auc": round(lgbm_auc, 4),
            "ensemble_auc": round(ensemble_auc, 4),
            "calibration": "isotonic" if calibration_points else "none",
        }

    except Exception as e:
        logger.error("ML-Ensemble: training error: %s", e, exc_info=True)
        return {"status": "error", "error": str(e)}
    finally:
        _training_in_progress = False


# =============================================================================
# PREDICTION
# =============================================================================

def predict_win_probability_ensemble(features: dict) -> float:
    """Predict win probability using the XGB+LGBM ensemble + isotonic calibration.

    Returns 0.5 (neutral) if either model is missing -- never gates the bot
    when the ensemble artifacts don't exist yet.

    Threadsafe. Reloads models from disk at most every 5 minutes so
    background retrains are picked up automatically.
    """
    _maybe_reload_models()

    with _lock:
        xgb_model = _xgb_model
        lgbm_model = _lgbm_model
        calibration = _calibration

    if xgb_model is None or lgbm_model is None:
        return 0.5

    try:
        vec = ml_engine.extract_features(features)
        X = np.array([vec], dtype=np.float32)

        xgb_prob = float(xgb_model.predict_proba(X)[0][1])
        # LightGBM Booster.predict returns 1d array of class-1 probabilities
        # for binary objective.
        lgbm_prob = float(lgbm_model.predict(X)[0])

        avg = 0.5 * xgb_prob + 0.5 * lgbm_prob
        return _apply_calibration(avg, calibration)
    except Exception as e:
        logger.warning("ML-Ensemble: prediction failed: %s", e)
        return 0.5


def get_ensemble_status() -> dict:
    """Return a status snapshot for dashboards / diagnostics."""
    status: dict[str, Any] = {
        "xgb_model_exists": XGB_MODEL_FILE.exists(),
        "lgbm_model_exists": LGBM_MODEL_FILE.exists(),
        "meta_exists": ENSEMBLE_META_FILE.exists(),
        "models_loaded": _models_loaded,
        "training_in_progress": _training_in_progress,
    }
    if ENSEMBLE_META_FILE.exists():
        try:
            with open(ENSEMBLE_META_FILE, "r") as f:
                meta = json.load(f)
            for k in (
                "trained_at", "trades", "feature_count",
                "xgb_auc", "lgbm_auc", "ensemble_auc",
                "cv_method", "cv_splits", "calibration",
            ):
                if k in meta:
                    status[k] = meta[k]
        except Exception as e:
            status["meta_read_error"] = str(e)
    return status


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _load_trade_history() -> list:
    """Load trade history from disk (same pattern as ml_engine)."""
    if TRADE_HISTORY_FILE.exists():
        try:
            with open(TRADE_HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _xgb_params(n_trades: int) -> dict:
    """Match ml_engine.train_model's XGBoost hyperparameters."""
    if n_trades < 200:
        return {
            "max_depth": 3,
            "learning_rate": 0.05,
            "n_estimators": 50,
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
            "min_child_weight": 5,
        }
    return {
        "max_depth": 4,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_weight": 3,
    }


def _lgbm_params() -> dict:
    """LightGBM defaults tuned for small tabular trading data."""
    return {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.5,
        "verbosity": -1,
        "objective": "binary",
    }


def _build_xgb(params: dict, use_gpu: bool = True):
    """Build an XGBClassifier, optionally GPU-accelerated."""
    import xgboost as xgb
    kwargs: dict[str, Any] = dict(
        tree_method="hist",
        objective="binary:logistic",
        eval_metric="auc",
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        verbosity=0,
    )
    if use_gpu:
        kwargs["device"] = "cuda"
    kwargs.update(params)
    return xgb.XGBClassifier(**kwargs)


def _apply_calibration(raw_prob: float, calibration: Optional[dict]) -> float:
    """Apply piecewise isotonic calibration to an ensemble probability."""
    if not calibration:
        return float(raw_prob)
    try:
        xs = calibration["x"]
        ys = calibration["y"]
        return float(np.interp(raw_prob, xs, ys))
    except Exception:
        return float(raw_prob)


def _maybe_reload_models() -> None:
    """Reload models if we haven't yet, or if 5 minutes elapsed since the
    last reload. Triggers immediate reload when no models are cached."""
    global _last_reload_ts
    now = time.time()
    with _lock:
        need = (
            not _models_loaded
            or (now - _last_reload_ts) > _RELOAD_SECONDS
        )
    if need:
        _reload_models()


def _reload_models(force: bool = False) -> None:
    """Reload XGB + LGBM + calibration from disk into memory.

    Called automatically on a 5-minute window from _maybe_reload_models,
    and explicitly right after training.
    """
    global _xgb_model, _lgbm_model, _calibration, _models_loaded, _last_reload_ts

    with _lock:
        _last_reload_ts = time.time()

        # XGBoost
        xgb_model = None
        if XGB_MODEL_FILE.exists():
            try:
                import xgboost as xgb
                xgb_model = xgb.XGBClassifier()
                xgb_model.load_model(str(XGB_MODEL_FILE))
            except Exception as e:
                logger.warning("ML-Ensemble: failed to load XGB model: %s", e)
                xgb_model = None

        # LightGBM
        lgbm_model = None
        if LGBM_MODEL_FILE.exists():
            try:
                import lightgbm as lgb
                lgbm_model = lgb.Booster(model_file=str(LGBM_MODEL_FILE))
            except Exception as e:
                logger.warning("ML-Ensemble: failed to load LGBM model: %s", e)
                lgbm_model = None

        # Calibration (read from meta -- no extra file)
        calibration = None
        if ENSEMBLE_META_FILE.exists():
            try:
                with open(ENSEMBLE_META_FILE, "r") as f:
                    meta = json.load(f)
                cp = meta.get("calibration_points")
                if cp and "x" in cp and "y" in cp:
                    calibration = cp
            except Exception as e:
                logger.warning("ML-Ensemble: failed to load calibration: %s", e)
                calibration = None

        _xgb_model = xgb_model
        _lgbm_model = lgbm_model
        _calibration = calibration
        _models_loaded = (xgb_model is not None and lgbm_model is not None)

        if _models_loaded:
            note = " + isotonic calibration" if calibration else ""
            logger.info("ML-Ensemble: models loaded into memory%s", note)
