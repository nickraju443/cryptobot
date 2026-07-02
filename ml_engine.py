"""
ML Engine - XGBoost GPU-accelerated trade prediction
Learns from every trade, predicts win probability, detects market regime.
Trains in background threads to never block the trading bot.
"""

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODEL_FILE = Path(__file__).parent / "ml_model.ubj"
FEATURES_FILE = Path(__file__).parent / "ml_features.json"
TRADE_HISTORY_FILE = Path(__file__).parent / "trade_history.json"

# Cached model in memory
_model = None
_model_lock = threading.Lock()
_model_loaded = False
_training_in_progress = False

# Isotonic calibration curve — {"x": [...], "y": [...]} sorted by x
_calibration = None
CALIBRATION_FILE = Path(__file__).parent / "ml_calibration.json"


def _load_calibration():
    """Load isotonic calibration curve if available."""
    global _calibration
    if not CALIBRATION_FILE.exists():
        _calibration = None
        return
    try:
        with open(CALIBRATION_FILE, "r") as f:
            _calibration = json.load(f)
    except Exception:
        _calibration = None


def _calibrate(raw_prob: float) -> float:
    """Apply isotonic calibration curve to a raw XGBoost probability.
    Returns raw value if no calibration is loaded."""
    if not _calibration:
        return raw_prob
    try:
        xs = _calibration["x"]
        ys = _calibration["y"]
        return float(np.interp(raw_prob, xs, ys))
    except Exception:
        return raw_prob

# Market regime cache
_regime_cache = {"data": None, "expires": datetime.min}
_REGIME_CACHE_SECONDS = 300  # 5 minutes


# ==========================================
# FEATURE EXTRACTION
# ==========================================

# Ordered list of numerical features the model expects
NUMERICAL_FEATURES = [
    "rsi", "macd", "macd_signal", "macd_hist", "macd_hist_prev",
    "stoch_k", "stoch_d", "adx", "bb_pctb", "cmf", "rvol",
    "price_vs_vwap_pct", "price_vs_sma20_pct", "price_vs_ema9_pct",
    "confidence", "scalp_score", "risk_reward",
    "consecutive_candle_color", "body_atr_ratio",
    "upper_shadow_ratio", "lower_shadow_ratio",
    "sr_distance_pct", "open_position_count", "session_pnl_at_entry",
    "day_of_week", "minutes_since_open",
    "ticker_win_rate", "ticker_streak", "52w_range_pct",
    "vix_level",
]

# Categorical features -> one-hot encoded
TREND_VALUES = ["UPTREND", "DOWNTREND", "CONSOLIDATION", "UNKNOWN"]
REGIME_VALUES = ["LOW_VOL", "NORMAL", "HIGH_VOL", "CRISIS"]
SIDE_VALUES = []  # Removed — judge on setup quality, not long vs short bias
WEEKLY_TREND_VALUES = ["BULLISH", "BEARISH", "MIXED", "N/A"]
ICHIMOKU_VALUES = ["ABOVE_CLOUD", "BELOW_CLOUD", "IN_CLOUD", "UNKNOWN"]
SECTOR_VALUES = ["MAJOR", "L1", "L2", "DEFI", "AI", "MEME", "EXCH", "DEPIN", "GAME", "ORACLE", "PRIVACY", "LEGACY", "ALT"]

# Ticker encoding removed — model judges purely on setup quality, not ticker history
TICKER_VALUES = []

# Common candle pattern names (presence flags)
PATTERN_NAMES = [
    "Bullish Engulfing", "Bearish Engulfing", "Hammer", "Shooting Star",
    "Morning Star", "Evening Star", "Doji", "Dragonfly Doji", "Gravestone Doji",
    "Three White Soldiers", "Three Black Crows", "Bullish Marubozu", "Bearish Marubozu",
    "Piercing Line", "Dark Cloud Cover", "Bullish Harami", "Bearish Harami",
    "Tweezer Bottom", "Tweezer Top", "Hanging Man", "Inverted Hammer",
    "Three Inside Up", "Three Inside Down", "Spinning Top",
]

ALL_FEATURE_NAMES = (
    NUMERICAL_FEATURES
    + [f"trend_{v}" for v in TREND_VALUES]
    + [f"regime_{v}" for v in REGIME_VALUES]
    + [f"side_{v}" for v in SIDE_VALUES]
    + [f"weekly_{v}" for v in WEEKLY_TREND_VALUES]
    + [f"ichimoku_{v}" for v in ICHIMOKU_VALUES]
    + [f"pattern_{p}" for p in PATTERN_NAMES]
    + [f"sector_{v}" for v in SECTOR_VALUES]
    + [f"ticker_{v}" for v in TICKER_VALUES]
    + ["mtf_aligned"]
)


def extract_features(record: dict) -> list:
    """Convert a trade record into a flat feature vector for XGBoost."""
    features = []

    # Numerical features
    for feat in NUMERICAL_FEATURES:
        val = record.get(feat)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            features.append(np.nan)
        else:
            try:
                features.append(float(val))
            except (TypeError, ValueError):
                features.append(np.nan)

    # One-hot: trend
    trend = record.get("trend", "UNKNOWN")
    for v in TREND_VALUES:
        features.append(1.0 if trend == v else 0.0)

    # One-hot: regime
    regime = record.get("market_regime", "NORMAL")
    for v in REGIME_VALUES:
        features.append(1.0 if regime == v else 0.0)

    # One-hot: side
    side = record.get("side", "long")
    for v in SIDE_VALUES:
        features.append(1.0 if side == v else 0.0)

    # One-hot: weekly trend
    wt = record.get("weekly_trend", "N/A")
    for v in WEEKLY_TREND_VALUES:
        features.append(1.0 if wt == v else 0.0)

    # One-hot: ichimoku position
    ichi = record.get("ichimoku_position", "UNKNOWN")
    for v in ICHIMOKU_VALUES:
        features.append(1.0 if ichi == v else 0.0)

    # Pattern presence flags
    patterns = record.get("pattern_names", [])
    for p in PATTERN_NAMES:
        features.append(1.0 if p in patterns else 0.0)

    # One-hot: asset class (crypto sector analog)
    sector = str(record.get("ticker_sector", "ALT"))
    for v in SECTOR_VALUES:
        features.append(1.0 if sector == v else 0.0)

    # Ticker one-hot disabled — model learns from setup features, not from
    # symbol identity. Loop is kept for feature-vector length stability.
    for v in TICKER_VALUES:
        features.append(0.0)

    # MTF aligned
    mtf = record.get("mtf_aligned")
    features.append(1.0 if mtf else 0.0)

    return features


# ==========================================
# MARKET REGIME DETECTION
# ==========================================

def get_market_regime() -> dict:
    """Detect crypto market regime from BTC realized volatility and BTC trend.
    BTC plays the role that VIX+SPY play in equities. Cached for 5 minutes."""
    global _regime_cache
    if datetime.now() < _regime_cache["expires"] and _regime_cache["data"]:
        return _regime_cache["data"]

    # `vix` field kept as the volatility-percent proxy so ML feature names stay stable
    result = {
        "vix": 3.0,
        "regime": "NORMAL",
        "spy_trend": "NEUTRAL",
        "spy_above_sma50": True,
        "market_bias": "NEUTRAL",
    }

    try:
        from analysis import fetch_data, calc_ema, calc_sma, calc_atr
        btc_df = fetch_data("BTC/USDT", period="3mo", interval="1d")
        if len(btc_df) >= 50:
            btc_df["EMA_9"] = calc_ema(btc_df["Close"], 9)
            btc_df["EMA_21"] = calc_ema(btc_df["Close"], 21)
            btc_df["SMA_50"] = calc_sma(btc_df["Close"], 50)
            btc_df["ATR"] = calc_atr(btc_df)
            last = btc_df.iloc[-1]
            # ATR as % of price = realized vol proxy
            atr_pct = (last["ATR"] / last["Close"]) * 100 if last["Close"] > 0 else 3.0
            result["vix"] = round(atr_pct, 2)
            # Crypto vol thresholds (BTC ATR%): <2 quiet, 2-4 normal, 4-7 high, >7 crisis
            if atr_pct < 2.0:
                result["regime"] = "LOW_VOL"
            elif atr_pct <= 4.0:
                result["regime"] = "NORMAL"
            elif atr_pct <= 7.0:
                result["regime"] = "HIGH_VOL"
            else:
                result["regime"] = "CRISIS"
            # BTC trend
            result["spy_above_sma50"] = bool(last["Close"] > last["SMA_50"])
            if last["EMA_9"] > last["EMA_21"]:
                result["spy_trend"] = "BULLISH"
            elif last["EMA_9"] < last["EMA_21"]:
                result["spy_trend"] = "BEARISH"
            # Market bias same direction as BTC trend when above SMA50
            if result["spy_above_sma50"] and result["spy_trend"] == "BULLISH":
                result["market_bias"] = "BULLISH"
            elif (not result["spy_above_sma50"]) and result["spy_trend"] == "BEARISH":
                result["market_bias"] = "BEARISH"
    except Exception as e:
        logger.warning(f"Regime detection failed: {e}")

    _regime_cache = {"data": result, "expires": datetime.now() + timedelta(seconds=_REGIME_CACHE_SECONDS)}
    return result


# ==========================================
# XGBOOST MODEL — TRAINING
# ==========================================

def _load_trade_history() -> list:
    """Load trade history from disk."""
    if TRADE_HISTORY_FILE.exists():
        try:
            with open(TRADE_HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def train_model(force=False) -> dict:
    """Train XGBoost model on trade history. Returns training stats."""
    global _training_in_progress
    if _training_in_progress and not force:
        return {"status": "already_training"}
    _training_in_progress = True

    try:
        import xgboost as xgb
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import roc_auc_score

        history = _load_trade_history()

        # Use all trades with features — real and paper both feed the model
        # Use all trades with ML features
        ml_trades = [t for t in history if t.get("rsi") is not None]

        if len(ml_trades) < 100:
            return {"status": "insufficient_data", "trades": len(ml_trades), "needed": 100}

        # Sort by timestamp so temporal walk-forward CV is honest (no future leak)
        ml_trades.sort(key=lambda t: t.get("exit_time") or t.get("entry_time") or "")

        # Extract features and labels
        X_raw = []
        y = []
        for t in ml_trades:
            features = extract_features(t)
            X_raw.append(features)
            y.append(1 if t.get("won", False) else 0)

        X = np.array(X_raw, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        # Equal weight for all trades
        weights = np.ones(len(ml_trades), dtype=np.float32)

        # Regularization based on data size
        if len(ml_trades) < 200:
            # High regularization for small datasets
            params = {
                "max_depth": 3,
                "learning_rate": 0.05,
                "n_estimators": 50,
                "reg_alpha": 0.5,
                "reg_lambda": 2.0,
                "min_child_weight": 5,
            }
        else:
            params = {
                "max_depth": 4,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "min_child_weight": 3,
            }

        # Try GPU first, fall back to CPU
        try:
            model = xgb.XGBClassifier(
                device="cuda",
                tree_method="hist",
                objective="binary:logistic",
                eval_metric="auc",
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                verbosity=0,
                **params,
            )
            model.fit(X, y, sample_weight=weights)
        except Exception:
            # GPU not available, use CPU
            model = xgb.XGBClassifier(
                tree_method="hist",
                objective="binary:logistic",
                eval_metric="auc",
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                verbosity=0,
                **params,
            )
            model.fit(X, y, sample_weight=weights)

        # Walk-forward (temporal) CV — honest for time series, no future leak
        from sklearn.model_selection import TimeSeriesSplit
        cv_model = xgb.XGBClassifier(
            tree_method="hist",
            objective="binary:logistic",
            eval_metric="auc",
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            verbosity=0,
            **params,
        )
        n_splits = max(2, min(5, len(ml_trades) // 40))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = cross_val_score(cv_model, X, y, cv=tscv, scoring="accuracy")
        cv_mean = float(np.mean(cv_scores))
        try:
            cv_auc_scores = cross_val_score(cv_model, X, y, cv=tscv, scoring="roc_auc")
            cv_auc = float(np.mean(cv_auc_scores))
        except Exception:
            cv_auc = 0.0

        # Isotonic calibration — makes raw XGBoost scores trustworthy probabilities.
        # Fit on earliest 80% of trades (temporally), validate/calibrate on latest 20%.
        try:
            from sklearn.isotonic import IsotonicRegression
            split_idx = int(len(ml_trades) * 0.8)
            if split_idx >= 50 and (len(ml_trades) - split_idx) >= 20:
                cal_model = xgb.XGBClassifier(
                    tree_method="hist",
                    objective="binary:logistic",
                    eval_metric="auc",
                    subsample=0.8,
                    colsample_bytree=0.8,
                    use_label_encoder=False,
                    verbosity=0,
                    **params,
                )
                cal_model.fit(X[:split_idx], y[:split_idx])
                raw_probs = cal_model.predict_proba(X[split_idx:])[:, 1]
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(raw_probs, y[split_idx:])
                # Persist calibration as piecewise (x, y) arrays — no extra dependency at inference
                cal_points = {
                    "x": [float(v) for v in iso.X_thresholds_.tolist()],
                    "y": [float(v) for v in iso.y_thresholds_.tolist()],
                }
                with open(Path(__file__).parent / "ml_calibration.json", "w") as f:
                    json.dump(cal_points, f)
                logger.info(f"ML: Isotonic calibration fit on {len(ml_trades)-split_idx} holdout trades")
            else:
                # Not enough data for a clean holdout — clear any stale calibration
                cal_path = Path(__file__).parent / "ml_calibration.json"
                if cal_path.exists():
                    cal_path.unlink()
        except Exception as e:
            logger.warning(f"ML: Calibration step failed (non-fatal): {e}")

        # Check if new model is better than old
        old_cv = 0.0
        if FEATURES_FILE.exists():
            try:
                with open(FEATURES_FILE, "r") as f:
                    old_info = json.load(f)
                    old_cv = old_info.get("cv_accuracy", 0)
            except Exception:
                pass

        # Always accept new model — it needs to learn from latest market data
        logger.info(f"ML: New model cv={cv_mean:.3f} (old was {old_cv:.3f}) — accepting")

        # Save model
        model.save_model(str(MODEL_FILE))

        # Feature importance
        importance = model.feature_importances_
        feature_ranking = sorted(
            zip(ALL_FEATURE_NAMES, importance.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        # Training accuracy
        preds = model.predict_proba(X)[:, 1]
        try:
            auc = roc_auc_score(y, preds)
        except Exception:
            auc = 0.0

        info = {
            "trained_at": datetime.now().isoformat(),
            "trades_used": len(ml_trades),
            "cv_accuracy": round(cv_mean, 4),
            "cv_auc_walkforward": round(cv_auc, 4),
            "cv_method": "TimeSeriesSplit",
            "cv_splits": n_splits,
            "train_auc": round(auc, 4),
            "win_rate_in_data": round(float(np.mean(y)), 4),
            "top_features": feature_ranking[:15],
            "feature_count": len(ALL_FEATURE_NAMES),
            "calibration": "isotonic" if (Path(__file__).parent / "ml_calibration.json").exists() else "none",
        }
        with open(FEATURES_FILE, "w") as f:
            json.dump(info, f, indent=2)

        # Reload model in memory
        _reload_model()

        logger.info(f"ML: Trained on {len(ml_trades)} trades | CV accuracy: {cv_mean:.3f} | AUC: {auc:.3f}")
        logger.info(f"ML: Top features: {', '.join(f[0] for f in feature_ranking[:5])}")

        return {
            "status": "trained",
            "trades": len(ml_trades),
            "cv_accuracy": cv_mean,
            "auc": auc,
            "top_features": feature_ranking[:10],
        }

    except Exception as e:
        logger.error(f"ML training failed: {e}")
        return {"status": "error", "error": str(e)}
    finally:
        _training_in_progress = False


# ==========================================
# PREDICTION
# ==========================================

def _reload_model():
    """Reload model from disk into memory."""
    global _model, _model_loaded
    with _model_lock:
        if MODEL_FILE.exists():
            try:
                import xgboost as xgb
                _model = xgb.XGBClassifier()
                _model.load_model(str(MODEL_FILE))
                _model_loaded = True
                _load_calibration()
                cal_note = " + isotonic calibration" if _calibration else ""
                logger.info(f"ML: Model loaded into memory{cal_note}")
                print(f"  ML MODEL: Loaded and active (XGBoost){cal_note}")
            except Exception as e:
                logger.warning(f"ML: Failed to load model: {e}")
                _model = None
                _model_loaded = False
        else:
            _model = None
            _model_loaded = False


_last_model_check = 0.0

def predict_win_probability(features: dict) -> float:
    """Predict win probability for a potential trade.
    Returns 0.0-1.0. Returns 0.5 (neutral) if no model exists.

    Primary path: XGBoost + LightGBM ensemble with joint isotonic calibration
    (ml_ensemble.py). Falls back to the single-model XGBoost path if the
    ensemble is unavailable, and finally to 0.5 if neither is loaded."""
    # Try ensemble first (smarter brain)
    try:
        from ml_ensemble import predict_win_probability_ensemble
        prob = predict_win_probability_ensemble(features)
        if prob is not None and 0.0 <= prob <= 1.0 and prob != 0.5:
            return float(prob)
    except Exception as e:
        logger.debug(f"Ensemble predict unavailable, using fallback: {e}")

    # Fallback: single-model path
    global _model, _model_loaded, _last_model_check
    import time as _time

    now = _time.time()
    if not _model_loaded and (now - _last_model_check) > 60:
        _last_model_check = now
        _reload_model()
    elif _model_loaded and (now - _last_model_check) > 300:
        _last_model_check = now
        _reload_model()

    with _model_lock:
        if _model is None:
            return 0.5

    try:
        feature_vector = extract_features(features)
        X = np.array([feature_vector], dtype=np.float32)
        with _model_lock:
            prob = _model.predict_proba(X)[0][1]
        return _calibrate(float(prob))
    except Exception as e:
        logger.warning(f"ML prediction failed: {e}")
        return 0.5


# ==========================================
# BACKGROUND TRAINING
# ==========================================

_trades_since_retrain = 0

def maybe_retrain(trade_count: int):
    """Called after every closed trade. Retrains model every 10 trades after 100 usable trades."""
    global _trades_since_retrain
    _trades_since_retrain += 1

    if trade_count < 100:
        return
    if _trades_since_retrain < 10:
        return
    if _training_in_progress:
        return

    _trades_since_retrain = 0

    def _train():
        logger.info(f"ML: Background retraining triggered ({trade_count} total trades)...")
        # Single-model (fallback path)
        result = train_model()
        logger.info(f"ML: Single-model retrain: {result.get('status')}")
        # Ensemble (primary path)
        try:
            from ml_ensemble import train_ensemble
            ens_result = train_ensemble()
            logger.info(f"ML: Ensemble retrain: {ens_result.get('status')} "
                        f"xgb_auc={ens_result.get('xgb_auc')} "
                        f"lgbm_auc={ens_result.get('lgbm_auc')} "
                        f"ensemble_auc={ens_result.get('ensemble_auc')}")
        except Exception as e:
            logger.warning(f"ML: Ensemble retrain failed: {e}")

    t = threading.Thread(target=_train, daemon=True)
    t.start()


# ==========================================
# FEATURE IMPORTANCE FEEDBACK
# ==========================================

def get_top_features() -> list:
    """Return top features the model uses, from saved feature importance."""
    if FEATURES_FILE.exists():
        try:
            with open(FEATURES_FILE, "r") as f:
                info = json.load(f)
            return info.get("top_features", [])
        except Exception:
            pass
    return []


def get_ml_status() -> dict:
    """Get current ML system status for logging/dashboard."""
    status = {
        "model_exists": MODEL_FILE.exists(),
        "training_in_progress": _training_in_progress,
        "model_loaded": _model_loaded,
    }
    if FEATURES_FILE.exists():
        try:
            with open(FEATURES_FILE, "r") as f:
                info = json.load(f)
            status["trained_at"] = info.get("trained_at")
            status["trades_used"] = info.get("trades_used", 0)
            status["cv_accuracy"] = info.get("cv_accuracy", 0)
            status["train_auc"] = info.get("train_auc", 0)
            status["top_features"] = info.get("top_features", [])[:5]
        except Exception:
            pass
    return status
