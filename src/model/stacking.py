"""
Two-level stacking ensemble that combines CatBoost, XGBoost, and LightGBM.
"""

import logging
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score, mean_squared_error
import joblib

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import RANDOM_SEED, MODEL_DIR
from src.model.base_models import CatBoostWrapper, XGBoostWrapper, LightGBMWrapper

logger = logging.getLogger(__name__)


class StackedEnsemble:
    """
    Stacked ensemble model for match outcome prediction.
    
    Architecture:
    - Level 0: CatBoost, XGBoost, LightGBM
      Each produces: [P(Loss), P(Draw), P(Win)] + expected_goal_diff
      = 4 outputs per model x 3 models = 12 meta-features
    
    - Level 1: Logistic Regression (classification) + Ridge (regression)
      Learns optimal blending weights from OOF predictions
    
    Post-processing:
    - Platt scaling for probability calibration
    """

    def __init__(self, base_params: Optional[dict] = None):
        """
        Initialize the stacked ensemble.
        
        Args:
            base_params: Dict mapping model_name -> hyperparameters
                        (typically from Optuna tuning)
        """
        base_params = base_params or {}
        
        self.base_models = [
            CatBoostWrapper(params=base_params.get("catboost", {})),
            XGBoostWrapper(params=base_params.get("xgboost", {})),
            LightGBMWrapper(params=base_params.get("lightgbm", {})),
        ]
        
        # Meta-learners
        self.meta_clf = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=RANDOM_SEED,
        )
        self.meta_reg = Ridge(alpha=1.0, random_state=RANDOM_SEED)
        
        # Calibrated meta-classifier (Platt scaling)
        self.calibrated_meta = None
        
        self.is_fitted = False
        self.training_metrics: dict = {}
        self.use_stacked = True
        self.best_clf_model_name: Optional[str] = None
        self.best_reg_model_name: Optional[str] = None

    def _get_base_model(self, name: Optional[str]):
        """Return a fitted base wrapper by name, or None if unavailable."""
        if name is None:
            return None
        for model in self.base_models:
            if model.name == name:
                return model
        return None

    @staticmethod
    def _normalize_proba(proba: np.ndarray) -> np.ndarray:
        """Clamp and normalize probability rows for stable logloss."""
        proba = np.asarray(proba, dtype=float)
        proba = np.clip(proba, 1e-15, 1.0)
        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return proba / row_sums

    def fit(self, X: pd.DataFrame, y_wdl: pd.Series, y_gd: pd.Series, sample_weights: Optional[pd.Series] = None):
        """
        Train the full stacked ensemble.
        
        1. Generate OOF predictions from each base model
        2. Stack OOF predictions into meta-features
        3. Train meta-learner on meta-features
        4. Calibrate probabilities
        
        Args:
            X: Feature matrix
            y_wdl: Win/Draw/Loss target (0, 1, 2)
            y_gd: Goal differential target
            sample_weights: Exponential decay weights prioritizing recent matches
        """
        logger.info("=" * 60)
        logger.info("Training Stacked Ensemble")
        logger.info("=" * 60)
        
        # Step 1: Generate OOF predictions from each base model
        oof_probas = []
        oof_gds = []
        
        for model in self.base_models:
            logger.info(f"\nTraining {model.name}...")
            oof_proba, oof_gd = model.get_oof_predictions(X, y_wdl, y_gd, sample_weights=sample_weights)
            oof_probas.append(oof_proba)
            oof_gds.append(oof_gd)
            
            # Log base model performance
            # Only evaluate on non-zero predictions (skipping first fold's training data)
            mask = oof_proba.sum(axis=1) > 0
            if mask.any():
                normalized_oof = self._normalize_proba(oof_proba[mask])
                base_ll = log_loss(y_wdl[mask], normalized_oof, labels=[0, 1, 2])
                base_acc = accuracy_score(y_wdl[mask], normalized_oof.argmax(axis=1))
                base_mse = mean_squared_error(y_gd[mask], oof_gd[mask])
                logger.info(f"  {model.name} OOF - LogLoss: {base_ll:.4f}, "
                          f"Accuracy: {base_acc:.3f}, GD-MSE: {base_mse:.4f}")
                self.training_metrics[f"{model.name}_logloss"] = base_ll
                self.training_metrics[f"{model.name}_accuracy"] = base_acc
                self.training_metrics[f"{model.name}_gd_mse"] = base_mse

        # Step 2: Stack into meta-features
        # Shape: (n_samples, 12) = 3 models x (3 probs + 1 gd)
        meta_features = np.hstack(
            [proba for proba in oof_probas] + 
            [gd.reshape(-1, 1) for gd in oof_gds]
        )
        
        # Remove samples where all OOF predictions are zero (from first fold)
        valid_mask = meta_features.sum(axis=1) != 0
        meta_features_valid = meta_features[valid_mask]
        y_wdl_valid = y_wdl[valid_mask].reset_index(drop=True)
        y_gd_valid = y_gd[valid_mask].reset_index(drop=True)
        
        logger.info(f"\nMeta-features shape: {meta_features_valid.shape}")

        # Step 3: Train meta-learners
        logger.info("Training meta-learner (Logistic Regression)...")
        # Ensure valid_mask handles sample weights
        weights_valid = sample_weights[valid_mask].reset_index(drop=True) if sample_weights is not None else None
        
        fit_params = {}
        if weights_valid is not None:
            fit_params['sample_weight'] = weights_valid
            
        self.meta_clf.fit(meta_features_valid, y_wdl_valid, **fit_params)
        self.meta_reg.fit(meta_features_valid, y_gd_valid, **fit_params)
        
        # Step 4: Calibrate
        logger.info("Calibrating probabilities (Platt scaling)...")
        self.calibrated_meta = CalibratedClassifierCV(
            self.meta_clf, method="sigmoid", cv=3
        )
        self.calibrated_meta.fit(meta_features_valid, y_wdl_valid, **fit_params)

        # Evaluate stacked model
        meta_proba = self._normalize_proba(self.calibrated_meta.predict_proba(meta_features_valid))
        meta_pred = meta_proba.argmax(axis=1)
        meta_gd = self.meta_reg.predict(meta_features_valid)
        
        stack_ll = log_loss(y_wdl_valid, meta_proba, labels=[0, 1, 2])
        stack_acc = accuracy_score(y_wdl_valid, meta_pred)
        stack_mse = mean_squared_error(y_gd_valid, meta_gd)
        
        self.training_metrics["stacked_logloss"] = stack_ll
        self.training_metrics["stacked_accuracy"] = stack_acc
        self.training_metrics["stacked_gd_mse"] = stack_mse

        base_loglosses = {
            model.name: self.training_metrics[f"{model.name}_logloss"]
            for model in self.base_models
            if f"{model.name}_logloss" in self.training_metrics
        }
        base_gd_mses = {
            model.name: self.training_metrics[f"{model.name}_gd_mse"]
            for model in self.base_models
            if f"{model.name}_gd_mse" in self.training_metrics
        }

        if base_loglosses:
            self.best_clf_model_name = min(base_loglosses, key=base_loglosses.get)
            best_base_ll = base_loglosses[self.best_clf_model_name]
            if best_base_ll < stack_ll:
                self.use_stacked = False
                logger.info(
                    f"Using {self.best_clf_model_name} for W/D/L probabilities "
                    f"because its OOF LogLoss ({best_base_ll:.4f}) beats "
                    f"the stacked model ({stack_ll:.4f})."
                )
            else:
                self.use_stacked = True
                logger.info(
                    f"Using stacked probabilities; stacked LogLoss ({stack_ll:.4f}) "
                    f"beats best base LogLoss ({best_base_ll:.4f})."
                )

        if base_gd_mses:
            self.best_reg_model_name = min(base_gd_mses, key=base_gd_mses.get)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"STACKED ENSEMBLE RESULTS:")
        logger.info(f"  LogLoss: {stack_ll:.4f}")
        logger.info(f"  Accuracy: {stack_acc:.3f}")
        logger.info(f"  Goal Diff MSE: {stack_mse:.4f}")
        logger.info(f"{'='*60}")
        
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> dict:
        """
        Predict match outcome for feature vectors.
        
        Args:
            X: Feature matrix (single match or batch)
            
        Returns:
            Dict with keys:
            - proba: ndarray of shape (n, 3) - [P(Loss), P(Draw), P(Win)]
            - pred: ndarray of predicted classes
            - goal_diff: ndarray of expected goal differentials
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if not self.use_stacked:
            clf_model = self._get_base_model(self.best_clf_model_name)
            reg_model = self._get_base_model(self.best_reg_model_name or self.best_clf_model_name)
            if clf_model is not None and reg_model is not None:
                proba = self._normalize_proba(clf_model.predict_proba(X))
                goal_diff = reg_model.predict_gd(X)
                return {
                    "proba": proba,
                    "pred": proba.argmax(axis=1),
                    "goal_diff": goal_diff,
                }

        # Get base model predictions
        base_probas = [self._normalize_proba(model.predict_proba(X)) for model in self.base_models]
        base_gds = [model.predict_gd(X) for model in self.base_models]
        
        # Stack into meta-features
        meta_features = np.hstack(
            base_probas + [gd.reshape(-1, 1) for gd in base_gds]
        )
        
        # Meta-learner predictions
        proba = self._normalize_proba(self.calibrated_meta.predict_proba(meta_features))
        pred = proba.argmax(axis=1)
        goal_diff = self.meta_reg.predict(meta_features)
        
        return {
            "proba": proba,       # [P(Loss), P(Draw), P(Win)]
            "pred": pred,         # 0=Loss, 1=Draw, 2=Win
            "goal_diff": goal_diff,
        }

    def predict_match(self, X: pd.DataFrame) -> dict:
        """
        Predict a single match with detailed output.
        
        Args:
            X: Single-row feature DataFrame
            
        Returns:
            Dict with win/draw/loss probabilities and expected goal diff
        """
        result = self.predict(X)
        
        return {
            "loss_prob": float(result["proba"][0][0]),
            "draw_prob": float(result["proba"][0][1]),
            "win_prob": float(result["proba"][0][2]),
            "predicted_result": ["Loss", "Draw", "Win"][int(result["pred"][0])],
            "expected_goal_diff": float(result["goal_diff"][0]),
        }

    def save(self, path: Optional[Path] = None):
        """Save the complete ensemble to disk."""
        path = path or MODEL_DIR / "stacked_ensemble.joblib"
        
        # Save base models individually
        for model in self.base_models:
            model.save()
        
        # Save meta-learner and metadata
        joblib.dump({
            "meta_clf": self.meta_clf,
            "meta_reg": self.meta_reg,
            "calibrated_meta": self.calibrated_meta,
            "training_metrics": self.training_metrics,
            "base_model_names": [m.name for m in self.base_models],
            "use_stacked": self.use_stacked,
            "best_clf_model_name": self.best_clf_model_name,
            "best_reg_model_name": self.best_reg_model_name,
        }, path)
        
        logger.info(f"Stacked ensemble saved to {path}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "StackedEnsemble":
        """Load a complete ensemble from disk."""
        path = path or MODEL_DIR / "stacked_ensemble.joblib"
        
        data = joblib.load(path)
        ensemble = cls()
        
        ensemble.meta_clf = data["meta_clf"]
        ensemble.meta_reg = data["meta_reg"]
        ensemble.calibrated_meta = data["calibrated_meta"]
        ensemble.training_metrics = data["training_metrics"]
        ensemble.use_stacked = data.get("use_stacked", True)
        ensemble.best_clf_model_name = data.get("best_clf_model_name")
        ensemble.best_reg_model_name = data.get("best_reg_model_name")
        
        # Load base models
        for model in ensemble.base_models:
            model_path = MODEL_DIR / f"{model.name}_model.joblib"
            if model_path.exists():
                model_data = joblib.load(model_path)
                model.clf_model = model_data["clf_model"]
                model.reg_model = model_data["reg_model"]
                model.params = model_data["params"]
                model.is_fitted = True
        
        ensemble.is_fitted = True
        logger.info(f"Stacked ensemble loaded from {path}")
        return ensemble
