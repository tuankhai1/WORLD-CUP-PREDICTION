"""
Wrappers around CatBoost, XGBoost, and LightGBM that provide a unified.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import joblib

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import RANDOM_SEED, MODEL_DIR, OPTUNA_CV_FOLDS

logger = logging.getLogger(__name__)


class BaseModelWrapper:
    """
    Abstract base for all model wrappers.
    Provides unified interface for classification + regression.
    """
    
    def __init__(self, name: str, params: Optional[dict] = None):
        self.name = name
        self.params = params or {}
        self.clf_model = None    # W/D/L classifier
        self.reg_model = None    # Goal diff regressor
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y_wdl: pd.Series, y_gd: pd.Series, sample_weights: Optional[pd.Series] = None):
        raise NotImplementedError
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict W/D/L probabilities. Shape: (n_samples, 3)"""
        raise NotImplementedError
    
    def predict_gd(self, X: pd.DataFrame) -> np.ndarray:
        """Predict expected goal differential. Shape: (n_samples,)"""
        raise NotImplementedError
    
    def get_oof_predictions(self, X: pd.DataFrame, y_wdl: pd.Series, 
                             y_gd: pd.Series, n_folds: int = OPTUNA_CV_FOLDS,
                             sample_weights: Optional[pd.Series] = None
                             ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate out-of-fold predictions for stacking.
        
        For each fold, train on K-1 folds and predict on the held-out fold.
        This avoids data leakage in the stacking meta-learner.
        
        Returns:
            Tuple of (oof_proba [n_samples, 3], oof_gd [n_samples])
        """
        raise NotImplementedError
    
    def save(self, path: Optional[Path] = None):
        path = path or MODEL_DIR / f"{self.name}_model.joblib"
        joblib.dump({
            "clf_model": self.clf_model,
            "reg_model": self.reg_model,
            "params": self.params,
        }, path)
        logger.info(f"Saved {self.name} model to {path}")


class CatBoostWrapper(BaseModelWrapper):
    """CatBoost model wrapper with native categorical support."""
    
    def __init__(self, params: Optional[dict] = None):
        super().__init__("catboost", params)
        self._default_params = {
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "loss_function": "MultiClass",
            "classes_count": 3,
            "random_seed": RANDOM_SEED,
            "verbose": 0,
            "thread_count": -1,
        }
    
    def fit(self, X: pd.DataFrame, y_wdl: pd.Series, y_gd: pd.Series, sample_weights: Optional[pd.Series] = None):
        from catboost import CatBoostClassifier, CatBoostRegressor
        
        # Determine weight argument depending on whether sample_weights is None
        fit_kwargs = {}
        if sample_weights is not None:
            fit_kwargs['sample_weight'] = sample_weights

        clf_params = {**self._default_params, **self.params}
        self.clf_model = CatBoostClassifier(**clf_params)
        self.clf_model.fit(X, y_wdl, verbose=0, **fit_kwargs)
        
        reg_params = {k: v for k, v in clf_params.items() 
                      if k not in ["loss_function", "classes_count"]}
        reg_params["loss_function"] = "RMSE"
        self.reg_model = CatBoostRegressor(**reg_params)
        self.reg_model.fit(X, y_gd, verbose=0, **fit_kwargs)
        
        self.is_fitted = True
        logger.info(f"CatBoost fitted on {len(X)} samples")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf_model.predict_proba(X)
    
    def predict_gd(self, X: pd.DataFrame) -> np.ndarray:
        return self.reg_model.predict(X)
    
    def get_oof_predictions(self, X: pd.DataFrame, y_wdl: pd.Series, 
                             y_gd: pd.Series, n_folds: int = OPTUNA_CV_FOLDS,
                             sample_weights: Optional[pd.Series] = None
                             ) -> tuple[np.ndarray, np.ndarray]:
        from catboost import CatBoostClassifier, CatBoostRegressor
        
        oof_proba = np.zeros((len(X), 3))
        oof_gd = np.zeros(len(X))
        
        tscv = TimeSeriesSplit(n_splits=n_folds)
        clf_params = {**self._default_params, **self.params}
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_wdl_train, y_wdl_val = y_wdl.iloc[train_idx], y_wdl.iloc[val_idx]
            y_gd_train = y_gd.iloc[train_idx]
            
            
            # Classification
            clf = CatBoostClassifier(**clf_params)
            fit_kwargs = {}
            if sample_weights is not None:
                fit_kwargs['sample_weight'] = sample_weights.iloc[train_idx]
                
            clf.fit(X_train, y_wdl_train, eval_set=(X_val, y_wdl_val),
                    early_stopping_rounds=50, verbose=0, **fit_kwargs)
            oof_proba[val_idx] = clf.predict_proba(X_val)
            
            # Regression
            reg_params = {k: v for k, v in clf_params.items() 
                          if k not in ["loss_function", "classes_count"]}
            reg_params["loss_function"] = "RMSE"
            reg = CatBoostRegressor(**reg_params)
            reg.fit(X_train, y_gd_train, verbose=0, **fit_kwargs)
            oof_gd[val_idx] = reg.predict(X_val)
        
        # Train final model on all data
        self.fit(X, y_wdl, y_gd, sample_weights=sample_weights)
        
        return oof_proba, oof_gd

    def feature_importance(self) -> pd.Series:
        """Get feature importance from the classifier."""
        if self.clf_model is None:
            return pd.Series()
        importances = self.clf_model.get_feature_importance()
        feature_names = self.clf_model.feature_names_
        return pd.Series(importances, index=feature_names).sort_values(ascending=False)


class XGBoostWrapper(BaseModelWrapper):
    """XGBoost model wrapper."""
    
    def __init__(self, params: Optional[dict] = None):
        super().__init__("xgboost", params)
        self._default_params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "multi:softprob",
            "num_class": 3,
            "random_state": RANDOM_SEED,
            "verbosity": 0,
            "n_jobs": -1,
        }
    
    def fit(self, X: pd.DataFrame, y_wdl: pd.Series, y_gd: pd.Series, sample_weights: Optional[pd.Series] = None):
        from xgboost import XGBClassifier, XGBRegressor
        
        fit_kwargs = {}
        if sample_weights is not None:
            fit_kwargs['sample_weight'] = sample_weights
            
        clf_params = {**self._default_params, **self.params}
        self.clf_model = XGBClassifier(**clf_params)
        self.clf_model.fit(X, y_wdl, verbose=False, **fit_kwargs)
        
        reg_params = {k: v for k, v in clf_params.items() 
                      if k not in ["objective", "num_class"]}
        reg_params["objective"] = "reg:squarederror"
        self.reg_model = XGBRegressor(**reg_params)
        self.reg_model.fit(X, y_gd, verbose=False, **fit_kwargs)
        
        self.is_fitted = True
        logger.info(f"XGBoost fitted on {len(X)} samples")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf_model.predict_proba(X)
    
    def predict_gd(self, X: pd.DataFrame) -> np.ndarray:
        return self.reg_model.predict(X)
    
    def get_oof_predictions(self, X: pd.DataFrame, y_wdl: pd.Series, 
                             y_gd: pd.Series, n_folds: int = OPTUNA_CV_FOLDS,
                             sample_weights: Optional[pd.Series] = None
                             ) -> tuple[np.ndarray, np.ndarray]:
        from xgboost import XGBClassifier, XGBRegressor
        
        oof_proba = np.zeros((len(X), 3))
        oof_gd = np.zeros(len(X))
        
        tscv = TimeSeriesSplit(n_splits=n_folds)
        clf_params = {**self._default_params, **self.params}
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_wdl_train = y_wdl.iloc[train_idx]
            y_gd_train = y_gd.iloc[train_idx]
            
            clf = XGBClassifier(**clf_params)
            fit_kwargs = {}
            if sample_weights is not None:
                fit_kwargs['sample_weight'] = sample_weights.iloc[train_idx]
                
            clf.fit(X_train, y_wdl_train, verbose=False, **fit_kwargs)
            oof_proba[val_idx] = clf.predict_proba(X_val)
            
            reg_params = {k: v for k, v in clf_params.items() 
                          if k not in ["objective", "num_class"]}
            reg_params["objective"] = "reg:squarederror"
            reg = XGBRegressor(**reg_params)
            reg.fit(X_train, y_gd_train, verbose=False, **fit_kwargs)
            oof_gd[val_idx] = reg.predict(X_val)
        
        self.fit(X, y_wdl, y_gd, sample_weights=sample_weights)
        return oof_proba, oof_gd

    def feature_importance(self) -> pd.Series:
        if self.clf_model is None:
            return pd.Series()
        return pd.Series(
            self.clf_model.feature_importances_,
            index=self.clf_model.get_booster().feature_names
        ).sort_values(ascending=False)


class LightGBMWrapper(BaseModelWrapper):
    """LightGBM model wrapper."""
    
    def __init__(self, params: Optional[dict] = None):
        super().__init__("lightgbm", params)
        self._default_params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "objective": "multiclass",
            "num_class": 3,
            "random_state": RANDOM_SEED,
            "verbosity": -1,
            "n_jobs": -1,
        }
    
    def fit(self, X: pd.DataFrame, y_wdl: pd.Series, y_gd: pd.Series, sample_weights: Optional[pd.Series] = None):
        from lightgbm import LGBMClassifier, LGBMRegressor
        
        fit_kwargs = {}
        if sample_weights is not None:
            fit_kwargs['sample_weight'] = sample_weights
            
        clf_params = {**self._default_params, **self.params}
        self.clf_model = LGBMClassifier(**clf_params)
        self.clf_model.fit(X, y_wdl, **fit_kwargs)
        
        reg_params = {k: v for k, v in clf_params.items() 
                      if k not in ["objective", "num_class"]}
        reg_params["objective"] = "regression"
        self.reg_model = LGBMRegressor(**reg_params)
        self.reg_model.fit(X, y_gd, **fit_kwargs)
        
        self.is_fitted = True
        logger.info(f"LightGBM fitted on {len(X)} samples")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf_model.predict_proba(X)
    
    def predict_gd(self, X: pd.DataFrame) -> np.ndarray:
        return self.reg_model.predict(X)
    
    def get_oof_predictions(self, X: pd.DataFrame, y_wdl: pd.Series, 
                             y_gd: pd.Series, n_folds: int = OPTUNA_CV_FOLDS,
                             sample_weights: Optional[pd.Series] = None
                             ) -> tuple[np.ndarray, np.ndarray]:
        from lightgbm import LGBMClassifier, LGBMRegressor
        
        oof_proba = np.zeros((len(X), 3))
        oof_gd = np.zeros(len(X))
        
        tscv = TimeSeriesSplit(n_splits=n_folds)
        clf_params = {**self._default_params, **self.params}
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_wdl_train = y_wdl.iloc[train_idx]
            y_gd_train = y_gd.iloc[train_idx]
            
            clf = LGBMClassifier(**clf_params)
            fit_kwargs = {}
            if sample_weights is not None:
                fit_kwargs['sample_weight'] = sample_weights.iloc[train_idx]
                
            clf.fit(X_train, y_wdl_train, **fit_kwargs)
            oof_proba[val_idx] = clf.predict_proba(X_val)
            
            reg_params = {k: v for k, v in clf_params.items() 
                          if k not in ["objective", "num_class"]}
            reg_params["objective"] = "regression"
            reg = LGBMRegressor(**reg_params)
            reg.fit(X_train, y_gd_train, **fit_kwargs)
            oof_gd[val_idx] = reg.predict(X_val)
        
        self.fit(X, y_wdl, y_gd, sample_weights=sample_weights)
        return oof_proba, oof_gd

    def feature_importance(self) -> pd.Series:
        if self.clf_model is None:
            return pd.Series()
        return pd.Series(
            self.clf_model.feature_importances_,
            index=self.clf_model.feature_name_
        ).sort_values(ascending=False)
