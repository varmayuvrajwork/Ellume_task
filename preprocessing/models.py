import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor


class SolarForecastingEnsemble:
    """
    5-Fold ensemble of LightGBM, XGBoost, and CatBoost with Non-Negative Ridge
    meta-blending. Weights are learned from OOF predictions so no base model
    can receive a negative contribution.
    """
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits      = n_splits
        self.random_state  = random_state
        self.lgb_models:   List[lgb.LGBMRegressor]   = []
        self.xgb_models:   List[xgb.XGBRegressor]    = []
        self.cat_models:   List[CatBoostRegressor]    = []
        self.weights:      np.ndarray                 = np.array([0.333, 0.333, 0.334])
        self.feature_names: List[str]                 = []

    def fit(self, X: pd.DataFrame, y: pd.Series,
            sample_weight: np.ndarray = None) -> Dict[str, float]:
        """Train ensemble and return OOF metrics."""
        _, metrics = self.fit_and_return_oof(X, y, sample_weight)
        return metrics

    def fit_and_return_oof(self, X: pd.DataFrame, y: pd.Series,
                            sample_weight: np.ndarray = None) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Train 5-fold ensemble across three base learners.
        Returns the full-length OOF blend array and per-model metrics.
        The OOF blend is computed with the same NNLS weights used at inference,
        so benchmark metrics and predictions.csv are produced by identical logic.
        """
        self.feature_names = X.columns.tolist()
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        oof_lgb = np.zeros(len(X))
        oof_xgb = np.zeros(len(X))
        oof_cat = np.zeros(len(X))

        self.lgb_models = []
        self.xgb_models = []
        self.cat_models = []

        print(f"--- Training {self.n_splits}-Fold Ensemble Models ---")

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X.iloc[val_idx],   y.iloc[val_idx]

            lgb_m = lgb.LGBMRegressor(
                n_estimators=1200, learning_rate=0.03, num_leaves=63,
                subsample=0.8, colsample_bytree=0.8,
                random_state=self.random_state + fold, verbose=-1,
            )
            lgb_m.fit(X_tr, y_tr,
                      eval_X=X_va, eval_y=y_va,
                      callbacks=[lgb.early_stopping(50, verbose=False)])
            oof_lgb[val_idx] = lgb_m.predict(X_va)
            self.lgb_models.append(lgb_m)

            xgb_m = xgb.XGBRegressor(
                n_estimators=1200, learning_rate=0.03, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                random_state=self.random_state + fold,
                early_stopping_rounds=50, verbosity=0,
            )
            xgb_m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            oof_xgb[val_idx] = xgb_m.predict(X_va)
            self.xgb_models.append(xgb_m)

            cat_m = CatBoostRegressor(
                iterations=1200, learning_rate=0.03, depth=6,
                random_seed=self.random_state + fold,
                early_stopping_rounds=50,
                allow_writing_files=False, verbose=False,
            )
            cat_m.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
            oof_cat[val_idx] = cat_m.predict(X_va)
            self.cat_models.append(cat_m)

        # Non-Negative Ridge meta-blending — weights learned from OOF predictions
        oof_matrix = np.column_stack([oof_lgb, oof_cat, oof_xgb])
        meta       = Ridge(alpha=1.0, positive=True, fit_intercept=False)
        meta.fit(oof_matrix, y)
        raw_w        = meta.coef_
        self.weights = raw_w / raw_w.sum() if raw_w.sum() > 0 else np.array([0.40, 0.35, 0.25])

        print(f"  Learned Weights (LGB, Cat, XGB): "
              f"[{self.weights[0]:.3f}, {self.weights[1]:.3f}, {self.weights[2]:.3f}]")

        oof_blend = oof_matrix @ self.weights

        metrics = {
            'lgb_rmse':      root_mean_squared_error(y, oof_lgb),
            'xgb_rmse':      root_mean_squared_error(y, oof_xgb),
            'cat_rmse':      root_mean_squared_error(y, oof_cat),
            'ensemble_rmse': root_mean_squared_error(y, oof_blend),
            'ensemble_mae':  mean_absolute_error(y, oof_blend),
            'ensemble_r2':   r2_score(y, oof_blend),
        }

        print("\n=== Out-of-Fold Performance ===")
        print(f"  LightGBM : {metrics['lgb_rmse']:.2f} kW RMSE")
        print(f"  XGBoost  : {metrics['xgb_rmse']:.2f} kW RMSE")
        print(f"  CatBoost : {metrics['cat_rmse']:.2f} kW RMSE")
        print(f"  Ensemble : {metrics['ensemble_rmse']:.2f} kW RMSE  |  "
              f"{metrics['ensemble_mae']:.2f} kW MAE  |  R²={metrics['ensemble_r2']:.4f}")

        return oof_blend, metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Average fold-model predictions and apply learned NNLS blend weights."""
        X_feat = X[self.feature_names]
        lgb_p  = np.mean([m.predict(X_feat) for m in self.lgb_models], axis=0)
        xgb_p  = np.mean([m.predict(X_feat) for m in self.xgb_models], axis=0)
        cat_p  = np.mean([m.predict(X_feat) for m in self.cat_models], axis=0)
        return np.column_stack([lgb_p, cat_p, xgb_p]) @ self.weights
