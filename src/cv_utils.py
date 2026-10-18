"""
Shared Cross-Validation & Pipeline Utilities for DSA211 Project
Strictly follows guidelines in docs/project_cv_workflow.html:
- Fixed KFold splitter (k=10, shuffle=True, random_state=2026).
- Standard evaluation metrics: Count-scale MSE and RMSE with Standard Error (SE = SD / sqrt(k)).
- Log-normal back-transformation with variance correction: exp(log_pred + sigma^2 / 2).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

RANDOM_STATE = 2026
N_SPLITS = 10

def get_cv_splitter(n_splits=N_SPLITS, random_state=RANDOM_STATE):
    """Returns the shared KFold splitter for all candidate models."""
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

def back_transform_preds(log_preds, resid_variance):
    """
    Back-transforms predictions from log scale to original count scale
    using the log-normal expectation correction: exp(log_pred + sigma^2 / 2).
    """
    return np.exp(log_preds + resid_variance / 2.0)

def evaluate_cv_model(estimator, X, y, cv=None, is_log_target=False):
    """
    Evaluates an estimator or Pipeline across cross-validation folds.
    If is_log_target is True, fits on log(y) and back-transforms fold predictions
    before computing count-scale metrics.
    
    Returns a dict with:
      - 'fold_mse': list of MSE per fold (count scale)
      - 'fold_rmse': list of RMSE per fold (count scale)
      - 'mean_mse': mean MSE across folds
      - 'se_mse': standard error of MSE across folds (SD / sqrt(k))
      - 'mean_rmse': count-scale RMSE
      - 'se_rmse': standard error of RMSE across folds
      - 'mean_mae': mean MAE across folds
    """
    if cv is None:
        cv = get_cv_splitter()
        
    fold_mse = []
    fold_rmse = []
    fold_mae = []
    
    # Ensure X and y are indexed consistently
    X_df = X.reset_index(drop=True) if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    y_ser = y.reset_index(drop=True) if isinstance(y, (pd.Series, pd.DataFrame)) else pd.Series(y)
    
    k = cv.get_n_splits()
    
    for train_idx, val_idx in cv.split(X_df, y_ser):
        X_tr, X_val = X_df.iloc[train_idx], X_df.iloc[val_idx]
        y_tr, y_val = y_ser.iloc[train_idx], y_ser.iloc[val_idx]
        
        if is_log_target:
            y_tr_fit = np.log(y_tr)
            estimator.fit(X_tr, y_tr_fit)
            log_preds_val = estimator.predict(X_val)
            
            # Estimate residual variance from training fit
            log_preds_tr = estimator.predict(X_tr)
            resid_var = np.var(y_tr_fit - log_preds_tr, ddof=1)
            
            preds_val = back_transform_preds(log_preds_val, resid_var)
        else:
            estimator.fit(X_tr, y_tr)
            preds_val = estimator.predict(X_val)
            
        mse = mean_squared_error(y_val, preds_val)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val, preds_val)
        
        fold_mse.append(mse)
        fold_rmse.append(rmse)
        fold_mae.append(mae)
        
    mean_mse = np.mean(fold_mse)
    se_mse = np.std(fold_mse, ddof=1) / np.sqrt(k)
    
    mean_rmse = np.mean(fold_rmse)
    se_rmse = np.std(fold_rmse, ddof=1) / np.sqrt(k)
    
    mean_mae = np.mean(fold_mae)
    se_mae = np.std(fold_mae, ddof=1) / np.sqrt(k)
    
    return {
        'fold_mse': fold_mse,
        'fold_rmse': fold_rmse,
        'mean_mse': mean_mse,
        'se_mse': se_mse,
        'mean_rmse': mean_rmse,
        'se_rmse': se_rmse,
        'mean_mae': mean_mae,
        'se_mae': se_mae
    }
