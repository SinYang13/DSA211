"""
Shared Cross-Validation & Pipeline Utilities for DSA211 Project
Strictly follows guidelines in docs/project_cv_workflow.html:
- Expanding-window time-series splitter: one validation fold per calendar month
  (Feb 2012 - Sep 2012, k = 8), each trained on all earlier months.
- Standard evaluation metrics: Count-scale MSE and RMSE with Standard Error (SE = SD / sqrt(k)).
- Log-normal back-transformation with variance correction: exp(log_pred + sigma^2 / 2).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RANDOM_STATE = 2026
FIRST_VAL_MONTH = '2012-02'
LAST_VAL_MONTH = '2012-09'

class ExpandingWindowSplit:
    """
    Expanding-window CV over calendar months. For each validation month m,
    trains on every row dated before m and validates on the rows in m.
    Row positions follow `dates`, so X passed to split() must be in the same row order.
    Works anywhere scikit-learn accepts a cv object (cross_val_score, GridSearchCV).
    """

    def __init__(self, dates, first_val_month=FIRST_VAL_MONTH, last_val_month=LAST_VAL_MONTH):
        self.months = pd.to_datetime(pd.Series(dates)).dt.to_period('M').to_numpy()
        first, last = pd.Period(first_val_month, 'M'), pd.Period(last_val_month, 'M')
        self.val_months = [m for m in sorted(set(self.months)) if first <= m <= last]

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.val_months)

    def split(self, X, y=None, groups=None):
        if len(X) != len(self.months):
            raise ValueError(f"X has {len(X)} rows but the splitter was built on {len(self.months)} dates")
        for m in self.val_months:
            yield np.flatnonzero(self.months < m), np.flatnonzero(self.months == m)

def get_cv_splitter(dates):
    """Returns the shared expanding-window splitter (k = 8) for all candidate models."""
    return ExpandingWindowSplit(dates)

def back_transform_preds(log_preds, resid_variance):
    """
    Back-transforms predictions from log scale to original count scale
    using the log-normal expectation correction: exp(log_pred + sigma^2 / 2).
    """
    return np.exp(log_preds + resid_variance / 2.0)

def evaluate_cv_model(estimator, X, y, cv, is_log_target=False):
    """
    Evaluates an estimator or Pipeline across cross-validation folds.
    cv is the shared splitter from get_cv_splitter(dates), built on the same rows as X.
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
