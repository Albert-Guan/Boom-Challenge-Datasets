import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import lightgbm as lgb

# 1. Base features shared by all models
LENGTH_LINEAR_FEATURES = [
    'log_pi_strength', 'log_coupling', 'log_porosity', 'log_shape',
    'sin_angle', 'cos_angle',
]

BASE_LGB_FEATURES = LENGTH_LINEAR_FEATURES + [
    'log_energy', 'log_strength', 'log_gravity', 'log_atmosphere',
]

# 2. Target-specific feature lists
FINES_LINEAR_FEATURES = LENGTH_LINEAR_FEATURES + ['log_pi_threshold_fines']
FINES_LGB_FEATURES    = BASE_LGB_FEATURES + ['log_pi_threshold_fines']

OVERSIZE_LINEAR_FEATURES = LENGTH_LINEAR_FEATURES + ['log_pi_threshold_oversize']
OVERSIZE_LGB_FEATURES    = BASE_LGB_FEATURES + ['log_pi_threshold_oversize']

LENGTH_TARGETS = ['log_pi_P80', 'log_pi_R95', 'log_pi_R50_fines', 'log_pi_R50_oversize']
FRACTION_TARGETS = ['logit_fines_frac', 'logit_oversize_frac']
ALL_TARGETS = LENGTH_TARGETS + FRACTION_TARGETS

class HybridPhysicsModel:
    def __init__(self, linear_features, lgb_features, lgb_params=None):
        self.linear_features = linear_features
        self.lgb_features = lgb_features
        self.lgb_params = lgb_params or {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_child_samples': 10,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbose': -1,
        }
        self.baseline = None
        self.residual = None
        self.best_iteration_ = None

    def fit(self, X, y, eval_set=None, eval_names=None, eval_metric='l2', callbacks=None):
        self.baseline = LinearRegression().fit(X[self.linear_features], y)
        residuals = y - self.baseline.predict(X[self.linear_features])
        residual_eval_set = None
        if eval_set is not None:
            residual_eval_set = []
            for X_eval, y_eval in eval_set:
                residual_eval_set.append((
                    X_eval[self.lgb_features],
                    y_eval - self.baseline.predict(X_eval[self.linear_features]),
                ))

        self.residual = lgb.LGBMRegressor(**self.lgb_params).fit(
            X[self.lgb_features],
            residuals,
            eval_set=residual_eval_set,
            eval_names=eval_names,
            eval_metric=eval_metric,
            callbacks=callbacks,
        )
        self.best_iteration_ = getattr(self.residual, 'best_iteration_', None)
        return self

    def predict(self, X):
        num_iteration = self.best_iteration_ if self.best_iteration_ else None
        return (self.baseline.predict(X[self.linear_features])
                + self.residual.predict(X[self.lgb_features], num_iteration=num_iteration))

    def baseline_coefficients(self):
        return pd.Series(self.baseline.coef_, index=self.linear_features)


def make_model(target: str) -> HybridPhysicsModel:
    if target == 'logit_fines_frac':
        return HybridPhysicsModel(linear_features=FINES_LINEAR_FEATURES, lgb_features=FINES_LGB_FEATURES)
    elif target == 'logit_oversize_frac':
        return HybridPhysicsModel(linear_features=OVERSIZE_LINEAR_FEATURES, lgb_features=OVERSIZE_LGB_FEATURES)
    else:
        return HybridPhysicsModel(linear_features=LENGTH_LINEAR_FEATURES, lgb_features=BASE_LGB_FEATURES)
