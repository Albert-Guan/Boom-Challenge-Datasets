import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.preprocessing import StandardScaler

from geometry import build_geometry

# Compute log(raw physical quantity), then z-score those log columns.
_LOG_SOURCES = {
    "pi_strength": "log_pi_strength",
    "coupling": "log_coupling",
    "porosity": "log_porosity",
    "shape_factor": "log_shape",
    "pi_threshold_fines": "log_pi_threshold_fines",
    "pi_threshold_oversize": "log_pi_threshold_oversize",
    "energy": "log_energy",
    "strength": "log_strength",
    "gravity": "log_gravity",
    "atmosphere": "log_atmosphere",
}

_EPS = 1e-6


class ZLogFeatureTransforms:
    """Fit StandardScaler on log-features; reuse it for validation/test."""

    def __init__(self):
        self._scalers: dict[str, StandardScaler] = {}

    def _raw_logs(self, geom: pd.DataFrame) -> pd.DataFrame:
        logs = pd.DataFrame(index=geom.index)
        for source, log_col in _LOG_SOURCES.items():
            logs[log_col] = np.log(geom[source].clip(lower=_EPS))
        return logs

    def fit(self, geom: pd.DataFrame) -> "ZLogFeatureTransforms":
        logs = self._raw_logs(geom)
        for log_col in logs:
            self._scalers[log_col] = StandardScaler().fit(logs[[log_col]].to_numpy())
        return self

    def transform(self, geom: pd.DataFrame) -> pd.DataFrame:
        X = geom.copy()
        logs = self._raw_logs(geom)
        for log_col in logs:
            raw_log = logs[[log_col]].to_numpy()
            X[f"{log_col}_raw"] = raw_log.ravel()
            X[log_col] = self._scalers[log_col].transform(raw_log).ravel()
        return X


def build_features(df: pd.DataFrame, z_log: ZLogFeatureTransforms | None = None) -> pd.DataFrame:
    """Raw scenario table -> full feature matrix.

    Pass a train-fitted z_log to reuse training statistics for validation/test.
    If omitted, fit on df for backwards-compatible exploratory use.
    """
    geom = build_geometry(df)
    if z_log is None:
        z_log = ZLogFeatureTransforms().fit(geom)
    return z_log.transform(geom)


def transform_targets(y: pd.DataFrame, L_char: pd.Series) -> pd.DataFrame:
    y_t = pd.DataFrame(index=y.index)
    for col in ['P80', 'R95', 'R50_fines', 'R50_oversize']:
        y_t[f'log_pi_{col}'] = np.log(y[col] / L_char)

    eps = 1e-6
    y_t['logit_fines_frac'] = logit(y['fines_frac'].clip(eps, 1 - eps))
    y_t['logit_oversize_frac'] = logit(y['oversize_frac'].clip(eps, 1 - eps))
    return y_t


def invert_targets(y_pred: pd.DataFrame, L_char: pd.Series) -> pd.DataFrame:
    from scipy.special import expit
    y_out = pd.DataFrame(index=y_pred.index)
    for col in ['P80', 'R95', 'R50_fines', 'R50_oversize']:
        y_out[col] = np.exp(y_pred[f'log_pi_{col}']) * L_char
    y_out['fines_frac'] = expit(y_pred['logit_fines_frac'])
    y_out['oversize_frac'] = expit(y_pred['logit_oversize_frac'])
    return y_out
