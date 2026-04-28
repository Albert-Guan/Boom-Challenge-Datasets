import numpy as np
import pandas as pd


def build_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Build raw physics geometry and dimensionless pi-group features."""
    X = df.copy()
    X["L_char"] = (X["energy"] / (X["atmosphere"] * X["gravity"])) ** 0.25
    X["pi_strength"] = X["strength"] / (X["atmosphere"] * X["gravity"] * X["L_char"])

    X["sin_angle"] = np.sin(X["angle_rad"])
    X["cos_angle"] = np.cos(X["angle_rad"])

    X["D_fines"] = 40.0
    X["D_oversize"] = 120.0
    X["pi_threshold_fines"] = X["D_fines"] / X["L_char"]
    X["pi_threshold_oversize"] = X["D_oversize"] / X["L_char"]
    return X
