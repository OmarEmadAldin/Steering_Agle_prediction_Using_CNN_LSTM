"""
metrics.py
Evaluation metrics for steering angle prediction, in interpretable units.

Assumes angles are stored in radians (standard for driving datasets). If
your dataset's ANGLE_COL is already in degrees, set RAD_TO_DEG = 1.0 below.
"""

import numpy as np

# steering-dataset (phamvoquoclong) stores angles in DEGREES already
# (values like -66.25), so no radians->degrees conversion is needed here.
RAD_TO_DEG = 1.0


def mae_degrees(preds, targets):
    """Mean absolute error, converted to degrees."""
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    return float(np.mean(np.abs(preds - targets)) * RAD_TO_DEG)


def rmse_degrees(preds, targets):
    """Root mean squared error, converted to degrees."""
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    return float(np.sqrt(np.mean((preds - targets) ** 2)) * RAD_TO_DEG)


def straight_vs_turn_mae(preds, targets, turn_threshold_deg=5.0):
    """
    Splits MAE into 'straight' frames (|angle| below threshold) and 'turn'
    frames (|angle| at or above threshold). A model that just predicts ~0
    can still show a low overall MAE while doing badly on turns -- this
    metric catches that.
    """
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    threshold_rad = turn_threshold_deg / RAD_TO_DEG

    straight_mask = np.abs(targets) < threshold_rad
    turn_mask = ~straight_mask

    straight_mae = (
        float(np.mean(np.abs(preds[straight_mask] - targets[straight_mask])) * RAD_TO_DEG)
        if straight_mask.any() else float("nan")
    )
    turn_mae = (
        float(np.mean(np.abs(preds[turn_mask] - targets[turn_mask])) * RAD_TO_DEG)
        if turn_mask.any() else float("nan")
    )

    return {
        "straight_mae_deg": straight_mae,
        "turn_mae_deg": turn_mae,
        "num_straight": int(straight_mask.sum()),
        "num_turn": int(turn_mask.sum()),
    }


def evaluate(preds, targets, turn_threshold_deg=5.0):
    """Convenience wrapper returning all metrics as one dict."""
    result = {
        "mae_deg": mae_degrees(preds, targets),
        "rmse_deg": rmse_degrees(preds, targets),
    }
    result.update(straight_vs_turn_mae(preds, targets, turn_threshold_deg))
    return result