import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, roc_auc_score)


def evaluate(y_odx, reg_pred, prob, tau: float = 25.0, thr: float = 0.5):
    y_odx, reg_pred, prob = map(np.asarray, (y_odx, reg_pred, prob))
    y = (y_odx > tau).astype(int)
    pred = (prob > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y, prob)) if len(set(y)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "mae": float(mean_absolute_error(y_odx, reg_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_odx, reg_pred))),
    }
