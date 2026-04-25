import numpy as np

def mae(pred, gt) -> float:
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    if len(gt) == 0:
        return float("nan")
    return float(np.mean(np.abs(pred - gt)))

def rmse(pred, gt) -> float:
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    if len(gt) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((pred - gt) ** 2)))

def relative_error(pred, gt) -> float:
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    if len(gt) == 0:
        return float("nan")
    return float(np.mean(np.abs(pred - gt) / np.maximum(gt, 1.0)))

def count_metrics(pred_counts, gt_counts):
    pred = np.asarray(pred_counts, dtype=float)
    gt = np.asarray(gt_counts, dtype=float)
    return {
        "n_images": int(len(gt)),
        "MAE": mae(pred, gt),
        "RMSE": rmse(pred, gt),
        "relative_error": relative_error(pred, gt)
          }