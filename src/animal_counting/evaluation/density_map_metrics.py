import numpy as np
from skimage.metrics import structural_similarity as ssim


def compute_ssim_metrics(pred_maps, gt_maps):
    scores = []
    for pred, gt in zip(pred_maps, gt_maps):
        data_range = float(gt.max() - gt.min())
        # If the ground truth map has no variation (i.e., all values are the same), we cannot compute SSIM, so we skip this pair.
        if data_range == 0:
            continue 
        score = ssim(pred, gt, data_range=data_range)
        scores.append(score)

    # Return the average SSIM score across all pairs, or NaN if there are no valid pairs
    return {"SSIM": float(np.mean(scores)) if scores else float("nan")}