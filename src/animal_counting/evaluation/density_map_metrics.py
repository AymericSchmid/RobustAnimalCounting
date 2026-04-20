import numpy as np
from skimage.metrics import structural_similarity as ssim


def compute_ssim_metrics(pred_maps, gt_maps):
    scores = []
    for pred, gt in zip(pred_maps, gt_maps):
        data_range = float(gt.max() - gt.min())
        if data_range == 0:
            continue 
        score = ssim(pred, gt, data_range=data_range)
        scores.append(score)
    
    return {
        "SSIM": float(np.mean(scores)) if scores else float("nan")
    }