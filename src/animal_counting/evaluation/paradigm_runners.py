from .counting_metrics import count_metrics
from .density_buckets import density_list, split_by_density
from .density_map_metrics import compute_ssim_metrics


def evaluate_yolo_density(image_ids, pred_counts, gt_counts):
    # Overall metric value
    results = {"overall": count_metrics(pred_counts, gt_counts)}

    # Per density value
    density_groups = split_by_density(image_ids, pred_counts, gt_counts)
    for density_name in density_list:
        g = density_groups[density_name]
        results[density_name] = count_metrics(g["pred_counts"], g["gt_counts"])
    return results


def evaluate_yolo_cross(pred_counts, gt_counts):
    return {"overall": count_metrics(pred_counts, gt_counts)}


def evaluate_csrnet_density(image_ids, pred_counts, gt_counts,
                             pred_maps, gt_maps):
    # Overall
    results = {"overall": count_metrics(pred_counts, gt_counts)}
    results["overall"].update(compute_ssim_metrics(pred_maps, gt_maps))

    # Per density metric
    density_groups = split_by_density(image_ids, pred_counts, gt_counts)
    for density_name in density_list:
        g = density_groups[density_name]
        indices = g["indices"]
        bucket_pred_maps = [pred_maps[i] for i in indices]
        bucket_gt_maps   = [gt_maps[i]   for i in indices]
        results[density_name] = count_metrics(g["pred_counts"], g["gt_counts"])
        results[density_name].update(compute_ssim_metrics(bucket_pred_maps,
                                                           bucket_gt_maps))
    return results


def evaluate_csrnet_cross(pred_counts, gt_counts,
                           pred_maps, gt_maps):
    results = {"overall": count_metrics(pred_counts, gt_counts)}
    results["overall"].update(compute_ssim_metrics(pred_maps, gt_maps))
    return results


def evaluate_p2pnet_density(image_ids, pred_counts, gt_counts):
    results = {"overall": count_metrics(pred_counts, gt_counts)}

    # Per density metric
    density_groups = split_by_density(image_ids, pred_counts, gt_counts)
    for density_name in density_list:
        g = density_groups[density_name]
        results[density_name] = count_metrics(g["pred_counts"], g["gt_counts"])
    return results


def evaluate_p2pnet_cross(pred_counts, gt_counts):
    return {"overall": count_metrics(pred_counts, gt_counts)}