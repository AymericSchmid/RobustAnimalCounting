density_list = ["sparse", "medium", "crowded"]
LIMITS = [10, 50]  # Define the limits for the density buckets

def classify_by_density(gt_count: int) -> str:
    if gt_count <= LIMITS[0]:
        return density_list[0]  # "sparse"
    elif gt_count <= LIMITS[1]:
        return density_list[1]  # "medium"
    else:
        return density_list[2]  # "crowded"

def split_by_density(image_ids, pred_counts, gt_counts):
    if not (len(image_ids) == len(pred_counts) == len(gt_counts)):
        raise ValueError("image_ids, pred_counts and gt_counts must have the same length.")

    density = {name: {"image_ids": [], "indices": [], "pred_counts": [], "gt_counts": []}
               for name in density_list}

    for i, (img_id, pred, gt) in enumerate(zip(image_ids, pred_counts, gt_counts)):
        bucket = classify_by_density(int(gt))
        density[bucket]["image_ids"].append(img_id)
        density[bucket]["indices"].append(i)
        density[bucket]["pred_counts"].append(pred)
        density[bucket]["gt_counts"].append(gt)
    return density