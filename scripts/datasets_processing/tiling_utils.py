"""Utility functions for tiling large images into fixed-size tiles."""

from pathlib import Path
from typing import Tuple, List, Dict, Any
import pandas as pd
from PIL import Image


def tile_image_and_annotations(
    src_image_path: Path,
    annotations: pd.DataFrame,
    image_name: str,
    output_dir: Path,
    split: str,
    tile_size: int = 1024,
    overlap: float = 0.2,
    save_empty_tiles: bool = False,
    bbox_columns: Tuple[str, str, str, str] = ("x1", "y1", "x2", "y2"),
    point_columns: Tuple[str, str] = None,
) -> List[Dict[str, Any]]:
    """
    Tile a large image and its annotations.

    Args:
        src_image_path: Path to source image file.
        annotations: DataFrame with annotations for this image.
        image_name: Identifier for the image in the dataset.
        output_dir: Output directory where tiles will be saved.
        split: Data split (train, val, test).
        tile_size: Tile width/height in pixels.
        overlap: Fractional overlap between tiles (0.0-0.5).
        save_empty_tiles: If True, save tiles without annotations.
        bbox_columns: Column names for bounding boxes (x1, y1, x2, y2).
        point_columns: Column names for points (x, y). If provided, points are kept as points in the output.

    Returns:
        List of annotation rows for the tiled image.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(src_image_path).convert("RGB")
    w, h = img.size

    # Use no overlap for test split to avoid duplicate detections at edges
    eff_overlap = overlap if split != "test" else 0.0
    stride = max(1, int(tile_size * (1.0 - eff_overlap)))

    tiled_rows = []

    for top in range(0, max(1, h), stride):
        for left in range(0, max(1, w), stride):
            right = left + tile_size
            bottom = top + tile_size

            # cropped region within image bounds
            crop_right = min(right, w)
            crop_bottom = min(bottom, h)

            crop = img.crop((left, top, crop_right, crop_bottom))

            # create padded tile of exact tile_size
            tile = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
            tile.paste(crop, (0, 0))

            tile_name = f"{Path(image_name).stem}_x{left}_y{top}{Path(image_name).suffix}"
            tile_path = output_dir / "images" / tile_name
            tile_path.parent.mkdir(parents=True, exist_ok=True)

            # determine which annotations intersect this tile
            tile_annotations = []

            # handle bounding boxes
            if bbox_columns and all(col in annotations.columns for col in bbox_columns):
                x1_col, y1_col, x2_col, y2_col = bbox_columns
                for _, r in annotations.iterrows():
                    x1, y1, x2, y2 = float(r[x1_col]), float(r[y1_col]), float(r[x2_col]), float(r[y2_col])
                    ix1 = max(x1, left)
                    iy1 = max(y1, top)
                    ix2 = min(x2, left + tile_size)
                    iy2 = min(y2, top + tile_size)
                    if ix2 > ix1 and iy2 > iy1:
                        # adjust coordinates relative to tile origin
                        nx1 = int(ix1 - left)
                        ny1 = int(iy1 - top)
                        nx2 = int(ix2 - left)
                        ny2 = int(iy2 - top)
                        tile_annotations.append((nx1, ny1, nx2, ny2, r))

            # handle points (keep as point annotations)
            elif point_columns and all(col in annotations.columns for col in point_columns):
                x_col, y_col = point_columns
                for _, r in annotations.iterrows():
                    x, y = float(r[x_col]), float(r[y_col])
                    if left <= x < left + tile_size and top <= y < top + tile_size:
                        nx = int(x - left)
                        ny = int(y - top)
                        tile_annotations.append((nx, ny, r))

            # decide whether to save tile
            save_tile = len(tile_annotations) > 0 or (split == "test") or save_empty_tiles
            if save_tile:
                tile.save(tile_path)

                # append annotation rows for this tile
                for annotation in tile_annotations:
                    if len(annotation) == 5:
                        nx1, ny1, nx2, ny2, row = annotation
                        df_row = {
                            "image_path": tile_name,
                            "x1": nx1,
                            "y1": ny1,
                            "x2": nx2,
                            "y2": ny2,
                            "species": row.get("species", "unknown"),
                            "split": split,
                        }
                    else:
                        nx, ny, row = annotation
                        df_row = {
                            "image_path": tile_name,
                            "x": nx,
                            "y": ny,
                            "species": row.get("species", "unknown"),
                            "split": split,
                        }

                    tiled_rows.append(df_row)

    return tiled_rows
