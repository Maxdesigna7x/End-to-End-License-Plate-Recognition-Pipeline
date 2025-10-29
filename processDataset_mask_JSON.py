#!/usr/bin/env python3
"""
Crop images and their masks using a single JSON (per image) containing per-char bboxes (bbox_xywh).
For each image + json:
 - merge all per-char boxes into a single polygon
 - compute its minimum rotated rectangle
 - crop & warp the image to that rectangle (rectified)
 - apply the same transform to the corresponding mask (basename_mask.png) using INTER_NEAREST
 - save outputs to out_dir as basename_crop.jpg and basename_mask_crop.png
"""
import os
import json
import math
from typing import List, Tuple, Optional

import numpy as np
import cv2
from shapely.geometry import Polygon
from tqdm import tqdm

# ---------- Helpers ----------

def get_image_json_pairs(data_dir: str) -> List[Tuple[str, str, Optional[str]]]:
    """Return list of (img_path, json_path, mask_path_or_None) for files in data_dir.
       We consider images with extensions .jpg/.jpeg/.png but ignore files ending with _mask.* as main images.
    """
    imgs = []
    exts = ('.jpg', '.jpeg', '.png')
    for fn in sorted(os.listdir(data_dir)):
        lower = fn.lower()
        if lower.endswith(exts) and not lower.endswith('_mask.png') and not lower.endswith('_mask.jpg') and not lower.endswith('_mask.jpeg'):
            base, _ = os.path.splitext(fn)
            img_path = os.path.join(data_dir, fn)
            json_path = os.path.join(data_dir, base + '.json')
            if not os.path.exists(json_path):
                continue
            mask_candidates = [
                os.path.join(data_dir, base + '_mask.png'),
                os.path.join(data_dir, base + '_mask.jpg'),
                os.path.join(data_dir, base + '_mask.jpeg'),
            ]
            mask_path = None
            for mc in mask_candidates:
                if os.path.exists(mc):
                    mask_path = mc
                    break
            imgs.append((img_path, json_path, mask_path))
    return imgs

def load_json_boxes(json_path: str) -> List[Tuple[float, float, float, float]]:
    """Load list of bbox_xywh from the JSON file. Returns list of (x,y,w,h).
       Supports JSON structures where root is list of objects having 'bbox_xywh'.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    boxes = []
    # if JSON is dict with a top-level list under a key, try to find it
    if isinstance(data, dict):
        # heuristics: find first list value holding dicts with bbox_xywh
        found = False
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict) and 'bbox_xywh' in v[0]:
                data = v
                found = True
                break
        if not found:
            # fallback: try to read 'annotations' or 'objects'
            for key in ('annotations', 'objects', 'items'):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break

    if isinstance(data, list):
        for obj in data:
            if not isinstance(obj, dict):
                continue
            if 'bbox_xywh' in obj:
                x, y, w, h = obj['bbox_xywh']
                boxes.append((float(x), float(y), float(w), float(h)))
            # fallback if bbox stored as 'bbox' or similar
            elif 'bbox' in obj and isinstance(obj['bbox'], (list, tuple)) and len(obj['bbox']) >= 4:
                x, y, w, h = obj['bbox'][:4]
                boxes.append((float(x), float(y), float(w), float(h)))
    return boxes

def merged_polygon_from_boxes(boxes: List[Tuple[float, float, float, float]]) -> Optional[Polygon]:
    """Given list of (x,y,w,h), produce merged polygon (shapely)."""
    polys = []
    for (x,y,w,h) in boxes:
        if w <= 0 or h <= 0:
            continue
        corners = [(x,y), (x + w, y), (x + w, y + h), (x, y + h)]
        polys.append(Polygon(corners))
    if not polys:
        return None
    merged = polys[0]
    for p in polys[1:]:
        merged = merged.union(p)
    return merged

def min_rect_params_from_polygon(poly: Polygon) -> Tuple[float,float,float,float,float]:
    """Return (cx, cy, width, height, angle_rad) of the minimum rotated rectangle around 'poly'.
       Angle is in radians in range [-pi/2, pi/2].
    """
    min_rect = poly.minimum_rotated_rectangle
    coords = np.array(min_rect.exterior.coords[:-1], dtype=np.float32)  # 4x2, but order might vary
    # compute center as mean
    cx = float(coords[:,0].mean())
    cy = float(coords[:,1].mean())

    # compute edge lengths
    edge01 = np.linalg.norm(coords[0] - coords[1])
    edge12 = np.linalg.norm(coords[1] - coords[2])
    if edge01 >= edge12:
        width = float(edge01)
        height = float(edge12)
        angle_rad = math.atan2(coords[1,1] - coords[0,1], coords[1,0] - coords[0,0])
    else:
        width = float(edge12)
        height = float(edge01)
        angle_rad = math.atan2(coords[2,1] - coords[1,1], coords[2,0] - coords[1,0])

    # normalize angle to [-pi/2, pi/2]
    if angle_rad > math.pi/2:
        angle_rad -= math.pi
    if angle_rad < -math.pi/2:
        angle_rad += math.pi

    return cx, cy, width, height, angle_rad

def bbox_to_corners(cx: float, cy: float, w: float, h: float, angle: float) -> np.ndarray:
    """Return corners (tl, tr, br, bl) as float32 array shape (4,2)."""
    corners = np.array([
        [-w/2.0, -h/2.0],
        [ w/2.0, -h/2.0],
        [ w/2.0,  h/2.0],
        [-w/2.0,  h/2.0],
    ], dtype=np.float32)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    R = np.array([[cos_a, -sin_a],
                  [sin_a,  cos_a]], dtype=np.float32)
    rotated = corners @ R.T
    translated = rotated + np.array([cx, cy], dtype=np.float32)
    return translated

def order_corners_clockwise(corners: np.ndarray) -> np.ndarray:
    """Return corners ordered clockwise starting at top-left (approx) given arbitrary 4 points.
       corners: (4,2) numpy array.
    """
    c = corners.copy()
    center = c.mean(axis=0)
    angles = np.arctan2(c[:,1] - center[1], c[:,0] - center[0])
    order = np.argsort(angles)  # increasing angle
    c_sorted = c[order]
    # ensure start at top-left (min x+y)
    sums = c_sorted.sum(axis=1)
    start = int(np.argmin(sums))
    c_reordered = np.roll(c_sorted, -start, axis=0)
    # after roll, ensure order is [tl, tr, br, bl] clockwise
    return c_reordered.astype(np.float32)

def crop_and_warp(img: np.ndarray, src_corners: np.ndarray, out_w: int, out_h: int, interp=cv2.INTER_LINEAR, borderValue=(0,0,0)) -> np.ndarray:
    """Warp src_corners (tl,tr,br,bl) to rectangle (out_w, out_h). img is np.uint8 HxWxC or HxW."""
    if src_corners.shape != (4,2):
        raise ValueError("src_corners must be shape (4,2)")
    src = np.array(src_corners, dtype=np.float32)
    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (int(out_w), int(out_h)), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=borderValue)
    return warped

# ---------- Main processing ----------

def process_one_pair(img_path: str,
                     json_path: str,
                     mask_path: Optional[str],
                     out_dir: str,
                     padding: int = 0,
                     out_size: Optional[Tuple[int,int]] = None) -> Optional[Tuple[str,str]]:
    """Process one triple. Returns (out_img_path, out_mask_path_or_None) or None on error."""
    try:
        # load image (color)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"ERROR: cannot read image {img_path}")
            return None
        h_img, w_img = img.shape[:2]

        boxes = load_json_boxes(json_path)
        if not boxes:
            print(f"WARNING: no boxes found in {json_path} -> skipping")
            return None

        merged = merged_polygon_from_boxes(boxes)
        if merged is None or merged.is_empty:
            print(f"WARNING: merged polygon empty for {json_path} -> skipping")
            return None

        cx, cy, width, height, angle = min_rect_params_from_polygon(merged)

        # apply padding in pixels
        width_p = max(1.0, width + 2.0 * float(padding))
        height_p = max(1.0, height + 2.0 * float(padding))

        # recompute corners from padded bbox (tl,tr,br,bl)
        corners = bbox_to_corners(cx, cy, width_p, height_p, angle)
        corners_ordered = order_corners_clockwise(corners)

        # determine output size: if out_size provided use it, else use the integer width/height of bbox (rounded)
        if out_size is not None:
            out_w, out_h = out_size
        else:
            # approximate size from edge lengths
            edge_w = np.linalg.norm(corners_ordered[0] - corners_ordered[1])
            edge_h = np.linalg.norm(corners_ordered[1] - corners_ordered[2])
            out_w = max(1, int(round(edge_w)))
            out_h = max(1, int(round(edge_h)))

        # crop & warp image (linear interpolation)
        warped = crop_and_warp(img, corners_ordered, out_w, out_h, interp=cv2.INTER_LINEAR, borderValue=(128,128,128))

        # process mask if exists
        warped_mask = None
        if mask_path:
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                print(f"WARNING: cannot read mask {mask_path} -> skipping mask")
            else:
                # if mask has multiple channels, convert to single-channel grayscale
                if len(mask.shape) == 3:
                    # if it's RGBA, take alpha if likely; else convert to gray
                    if mask.shape[2] == 4:
                        mask_single = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
                    else:
                        mask_single = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                else:
                    mask_single = mask
                warped_mask = crop_and_warp(mask_single, corners_ordered, out_w, out_h, interp=cv2.INTER_NEAREST, borderValue=255)

        # prepare output filenames
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_img_path = os.path.join(out_dir, f"{base}.jpg")
        cv2.imwrite(out_img_path, warped)

        out_mask_path = None
        if warped_mask is not None:
            out_mask_path = os.path.join(out_dir, f"{base}_clean.png")
            # ensure mask is single-channel 0..255
            if len(warped_mask.shape) == 3:
                warped_mask = cv2.cvtColor(warped_mask, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(out_mask_path, warped_mask)

        return out_img_path, out_mask_path

    except Exception as e:
        print(f"ERROR processing {img_path}: {e}")
        return None

def process_directory(data_dir: str, out_dir: str, padding: int = 0, out_size: Optional[Tuple[int,int]] = None, max_files: Optional[int]=None):
    os.makedirs(out_dir, exist_ok=True)
    pairs = get_image_json_pairs(data_dir)
    if not pairs:
        print("No image/json pairs found.")
        return

    total = len(pairs) if max_files is None else min(len(pairs), max_files)
    for i, (img_path, json_path, mask_path) in enumerate(tqdm(pairs[:total], desc="Processing")):
        res = process_one_pair(img_path, json_path, mask_path, out_dir, padding=padding, out_size=out_size)
        if res is None:
            continue
    print("Done.")

# ---------- MAIN: ejemplo listo para 'darle play' ----------
if __name__ == "__main__":
    import argparse
    import sys

    # Si se pasan argumentos por CLI, utilízalos; si no, usa valores de ejemplo (placeholder) para ejecución directa.
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Crop images + masks using JSON per-letter boxes (bbox_xywh).")
        parser.add_argument("--data_dir", required=True, help="Directory with images, jsons and masks (_mask.png).")
        parser.add_argument("--out_dir", required=True, help="Output directory for crops.")
        parser.add_argument("--padding", type=int, default=0, help="Padding in pixels to add around the merged box.")
        parser.add_argument("--out_width", type=int, default=0, help="If set (>0) force output width.")
        parser.add_argument("--out_height", type=int, default=0, help="If set (>0) force output height.")
        parser.add_argument("--max_files", type=int, default=0, help="Limit number of files to process (0 = all).")
        args = parser.parse_args()

        out_size = None
        if args.out_width > 0 and args.out_height > 0:
            out_size = (args.out_width, args.out_height)

        max_files = args.max_files if args.max_files > 0 else None

        process_directory(args.data_dir, args.out_dir, padding=args.padding, out_size=out_size, max_files=max_files)

    else:
        # EJEMPLO listo para darle PLAY: modifica estas rutas si quieres, o simplemente deja que cree carpetas de ejemplo.
        print("No command-line arguments detected — ejecutando ejemplo por defecto.")
        example_data_dir = r"C:\Universidad\Seminario\Roberto_Plates\V6\datasets\plate_denoise"   # <- Cambia esto si tu carpeta está en otra ruta
        example_out_dir  = r"C:\Universidad\Seminario\Roberto_Plates\V6\datasets\plate_denoise_crop"            # <- Carpeta de salida por defecto
        example_padding  = 10                          # padding en píxeles alrededor de la caja combinada
        example_out_size = (400, 200)                  # fuerza el tamaño de salida (width, height). Usa None para tamaño natural
        example_max_files = None                       # procesa todo. Pon un número para probar un subconjunto (ej. 10)

        print(f"Usando data_dir = '{example_data_dir}'")
        print(f"Usando out_dir  = '{example_out_dir}'")
        print(f"Padding = {example_padding} px, out_size = {example_out_size}, max_files = {example_max_files}")
        os.makedirs(example_out_dir, exist_ok=True)

        process_directory(example_data_dir, example_out_dir, padding=example_padding, out_size=example_out_size, max_files=example_max_files)
