#!/usr/bin/env python3
"""
register_series.py
==================

Register a directory of drone orthomosaics while managing multispectral (MS)
and infrared (IR) pairs. All outputs use the reference MS image geometry.

Expected filenames: ``<prefix>_<MS|IR>.tif``, for example
``2022_13h_STR_MS.tif`` and ``2022_13h_STR_IR.tif``. Each shared prefix is one
scene. MS images are matched to MS images and IR images to IR images.

Registration topology:
    1. Every MS -> reference MS.
    2. Every IR -> reference IR.
    3. Reference IR -> reference MS using georeferencing only.
    4. Compose each IR mapping as IR -> IR_ref -> MS_ref.

All outputs use the reference MS grid. IR images are upsampled, and MS images
are written one band per file under BANDS.

Pairwise registration uses SIFT/RANSAC tie points on an optional pyramid,
an optional `Reference_point` anchor, and georeferencing as a safe fallback.
The selected rigid, affine, polynomial, or TPS model is applied once at full
resolution.

Usage:
        phenolapseUAV-register-series --indir DIRECTORY --outdir OUTPUT --pyramid 4,2,1
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

try:
    from register_orthomosaics import setup_logging, _percentile_normalize
except Exception:
    def setup_logging(debug: bool) -> None:
        level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(
            level=level,
            format="[%(asctime)s] %(levelname)-7s %(message)s",
            datefmt="%H:%M:%S", stream=sys.stdout, force=True)

    def _percentile_normalize(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = np.full_like(arr, np.nan, dtype=np.float32)
        if mask.sum() < 10:
            return out
        valid = arr[mask]
        p2, p98 = np.percentile(valid, [2, 98])
        if p98 - p2 < 1e-9:
            return out
        norm = np.clip((arr - p2) / (p98 - p2), 0.0, 1.0)
        out[mask] = norm[mask]
        return out


LOG = logging.getLogger("register_series")

IMAGE_EXTS = (".tif", ".tiff", ".TIF", ".TIFF")


# ---------------------------------------------------------------------------
# Image discovery and MS/IR grouping by scene
# ---------------------------------------------------------------------------
def list_images(indir: Path) -> List[Path]:
    imgs = [p for p in indir.iterdir()
            if p.is_file() and p.suffix in IMAGE_EXTS]
    imgs.sort(key=lambda p: p.name.lower())
    return imgs


def group_units(images: List[Path]) -> "Dict[str, Dict[str, Path]]":
    """Group images by scene name after removing the _MS or _IR suffix."""
    units: "Dict[str, Dict[str, Path]]" = {}
    for p in images:
        stem = p.stem
        low = stem.lower()
        if low.endswith("_ms"):
            key, kind = stem[:-3], "MS"
        elif low.endswith("_ir"):
            key, kind = stem[:-3], "IR"
        else:
            key, kind = stem, "MS"  # Treat files without a suffix as MS.
        u = units.setdefault(key, {"MS": None, "IR": None})
        if u.get(kind) is not None:
            LOG.warning("Duplicate %s for scene %s: ignored %s.",
                        kind, key, p.name)
            continue
        u[kind] = p
    return units


def parse_reference_points(indir: Path) -> Dict[str, Tuple[float, float]]:
    """Read `Reference_point[.txt]` lines formatted as `image_name X Y`.

    X is the full-resolution column and Y is the full-resolution row.
    """
    cand = [p for p in indir.iterdir()
            if p.is_file() and p.stem.lower() == "reference_point"]
    if not cand:
        LOG.info("No 'Reference_point' file found; anchors are disabled.")
        return {}
    path = cand[0]
    LOG.info("Reference points: %s", path.name)
    pts: Dict[str, Tuple[float, float]] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 3:
            LOG.warning("  ignored line %d: %r", lineno, raw)
            continue
        try:
            x, y = float(parts[-2]), float(parts[-1])
        except ValueError:
            LOG.warning("  invalid X/Y values on line %d: %r", lineno, raw)
            continue
        pts[Path(parts[0]).name] = (x, y)
    LOG.info("  loaded %d point(s).", len(pts))
    return pts


# ---------------------------------------------------------------------------
# Downsampled reading and normalized working band
# ---------------------------------------------------------------------------
def read_norm_band_decimated(path: Path, band: Optional[int],
                             subsample: int) -> Dict[str, Any]:
    """Read a downsampled float32 band normalized to percentiles 2–98.

    When band is None, return the robust mean of all valid bands.
    """
    import rasterio
    from rasterio.enums import Resampling as RIOResampling

    with rasterio.open(path) as ds:
        full_w, full_h = ds.width, ds.height
        W = max(1, full_w // subsample)
        H = max(1, full_h // subsample)
        nodata = ds.nodata

        def _norm(idx: int) -> np.ndarray:
            a = ds.read(idx, out_shape=(H, W),
                        resampling=RIOResampling.average).astype(np.float32)
            m = np.isfinite(a)
            if nodata is not None:
                m &= a != nodata
            return _percentile_normalize(a, m)

        eff_band = band if (band is not None and band <= ds.count) else None
        if band is not None and band > ds.count:
            LOG.warning("  %s: band %d exceeds %d bands; using robust mean.",
                        path.name, band, ds.count)
        if eff_band is not None:
            arr = _norm(eff_band)
            desc = f"band #{eff_band}"
        else:
            bands = []
            for i in range(1, ds.count + 1):
                a = ds.read(i, out_shape=(H, W),
                            resampling=RIOResampling.average).astype(np.float32)
                m = np.isfinite(a)
                if nodata is not None:
                    m &= a != nodata
                if m.sum() < 50:
                    continue
                bands.append(_percentile_normalize(a, m))
            if not bands:
                raise RuntimeError(f"No usable band in {path.name}")
            arr = np.nanmean(np.stack(bands, 0), 0).astype(np.float32)
            desc = f"robust mean ({len(bands)}/{ds.count} bands)"

        return {
            "arr": arr,
            "scale_x": full_w / float(W),
            "scale_y": full_h / float(H),
            "transform": ds.transform,
            "crs": ds.crs,
            "full_shape": (full_h, full_w),
            "nodata": nodata,
            "count": ds.count,
            "desc": desc,
        }


def _to_u8(a: np.ndarray) -> np.ndarray:
    x = np.clip(np.nan_to_num(a, nan=0.0), 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Warp models: destination-grid pixels -> source-image pixels
# ---------------------------------------------------------------------------
class AffineModel:
    def __init__(self, M: np.ndarray):
        self.M = M.astype(np.float64)

    def predict(self, cols, rows):
        a, b, c = self.M[0]
        d, e, f = self.M[1]
        return a * cols + b * rows + c, d * cols + e * rows + f


def _poly2_features(cols, rows):
    return np.stack([np.ones_like(cols), cols, rows,
                     cols * cols, cols * rows, rows * rows], axis=-1)


class Poly2Model:
    def __init__(self, cx, cy):
        self.cx, self.cy = cx, cy

    def predict(self, cols, rows):
        feats = _poly2_features(cols.astype(np.float64), rows.astype(np.float64))
        return feats @ self.cx, feats @ self.cy


class TPSModel:
    def __init__(self, tps):
        self._tps = tps

    def predict(self, cols, rows):
        pts = np.stack([cols.ravel(), rows.ravel()], axis=-1).astype(np.float64)
        out = self._tps(pts)
        return (out[:, 0].reshape(cols.shape), out[:, 1].reshape(cols.shape))


class GeoAlignModel:
    """Geometric alignment through raster geotransforms.

    Destination pixel -> map coordinate -> source pixel.
    """

    def __init__(self, dest_transform, src_transform):
        self.d = dest_transform
        self.si = ~src_transform

    def predict(self, cols, rows):
        d, si = self.d, self.si
        mx = d.a * cols + d.b * rows + d.c
        my = d.d * cols + d.e * rows + d.f
        return (si.a * mx + si.b * my + si.c, si.d * mx + si.e * my + si.f)


class ComposedModel:
    """Map destination -> outer -> intermediate -> inner -> source."""

    def __init__(self, outer, inner):
        self.outer, self.inner = outer, inner

    def predict(self, cols, rows):
        mc, mr = self.outer.predict(cols, rows)
        return self.inner.predict(mc, mr)


class OffsetModel:
    """Add a source-pixel translation to an existing model."""

    def __init__(self, base, dx: float, dy: float):
        self.base, self.dx, self.dy = base, dx, dy

    def predict(self, cols, rows):
        sx, sy = self.base.predict(cols, rows)
        return sx + self.dx, sy + self.dy


def _fit_poly2(dst: np.ndarray, src: np.ndarray, iters: int = 3) -> Poly2Model:
    cols, rows = dst[:, 0].astype(np.float64), dst[:, 1].astype(np.float64)
    sx, sy = src[:, 0].astype(np.float64), src[:, 1].astype(np.float64)
    keep = np.ones(len(dst), dtype=bool)
    cx = cy = None
    for _ in range(iters):
        A = _poly2_features(cols[keep], rows[keep])
        cx, _, _, _ = np.linalg.lstsq(A, sx[keep], rcond=None)
        cy, _, _, _ = np.linalg.lstsq(A, sy[keep], rcond=None)
        Aall = _poly2_features(cols, rows)
        res = np.hypot(sx - Aall @ cx, sy - Aall @ cy)
        mad = np.median(np.abs(res - np.median(res))) + 1e-9
        new_keep = res < (np.median(res) + 3.0 * 1.4826 * mad)
        if new_keep.sum() < 6 or new_keep.sum() == keep.sum():
            break
        keep = new_keep
    return Poly2Model(cx, cy)


# ---------------------------------------------------------------------------
# SIFT and RANSAC tie points
# ---------------------------------------------------------------------------
def sift_tiepoints(ref_arr: np.ndarray, mov_arr: np.ndarray
                   ) -> Optional[Dict[str, Any]]:
    import cv2
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    r8 = clahe.apply(_to_u8(ref_arr))
    m8 = clahe.apply(_to_u8(mov_arr))
    sift = cv2.SIFT_create(nfeatures=15000, contrastThreshold=0.015)
    kp_r, des_r = sift.detectAndCompute(r8, None)
    kp_m, des_m = sift.detectAndCompute(m8, None)
    LOG.info("      SIFT keypoints: ref=%d mov=%d", len(kp_r), len(kp_m))
    if des_r is None or des_m is None or len(kp_r) < 30 or len(kp_m) < 30:
        LOG.warning("      Not enough SIFT keypoints.")
        return None
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = bf.knnMatch(des_m, des_r, k=2)
    good = [m for m, n in knn if m.distance < 0.75 * n.distance]
    reverse_knn = bf.knnMatch(des_r, des_m, k=2)
    reverse_good = [m for m, n in reverse_knn
                    if m.distance < 0.75 * n.distance]
    reverse_pairs = {(m.trainIdx, m.queryIdx) for m in reverse_good}
    good = [m for m in good
            if (m.queryIdx, m.trainIdx) in reverse_pairs]
    LOG.info("      SIFT matches: mutual Lowe=%d / %d", len(good), len(knn))
    if len(good) < 12:
        LOG.warning("      Too few matches (%d).", len(good))
        return None
    pts_mov = np.float32([kp_m[g.queryIdx].pt for g in good])
    pts_ref = np.float32([kp_r[g.trainIdx].pt for g in good])
    M, inliers = cv2.estimateAffinePartial2D(
        pts_mov, pts_ref, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        maxIters=5000, confidence=0.999, refineIters=20)
    if M is None or inliers is None:
        LOG.warning("      RANSAC failed.")
        return None
    inl = inliers.ravel().astype(bool)
    n_in = int(inl.sum())
    LOG.info("      RANSAC inliers: %d / %d", n_in, len(good))
    if n_in < 8:
        return None
    return {"pts_ref_dec": pts_ref[inl], "pts_mov_dec": pts_mov[inl],
            "M_sim_dec": M, "n_inliers": n_in}


# ---------------------------------------------------------------------------
# Build the model: full-resolution reference pixels -> moving source pixels
# ---------------------------------------------------------------------------
def build_warp_model(tp, ref_scale, mov_scale, seed_corr, warp,
                     geoalign: Optional[GeoAlignModel] = None
                     ) -> Tuple[Any, str, int]:
    rsx, rsy = ref_scale
    msx, msy = mov_scale
    dst_pts: List[List[float]] = []
    src_pts: List[List[float]] = []

    if tp is not None:
        ref_full = tp["pts_ref_dec"] * np.array([rsx, rsy], np.float32)
        mov_full = tp["pts_mov_dec"] * np.array([msx, msy], np.float32)
        dst_pts.extend(ref_full.tolist())
        src_pts.extend(mov_full.tolist())

    if seed_corr is not None:
        (xr, yr), (xm, ym) = seed_corr
        for _ in range(max(3, len(dst_pts) // 20 + 1)):
            dst_pts.append([xr, yr]); src_pts.append([xm, ym])

    n = len(dst_pts)

    if tp is None:
        if seed_corr is not None:
            (xr, yr), (xm, ym) = seed_corr
            M = np.array([[1., 0., xm - xr], [0., 1., ym - yr]], np.float64)
            return AffineModel(M), "seed_translation", n
        if geoalign is not None:
            return geoalign, "geoalign(georef)", 0
        raise RuntimeError("No tie points, anchor, or georeferencing available.")

    dst = np.asarray(dst_pts, np.float64)
    src = np.asarray(src_pts, np.float64)

    import cv2
    M_inv = cv2.invertAffineTransform(tp["M_sim_dec"])  # ref_dec -> mov_dec
    S_ref_inv = np.array([[1. / rsx, 0, 0], [0, 1. / rsy, 0], [0, 0, 1]])
    S_mov = np.array([[msx, 0, 0], [0, msy, 0], [0, 0, 1]])
    M_full = (S_mov @ np.vstack([M_inv, [0, 0, 1]]) @ S_ref_inv)[:2, :]
    rigid_model = AffineModel(M_full)

    if warp == "rigid" or n < 12:
        return rigid_model, ("rigid" if warp == "rigid"
                             else "rigid(auto:too few points)"), n
    if warp == "affine":
        M_aff, _ = cv2.estimateAffine2D(
            dst.astype(np.float32), src.astype(np.float32),
            method=cv2.RANSAC, ransacReprojThreshold=3.0)
        return ((AffineModel(M_aff), "affine", n) if M_aff is not None
                else (rigid_model, "rigid(auto:affine failed)", n))
    if warp == "tps" and n >= 25:
        try:
            from skimage.transform import ThinPlateSplineTransform
            tps = None
            if hasattr(ThinPlateSplineTransform, "from_estimate"):
                res = ThinPlateSplineTransform.from_estimate(dst, src)
                if res:  # FailedEstimation is falsy.
                    tps = res
            else:
                t = ThinPlateSplineTransform()
                if t.estimate(dst, src):
                    tps = t
            if tps is not None:
                return TPSModel(tps), "tps", n
        except Exception as e:
            LOG.warning("      TPS unavailable or failed (%s) -> poly2.", e)
    return _fit_poly2(dst, src), "poly2", n


def _tiepoint_coverage(tp: Optional[Dict[str, Any]], shape: Tuple[int, int],
                       grid: int = 6) -> Tuple[bool, str]:
    """Check that inliers cover the image rather than a small repeated pattern."""
    if tp is None:
        return False, "no points"
    pts = tp["pts_ref_dec"]
    if len(pts) < 8:
        return False, f"{len(pts)} points"
    h, w = shape
    span_x = float(np.ptp(pts[:, 0]) / max(w, 1))
    span_y = float(np.ptp(pts[:, 1]) / max(h, 1))
    gx = np.clip((pts[:, 0] / max(w, 1) * grid).astype(int), 0, grid - 1)
    gy = np.clip((pts[:, 1] / max(h, 1) * grid).astype(int), 0, grid - 1)
    cells = len(set(zip(gx.tolist(), gy.tolist())))
    desc = (f"coverage={100 * span_x:.0f}%x{100 * span_y:.0f}% "
            f"cells={cells}/{grid * grid}")
    # Deliberately conservative threshold for repeated crop-row patterns.
    valid = span_x >= 0.30 and span_y >= 0.30 and cells >= 6
    return valid, desc


def _warp_preview_to_reference(mov_meta: Dict[str, Any],
                               ref_meta: Dict[str, Any], model: Any) -> np.ndarray:
    """Warp only the moving preview onto the downsampled reference grid."""
    import cv2
    h, w = ref_meta["arr"].shape
    cc, rr = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    ref_x = cc * ref_meta["scale_x"]
    ref_y = rr * ref_meta["scale_y"]
    mov_x, mov_y = model.predict(ref_x, ref_y)
    map_x = (mov_x / mov_meta["scale_x"]).astype(np.float32)
    map_y = (mov_y / mov_meta["scale_y"]).astype(np.float32)
    return cv2.remap(mov_meta["arr"], map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)


def _fit_control_model(dst: np.ndarray, src: np.ndarray, warp: str,
                       fallback: Any, threshold: float) -> Tuple[Any, str]:
    """Fit a full-resolution reference -> full-resolution source model."""
    import cv2
    if warp == "rigid":
        M, _ = cv2.estimateAffinePartial2D(
            dst.astype(np.float32), src.astype(np.float32), method=cv2.RANSAC,
            ransacReprojThreshold=threshold, maxIters=10000,
            confidence=0.999, refineIters=20)
        return (AffineModel(M), "rigid") if M is not None else (fallback, "retained")
    if warp == "affine":
        M, _ = cv2.estimateAffine2D(
            dst.astype(np.float32), src.astype(np.float32), method=cv2.RANSAC,
            ransacReprojThreshold=threshold, maxIters=10000,
            confidence=0.999, refineIters=20)
        return (AffineModel(M), "affine") if M is not None else (fallback, "retained")
    if warp == "tps" and len(dst) >= 25:
        try:
            from skimage.transform import ThinPlateSplineTransform
            if hasattr(ThinPlateSplineTransform, "from_estimate"):
                tps = ThinPlateSplineTransform.from_estimate(dst, src)
            else:
                tps = ThinPlateSplineTransform()
                if not tps.estimate(dst, src):
                    tps = None
            if tps is not None and bool(tps):
                return TPSModel(tps), "tps"
        except Exception as e:
            LOG.warning("      Pyramid TPS failed (%s) -> poly2.", e)
    if len(dst) >= 12:
        return _fit_poly2(dst, src), "poly2"
    return fallback, "retained"


def build_pyramid_model(ref_path: Path, mov_path: Path,
                        ref_band: Optional[int], mov_band: Optional[int],
                        levels: List[int], final_warp: str,
                        seed_corr=None) -> Tuple[Any, str, int]:
    """Register coarse to fine by composing models without rewriting GeoTIFFs.

    The first level estimates global geometry. At later levels, the current
    model prepositions the moving preview and residual tie points are mapped
    back into the original moving-image coordinates.
    """
    model = None
    details: List[str] = []
    total_points = 0

    for index, level in enumerate(levels):
        LOG.info("    [pyramid %d/%d] subsample=%d", index + 1,
                 len(levels), level)
        ref_meta = read_norm_band_decimated(ref_path, ref_band, level)
        mov_meta = read_norm_band_decimated(mov_path, mov_band, level)
        ref_scale = (ref_meta["scale_x"], ref_meta["scale_y"])
        mov_scale = (mov_meta["scale_x"], mov_meta["scale_y"])

        if model is None:
            model = GeoAlignModel(ref_meta["transform"], mov_meta["transform"])
            LOG.info("      Initial prepositioning: georeferencing")
            if seed_corr is not None:
                (xr, yr), (xm, ym) = seed_corr
                predicted_x, predicted_y = model.predict(xr, yr)
                model = OffsetModel(model, xm - float(predicted_x),
                                    ym - float(predicted_y))
                LOG.info("      Anchor applied to prepositioning.")

        warped = _warp_preview_to_reference(mov_meta, ref_meta, model)
        tp = sift_tiepoints(ref_meta["arr"], warped)
        valid, coverage = _tiepoint_coverage(tp, ref_meta["arr"].shape)
        LOG.info("      Residual distribution: %s", coverage)
        if tp is None or not valid:
            LOG.warning("      Rejected s%d refinement; previous model retained.",
                        level)
            details.append(f"s{level}:rejected")
            continue

        # pts_mov_dec uses the prepositioned preview and therefore the reference
        # grid. The current model maps it back into the source image.
        dst = tp["pts_ref_dec"].astype(np.float64) * np.array(ref_scale)
        previous_ref = (tp["pts_mov_dec"].astype(np.float64) *
                        np.array(ref_scale))
        sx, sy = model.predict(previous_ref[:, 0], previous_ref[:, 1])
        src = np.column_stack([sx, sy])
        if index == 0 and len(levels) > 1:
            stage_warp = "rigid"
        elif index == len(levels) - 1:
            stage_warp = final_warp
        else:
            stage_warp = "affine"
        refined, used = _fit_control_model(
            dst, src, stage_warp, model, threshold=3.0 * level)
        model = refined
        npts = int(tp["n_inliers"])
        total_points += npts
        details.append(f"s{level}:{used}/{npts}")
        LOG.info("      Accepted level %d: %s (%d points)", level, used, npts)

    return model, ",".join(details), total_points


# ---------------------------------------------------------------------------
# Apply a model and write onto the reference MS grid
# ---------------------------------------------------------------------------
def _model_coverage(model: Any, ref_shape: Tuple[int, int],
                    src_shape: Tuple[int, int], block: int = 512) -> float:
    """Return the fraction of reference-grid pixels falling inside the source."""
    ref_h, ref_w = ref_shape
    src_h, src_w = src_shape
    valid = 0
    for r0 in range(0, ref_h, block):
        r1 = min(r0 + block, ref_h)
        cc, rr = np.meshgrid(np.arange(ref_w, dtype=np.float64),
                             np.arange(r0, r1, dtype=np.float64))
        sx, sy = model.predict(cc, rr)
        valid += int(((sx >= 0) & (sx < src_w - 1) &
                      (sy >= 0) & (sy < src_h - 1)).sum())
    return valid / float(ref_h * ref_w)


def warp_target(target_path: Path, model: Any, ref_meta: Dict[str, Any],
                out_path: Path, resample: str = "cubic",
                split_bands: bool = False) -> List[Path]:
    import cv2
    import rasterio
    from rasterio.windows import Window

    ref_h, ref_w = ref_meta["full_shape"]
    interp = {"cubic": cv2.INTER_CUBIC, "linear": cv2.INTER_LINEAR,
              "nearest": cv2.INTER_NEAREST}[resample]
    block = 512

    with rasterio.open(target_path) as ds:
        count, dtype = ds.count, ds.dtypes[0]
        nodata = ds.nodata if ds.nodata is not None else 0
        descriptions = ds.descriptions
        coverage = _model_coverage(model, (ref_h, ref_w),
                                   (ds.height, ds.width), block)
        LOG.info("    Expected geometric coverage: %.2f%%", 100 * coverage)
        if coverage < 0.01:
            raise RuntimeError(
                f"model rejected before writing: coverage {100 * coverage:.4f}%")
        profile = {
            "driver": "GTiff", "width": ref_w, "height": ref_h,
            "count": 1, "dtype": dtype, "crs": ref_meta["crs"],
            "transform": ref_meta["transform"], "tiled": True,
            "compress": "deflate", "predictor": 2,
            "blockxsize": 512, "blockysize": 512, "nodata": nodata,
        }
        written: List[Path] = []
        for bi in range(1, count + 1):
            if split_bands:
                band_path = out_path.parent / (
                    f"{target_path.stem}_B{bi:02d}_registered.tif")
            else:
                band_path = out_path
            LOG.info("      Writing band %d/%d -> %s", bi, count,
                     band_path.name)
            band = ds.read(bi)
            with rasterio.open(band_path, "w", **profile) as dst:
                for r0 in range(0, ref_h, block):
                    r1 = min(r0 + block, ref_h)
                    cc, rr = np.meshgrid(
                        np.arange(ref_w, dtype=np.float64),
                        np.arange(r0, r1, dtype=np.float64))
                    sx, sy = model.predict(cc, rr)
                    remapped = cv2.remap(
                        band, sx.astype(np.float32), sy.astype(np.float32),
                        interp, borderMode=cv2.BORDER_CONSTANT,
                        borderValue=float(nodata))
                    dst.write(remapped, 1,
                              window=Window(0, r0, ref_w, r1 - r0))
                if descriptions[bi - 1]:
                    dst.set_band_description(1, descriptions[bi - 1])
                dst.update_tags(REG_TOOL="register_series",
                                REG_ONTO="reference_MS_grid",
                                SOURCE_BAND=str(bi))
            written.append(band_path)
        return written


def write_reference_bands(ref_path: Path, bands_dir: Path) -> List[Path]:
    """Extract each reference MS band without resampling."""
    import rasterio
    written: List[Path] = []
    with rasterio.open(ref_path) as src:
        profile = src.profile.copy()
        profile.update(count=1, tiled=True, compress="deflate", predictor=2,
                       blockxsize=512, blockysize=512)
        for bi in range(1, src.count + 1):
            out = bands_dir / f"{ref_path.stem}_B{bi:02d}_registered.tif"
            LOG.info("  Extracting reference band %d/%d -> %s", bi,
                     src.count, out.name)
            with rasterio.open(out, "w", **profile) as dst:
                for _, window in src.block_windows(bi):
                    dst.write(src.read(bi, window=window), 1, window=window)
                if src.descriptions[bi - 1]:
                    dst.set_band_description(1, src.descriptions[bi - 1])
                dst.update_tags(REG_TOOL="register_series",
                                REG_ONTO="reference_MS_grid",
                                REFERENCE="true", SOURCE_BAND=str(bi))
            written.append(out)
    return written


# ---------------------------------------------------------------------------
# Interactive selection
# ---------------------------------------------------------------------------
def _show_montage(arrays, titles, suptitle) -> None:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    n = len(arrays)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, arr, title in zip(axes, arrays, titles):
        ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(suptitle + "\n(close this window to choose)", fontsize=11)
    fig.tight_layout()
    plt.show()


def _tk_choose(title, prompt, options) -> int:
    import tkinter as tk
    root = tk.Tk()
    root.title(title)
    root.geometry("+200+200")
    var = tk.IntVar(value=0)
    tk.Label(root, text=prompt, font=("", 11, "bold"),
             justify="left").pack(anchor="w", padx=12, pady=(12, 6))
    frame = tk.Frame(root); frame.pack(fill="both", padx=12)
    for i, opt in enumerate(options):
        tk.Radiobutton(frame, text=opt, variable=var, value=i,
                       anchor="w").pack(fill="x", anchor="w")
    result = {"i": 0}

    def _ok():
        result["i"] = var.get(); root.destroy()

    tk.Button(root, text="Confirm", command=_ok, width=14).pack(pady=12)
    root.bind("<Return>", lambda e: _ok())
    root.mainloop()
    return result["i"]


def choose_reference_gui(ms_images: List[Path], display_subsample: int) -> Path:
    LOG.info("Interactive reference MS selection (subsample=%d)...",
             display_subsample)
    arrays, titles = [], []
    for idx, p in enumerate(ms_images):
        try:
            arrays.append(read_norm_band_decimated(p, None, display_subsample)["arr"])
        except Exception as e:
            LOG.warning("  could not generate preview for %s: %s", p.name, e)
            arrays.append(np.zeros((10, 10), np.float32))
        titles.append(f"[{idx}] {p.name}")
    _show_montage(arrays, titles, "Choose the REFERENCE MS image")
    i = _tk_choose("Reference MS", "Select the reference MS image:",
                   [f"[{k}] {p.name}" for k, p in enumerate(ms_images)])
    LOG.info("Selected reference: %s", ms_images[i].name)
    return ms_images[i]


def choose_band_gui(ref_path: Path, display_subsample: int) -> Optional[int]:
    import rasterio
    from rasterio.enums import Resampling as RIOResampling
    with rasterio.open(ref_path) as ds:
        count = ds.count
        W = max(1, ds.width // display_subsample)
        H = max(1, ds.height // display_subsample)
        nodata = ds.nodata
        arrays = []
        for i in range(1, count + 1):
            a = ds.read(i, out_shape=(H, W),
                        resampling=RIOResampling.average).astype(np.float32)
            m = np.isfinite(a)
            if nodata is not None:
                m &= a != nodata
            arrays.append(_percentile_normalize(a, m))
    _show_montage(arrays, [f"band {i}" for i in range(1, count + 1)],
                  f"Reference channels ({ref_path.name})")
    options = ["Robust mean (all bands)"] + \
              [f"band {i}" for i in range(1, count + 1)]
    i = _tk_choose("Matching band",
                   "Select the band used for registration:", options)
    if i == 0:
        LOG.info("Matching band: robust mean.")
        return None
    LOG.info("Matching band: %d", i)
    return i


def seed_correspondence(ref_name, mov_name, points):
    if ref_name not in points or mov_name not in points:
        return None
    return points[ref_name], points[mov_name]


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--reference", default=None,
                   help="Reference MS filename (e.g. 2022_13h_STR_MS.tif).")
    p.add_argument("--band", type=int, default=None,
                   help="1-based MS matching band. Interactive when omitted.")
    p.add_argument("--subsample", type=int, choices=[1, 2, 3, 4], default=1,
                   help="Single-pass downsampling (ignored with --pyramid).")
    p.add_argument("--pyramid", default=None, metavar="LEVELS",
                   help="Coarse-to-fine pyramid, e.g. 4,2,1. Native output.")
    p.add_argument("--warp", choices=["rigid", "affine", "poly2", "tps"],
                   default="tps",
                   help="Warp model (default: lightweight, smooth TPS).")
    p.add_argument("--resample", choices=["cubic", "linear", "nearest"],
                   default="cubic")
    p.add_argument("--display-subsample", type=int, default=4)
    p.add_argument("--limit", type=int, default=0,
                    help="Process only the first N moving scenes "
                        "(0 processes all scenes; useful for testing).")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def _parse_levels(pyramid: Optional[str], subsample: int) -> List[int]:
    if not pyramid:
        return [subsample]
    try:
        levels = [int(value.strip()) for value in pyramid.split(",")
                  if value.strip()]
    except ValueError as e:
        raise ValueError("--pyramid expects comma-separated integers") from e
    if not levels or any(level not in (1, 2, 3, 4) for level in levels):
        raise ValueError("--pyramid levels must be selected from 1,2,3,4")
    if any(a <= b for a, b in zip(levels, levels[1:])):
        raise ValueError("--pyramid must be strictly decreasing (e.g. 4,2,1)")
    return levels


def _fmt_eta(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def main() -> int:
    args = parse_args()
    setup_logging(args.debug)

    try:
        levels = _parse_levels(args.pyramid, args.subsample)
    except ValueError as e:
        LOG.error("Invalid option: %s", e)
        return 2

    indir, outdir = Path(args.indir), Path(args.outdir)
    bands_dir = outdir / "BANDS"
    LOG.info("=" * 72)
    LOG.info("register_series - MS/IR series -> reference MS geometry")
    LOG.info("=" * 72)
    LOG.info("indir=%s | outdir=%s", indir, outdir)
    LOG.info("pyramid=%s | final warp=%s | resampling=%s",
             "->".join(map(str, levels)), args.warp, args.resample)

    if not indir.is_dir():
        LOG.error("Directory not found: %s", indir); return 2
    bands_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("Single-band raster outputs: %s", bands_dir)

    images = list_images(indir)
    units = group_units(images)
    LOG.info("%d file(s), %d scene(s):", len(images), len(units))
    for key, u in units.items():
        LOG.info("  %-24s MS=%s IR=%s", key,
                 u["MS"].name if u["MS"] else "-",
                 u["IR"].name if u["IR"] else "-")

    ms_images = [u["MS"] for u in units.values() if u["MS"] is not None]
    if len(ms_images) < 2:
        LOG.error("At least two MS images are required."); return 3

    points = parse_reference_points(indir)

    # --- Reference MS ---
    if args.reference:
        matches = [p for p in ms_images if p.name == args.reference]
        if not matches:
            LOG.error("Reference MS '%s' was not found.", args.reference); return 4
        ref_ms = matches[0]
        LOG.info("Reference MS (option): %s", ref_ms.name)
    else:
        ref_ms = choose_reference_gui(ms_images, args.display_subsample)
    ref_key = next(k for k, u in units.items() if u["MS"] == ref_ms)
    ref_ir = units[ref_key]["IR"]

    # --- MS matching band ---
    band = args.band if args.band is not None else \
        choose_band_gui(ref_ms, args.display_subsample)

    # --- Read downsampled references ---
    LOG.info("Reading reference MS metadata (level %d)...", levels[0])
    ref_ms_meta = read_norm_band_decimated(ref_ms, band, levels[0])
    LOG.info("  MS ref: %s | full-resolution grid %dx%d", ref_ms_meta["desc"],
             ref_ms_meta["full_shape"][1], ref_ms_meta["full_shape"][0])
    model_refIR2MS = None
    ref_ir_meta = None
    if ref_ir is not None:
        LOG.info("Reading reference IR metadata...")
        ref_ir_meta = read_norm_band_decimated(ref_ir, None, levels[0])
        LOG.info(">>> IR_ref -> MS_ref bridge using georeferencing only")
        LOG.info("    %s -> %s grid", ref_ir.name, ref_ms.name)
        model_refIR2MS = GeoAlignModel(
            ref_ms_meta["transform"], ref_ir_meta["transform"])
        warp_used, npts = "geoalign(forced georeferencing)", 0
        LOG.info("    Bridge model: %s", warp_used)
    else:
        LOG.warning("The reference scene has no IR image; IR images will be "
                "placed directly on the reference MS by georeferencing.")

    # --- Count jobs for ETA ---
    movers = [(k, u) for k, u in units.items() if k != ref_key]
    if args.limit and args.limit > 0:
        movers = movers[:args.limit]
        LOG.info("Active limit: processing %d moving scene(s).", len(movers))
    n_jobs = (1 if ref_ir is not None else 0)
    for _, u in movers:
        n_jobs += (1 if u["MS"] else 0) + (1 if u["IR"] else 0)
    LOG.info("%d image(s) to resample, plus reference MS extraction.",
             n_jobs)

    report: List[str] = ["REGISTER_SERIES - report",
                         f"reference MS : {ref_ms.name}",
                         f"reference IR : {ref_ir.name if ref_ir else '-'}",
                         f"band         : {band if band else 'robust mean'}",
                         f"pyramid      : {'->'.join(map(str, levels))}",
                         f"warp final   : {args.warp}", ""]

    # --- Extract reference MS bands without resampling ---
    ref_outputs = write_reference_bands(ref_ms, bands_dir)
    LOG.info("Reference MS extracted without resampling: %d band(s)",
             len(ref_outputs))
    report.append(f"[REF ] {ref_ms.name} -> {len(ref_outputs)} band(s) in BANDS")

    t_start = time.time()
    times: List[float] = []
    done = 0
    n_ok = 0

    def _progress() -> str:
        if times:
            eta = (sum(times) / len(times)) * (n_jobs - done)
            return f" | ETA ~ {_fmt_eta(eta)}"
        return ""

    def _run_job(label: str, target: Path, model, out_path: Path,
                 info: str, split_bands: bool = False) -> None:
        nonlocal done, n_ok
        done += 1
        LOG.info("-" * 72)
        LOG.info(">>> [%d/%d] %s : %s%s", done, n_jobs, label, target.name,
                 _progress())
        t0 = time.time()
        try:
            outputs = warp_target(target, model, ref_ms_meta, out_path,
                                  args.resample, split_bands=split_bands)
            dt = time.time() - t0
            times.append(dt); n_ok += 1
            LOG.info("    -> %d file(s) in BANDS (%.1fs)", len(outputs), dt)
            report.append(f"[ OK ] {target.name} -> {len(outputs)} file(s) "
                          f"in BANDS | {info} | {dt:.1f}s")
        except Exception as e:
            LOG.exception("    FAILED %s: %s", target.name, e)
            report.append(f"[FAIL] {target.name} : {e}")

    # --- Reference IR -> reference MS grid ---
    if ref_ir is not None:
        _run_job("IR_ref -> MS_ref (georeferencing)", ref_ir,
                 model_refIR2MS,
                 bands_dir / f"{ref_ir.stem}_registered.tif",
                 "bridge=IR_ref->MS_ref by georeferencing")

    # --- Moving scenes ---
    for key, u in movers:
        # MS -> MS_ref
        if u["MS"] is not None:
            mov = u["MS"]
            LOG.info("=" * 72)
            LOG.info("Scene %s: matching MS -> MS_ref", key)
            seed = seed_correspondence(ref_ms.name, mov.name, points)
            if seed:
                LOG.info("      Using Reference_point anchor.")
            model_ms, wu, npts = build_pyramid_model(
                ref_ms, mov, band, band, levels, args.warp, seed)
            _run_job("MS -> MS_ref", mov, model_ms,
                     bands_dir / f"{mov.stem}_registered.tif",
                     f"pyramid={wu} pts={npts}", split_bands=True)

        # IR -> IR_ref -> MS_ref
        if u["IR"] is not None:
            mov_ir = u["IR"]
            if ref_ir_meta is not None:
                LOG.info("Scene %s: step 1, register IR -> IR_ref", key)
                model_ir, wu, npts = build_pyramid_model(
                    ref_ir, mov_ir, None, None, levels, args.warp)
                LOG.info("Scene %s: step 2, compose with the georeferenced "
                         "IR_ref -> MS_ref bridge", key)
                model = ComposedModel(outer=model_refIR2MS, inner=model_ir)
                info = f"pyramid=IR({wu})->IR_ref->MS_ref pts={npts}"
            else:
                import rasterio
                with rasterio.open(mov_ir) as ds:
                    ir_t = ds.transform
                model = GeoAlignModel(ref_ms_meta["transform"], ir_t)
                info = "warp=geoalign(IR->MS_ref direct)"
            _run_job("Composed IR: IR -> IR_ref -> MS_ref", mov_ir, model,
                     bands_dir / f"{mov_ir.stem}_registered.tif", info)

    elapsed = time.time() - t_start
    report.append("")
    report.append(f"Succeeded: {n_ok}/{n_jobs} | total: {_fmt_eta(elapsed)}")
    (outdir / "series_report.txt").write_text("\n".join(report), encoding="utf-8")

    LOG.info("=" * 72)
    LOG.info("DONE. Produced %d/%d output(s) in %s.", n_ok, n_jobs, _fmt_eta(elapsed))
    LOG.info("Report: %s", outdir / "series_report.txt")
    LOG.info("=" * 72)
    return 0 if n_ok == n_jobs else (0 if n_ok > 0 else 7)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        LOG.error("Interrupted"); sys.exit(130)
    except Exception:
        traceback.print_exc(); sys.exit(1)
