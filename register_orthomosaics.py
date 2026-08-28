#!/usr/bin/env python3
"""
register_orthomosaics.py
========================

Automatically register two georeferenced multispectral drone orthomosaics
(GeoTIFF). The preferred method is AROSICS. The fallback uses ORB and RANSAC
on a single-band working image, estimates a conservative translation or
Euclidean model, and reprojects each band with rasterio.

Usage:
    phenolapseUAV-register-pair --ref REFERENCE --mov MOVING --outdir OUTPUT
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

LOG = logging.getLogger("register")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    fmt = "[%(asctime)s] %(levelname)-7s %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S",
                        stream=sys.stdout, force=True)


# ---------------------------------------------------------------------------
# Report container
# ---------------------------------------------------------------------------
@dataclass
class Report:
    ref_path: str = ""
    mov_path: str = ""
    outdir: str = ""
    method_requested: str = ""
    method_used: str = ""
    band_choice: str = ""
    ref_meta: Dict[str, Any] = field(default_factory=dict)
    mov_meta: Dict[str, Any] = field(default_factory=dict)
    overlap: Dict[str, Any] = field(default_factory=dict)
    arosics_global: Dict[str, Any] = field(default_factory=dict)
    arosics_local: Dict[str, Any] = field(default_factory=dict)
    fallback: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""
    elapsed_s: float = 0.0

    def write(self, path: Path) -> None:
        lines: List[str] = []
        lines.append("=" * 78)
        lines.append("REGISTRATION REPORT")
        lines.append("=" * 78)
        lines.append(f"Started:  {self.started_at}")
        lines.append(f"Finished: {self.finished_at}")
        lines.append(f"Elapsed:  {self.elapsed_s:.2f} s")
        lines.append("")
        lines.append(f"Reference image : {self.ref_path}")
        lines.append(f"Moving image    : {self.mov_path}")
        lines.append(f"Output dir      : {self.outdir}")
        lines.append(f"Method requested: {self.method_requested}")
        lines.append(f"Method used     : {self.method_used}")
        lines.append(f"Band strategy   : {self.band_choice}")
        lines.append("")

        def block(title: str, d: Dict[str, Any]) -> None:
            lines.append("-" * 78)
            lines.append(title)
            lines.append("-" * 78)
            if not d:
                lines.append("(empty)")
            else:
                for k, v in d.items():
                    lines.append(f"  {k}: {v}")
            lines.append("")

        block("REFERENCE METADATA", self.ref_meta)
        block("MOVING METADATA", self.mov_meta)
        block("GEOGRAPHIC OVERLAP", self.overlap)
        block("AROSICS GLOBAL", self.arosics_global)
        block("AROSICS LOCAL", self.arosics_local)
        block("FALLBACK (ORB+RANSAC)", self.fallback)
        block("OUTPUT FILES", self.outputs)

        lines.append("-" * 78)
        lines.append("ERRORS / WARNINGS")
        lines.append("-" * 78)
        if not self.errors:
            lines.append("  (none)")
        else:
            for e in self.errors:
                lines.append(f"  - {e}")

        path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Raster inspection
# ---------------------------------------------------------------------------
def inspect_raster(path: Path) -> Dict[str, Any]:
    import rasterio
    with rasterio.open(path) as ds:
        b = ds.bounds
        t = ds.transform
        meta = {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size,
            "width": ds.width,
            "height": ds.height,
            "count": ds.count,
            "dtype": ds.dtypes[0],
            "crs": str(ds.crs),
            "transform": [t.a, t.b, t.c, t.d, t.e, t.f],
            "res_x": abs(t.a),
            "res_y": abs(t.e),
            "bounds": [b.left, b.bottom, b.right, b.top],
            "nodata": ds.nodata,
            "descriptions": list(ds.descriptions),
        }
    return meta


def log_meta(label: str, m: Dict[str, Any]) -> None:
    LOG.info("=== %s ===", label)
    for k in ("path", "exists", "size_bytes", "width", "height", "count",
             "dtype", "crs", "res_x", "res_y", "bounds", "nodata"):
        LOG.info("  %-13s : %s", k, m.get(k))
    LOG.debug("  transform     : %s", m.get("transform"))
    LOG.debug("  descriptions  : %s", m.get("descriptions"))


def compute_overlap(ref_meta: Dict[str, Any],
                    mov_meta: Dict[str, Any]) -> Dict[str, Any]:
    rb = ref_meta["bounds"]
    mb = mov_meta["bounds"]
    inter_left = max(rb[0], mb[0])
    inter_bot = max(rb[1], mb[1])
    inter_right = min(rb[2], mb[2])
    inter_top = min(rb[3], mb[3])
    has_overlap = inter_right > inter_left and inter_top > inter_bot
    if has_overlap:
        area_inter = (inter_right - inter_left) * (inter_top - inter_bot)
    else:
        area_inter = 0.0
    area_ref = (rb[2] - rb[0]) * (rb[3] - rb[1])
    area_mov = (mb[2] - mb[0]) * (mb[3] - mb[1])
    return {
        "ref_crs": ref_meta["crs"],
        "mov_crs": mov_meta["crs"],
        "same_crs": ref_meta["crs"] == mov_meta["crs"],
        "has_overlap": has_overlap,
        "intersection": [inter_left, inter_bot, inter_right, inter_top],
        "area_intersection": area_inter,
        "area_ref": area_ref,
        "area_mov": area_mov,
        "frac_of_ref": area_inter / area_ref if area_ref else 0.0,
        "frac_of_mov": area_inter / area_mov if area_mov else 0.0,
    }


# ---------------------------------------------------------------------------
# Mono-band working image
# ---------------------------------------------------------------------------
def build_workband(path: Path, band: Optional[int]) -> Tuple[np.ndarray, str]:
    """Return a float32 2D array used for matching, plus a description string."""
    import rasterio
    with rasterio.open(path) as ds:
        nodata = ds.nodata
        if band is not None:
            if band < 1 or band > ds.count:
                raise ValueError(
                    f"--band {band} out of range (1..{ds.count}) for {path}")
            arr = ds.read(band).astype(np.float32)
            mask = np.isfinite(arr)
            if nodata is not None:
                mask &= arr != nodata
            arr = _percentile_normalize(arr, mask)
            return arr, f"single band #{band}"
        # Robust mean across bands after percentile normalization
        bands = []
        for i in range(1, ds.count + 1):
            a = ds.read(i).astype(np.float32)
            m = np.isfinite(a)
            if nodata is not None:
                m &= a != nodata
            if m.sum() < 100:
                continue
            bands.append(_percentile_normalize(a, m))
        if not bands:
            raise RuntimeError(f"No usable bands in {path}")
        stack = np.stack(bands, axis=0)
        out = np.nanmean(stack, axis=0).astype(np.float32)
        return out, (f"robust mean of {len(bands)}/{ds.count} bands "
                     f"(percentile 2-98 normalized)")


def _percentile_normalize(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=np.float32)
    if mask.sum() < 10:
        return out
    valid = arr[mask]
    p2, p98 = np.percentile(valid, [2, 98])
    if p98 - p2 < 1e-9:
        return out
    norm = (arr - p2) / (p98 - p2)
    norm = np.clip(norm, 0.0, 1.0)
    out[mask] = norm[mask]
    return out


# ---------------------------------------------------------------------------
# Saturation / nodata mask (1 = bad, 0 = good), dilated
# ---------------------------------------------------------------------------
def saturation_mask(path: Path, dilate_px: int = 32,
                    include_zeros: bool = True) -> np.ndarray:
    """Build a per-pixel mask: 1 where any band == dtype max (saturation
    or implicit nodata), and optionally where any band == 0 (drone borders).
    Result is dilated by `dilate_px` pixels with a square kernel so that any
    matching window touching such a pixel is rejected.
    """
    import rasterio
    import cv2
    with rasterio.open(path) as ds:
        h, w = ds.height, ds.width
        dt = np.dtype(ds.dtypes[0])
        if not np.issubdtype(dt, np.integer):
            return np.zeros((h, w), dtype=np.uint8)
        dmax = np.iinfo(dt).max
        mask = np.zeros((h, w), dtype=bool)
        for i in range(1, ds.count + 1):
            a = ds.read(i)
            mask |= (a == dmax)
            if include_zeros:
                mask |= (a == 0)
    LOG.info("Saturation mask for %s: %d bad pixels (%.2f%%) before dilation",
             path.name, int(mask.sum()), 100.0 * mask.mean())
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    LOG.info("Saturation mask for %s: %d bad pixels (%.2f%%) after dilation %d",
             path.name, int(mask.sum()), 100.0 * mask.mean(), dilate_px)
    return mask.astype(np.uint8)


def write_mask_tif(src_path: Path, mask: np.ndarray, out_path: Path) -> None:
    """Write a single-band uint8 mask aligned with src_path's grid."""
    import rasterio
    with rasterio.open(src_path) as ds:
        profile = ds.profile.copy()
    profile.update(driver="GTiff", count=1, dtype="uint8", nodata=None,
                   compress="deflate", tiled=True,
                   blockxsize=512, blockysize=512)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)


# ---------------------------------------------------------------------------
# Reproject any raster onto the reference grid (same width, height, transform)
# ---------------------------------------------------------------------------
def reproject_to_ref_grid(src_path: Path, ref_path: Path, out_path: Path,
                          resampling: str = "cubic") -> None:
    import rasterio
    from rasterio.warp import reproject, Resampling
    rs_map = {"cubic": Resampling.cubic, "nearest": Resampling.nearest,
              "bilinear": Resampling.bilinear}
    rs = rs_map[resampling]
    with rasterio.open(ref_path) as ref:
        ref_w, ref_h = ref.width, ref.height
        ref_t, ref_c = ref.transform, ref.crs
    with rasterio.open(src_path) as src:
        profile = {
            "driver": "GTiff",
            "width": ref_w, "height": ref_h,
            "count": src.count,
            "dtype": src.dtypes[0],
            "crs": ref_c, "transform": ref_t,
            "tiled": True, "compress": "deflate", "predictor": 2,
            "blockxsize": 512, "blockysize": 512,
        }
        if src.nodata is not None:
            profile["nodata"] = src.nodata
        descs = src.descriptions
        with rasterio.open(out_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                arr = src.read(i)
                out = np.zeros((ref_h, ref_w), dtype=src.dtypes[0])
                reproject(
                    source=arr, destination=out,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=ref_t, dst_crs=ref_c,
                    resampling=rs,
                    src_nodata=src.nodata, dst_nodata=src.nodata,
                )
                dst.write(out, i)
                if descs[i - 1]:
                    dst.set_band_description(i, descs[i - 1])
    LOG.info("Reprojected %s -> %s on ref grid (%dx%d, %s)",
             src_path.name, out_path.name, ref_w, ref_h, resampling)


# ---------------------------------------------------------------------------
# Geoarray bool patch: AROSICS COREG_LOCAL builds bool GeoArrays internally,
# but GDAL doesn't accept bool. We transparently cast bool to uint8.
# ---------------------------------------------------------------------------
def patch_geoarray_bool() -> None:
    try:
        from geoarray.baseclasses import GeoArray
        if getattr(GeoArray, "_bool_patched", False):
            return
        getter = GeoArray.arr.fget
        orig_setter = GeoArray.arr.fset

        def new_setter(self, val):
            if isinstance(val, np.ndarray) and val.dtype == bool:
                val = val.astype(np.uint8)
            orig_setter(self, val)

        GeoArray.arr = property(getter, new_setter)
        GeoArray._bool_patched = True
        LOG.debug("GeoArray bool->uint8 setter patched.")
    except Exception as e:
        LOG.warning("Could not patch geoarray for bool dtype: %s", e)


# ---------------------------------------------------------------------------
# AROSICS pipeline
# ---------------------------------------------------------------------------
def run_arosics(ref: Path, mov: Path, outpath: Path, report: Report,
                debug: bool, band: Optional[int],
                ref_mask_path: Optional[Path],
                mov_mask_path: Optional[Path]) -> bool:
    """Run AROSICS global then local. Returns True on success."""
    try:
        from arosics import COREG, COREG_LOCAL
    except Exception as e:
        report.errors.append(f"AROSICS import failed: {e}")
        LOG.warning("AROSICS unavailable: %s", e)
        return False

    patch_geoarray_bool()

    band_match = band if band is not None else 1
    LOG.info("AROSICS will match on band %d (1-indexed)", band_match)
    if ref_mask_path:
        LOG.info("AROSICS ref bad-data mask: %s", ref_mask_path)
    if mov_mask_path:
        LOG.info("AROSICS mov bad-data mask: %s", mov_mask_path)

    LOG.info(">>> AROSICS global co-registration")
    try:
        cr = COREG(
            str(ref), str(mov),
            path_out=str(outpath),
            fmt_out="GTIFF",
            max_shift=200,
            ws=(512, 512),
            align_grids=True,
            match_gsd=False,
            r_b4match=band_match,
            s_b4match=band_match,
            mask_baddata_ref=str(ref_mask_path) if ref_mask_path else None,
            mask_baddata_tgt=str(mov_mask_path) if mov_mask_path else None,
            resamp_alg_deshift="cubic",
            resamp_alg_calc="cubic",
            q=not debug,
            v=debug,
            ignore_errors=True,
        )
        cr.calculate_spatial_shifts()
        report.arosics_global = {
            "success": bool(cr.success),
            "band_used": band_match,
            "window_size": (512, 512),
            "shift_px_x": getattr(cr, "x_shift_px", None),
            "shift_px_y": getattr(cr, "y_shift_px", None),
            "shift_map_x": getattr(cr, "x_shift_map", None),
            "shift_map_y": getattr(cr, "y_shift_map", None),
            "ssim_orig": getattr(cr, "ssim_orig", None),
            "ssim_deshifted": getattr(cr, "ssim_deshifted", None),
            "shift_reliability": getattr(cr, "shift_reliability", None),
        }
        for k, v in report.arosics_global.items():
            LOG.info("  %s = %s", k, v)
        if not cr.success:
            report.errors.append("AROSICS global: calculate_spatial_shifts "
                                 "did not succeed")
            return False
        cr.correct_shifts()
        # Tag the output with registration metadata
        try:
            import rasterio
            with rasterio.open(outpath, "r+") as ds:
                ds.update_tags(
                    REG_METHOD="arosics_global",
                    REG_BAND=str(band_match),
                    REG_SHIFT_PX_X=str(cr.x_shift_px),
                    REG_SHIFT_PX_Y=str(cr.y_shift_px),
                    REG_SHIFT_MAP_X=str(cr.x_shift_map),
                    REG_SHIFT_MAP_Y=str(cr.y_shift_map),
                    REG_SSIM_ORIG=str(cr.ssim_orig),
                    REG_SSIM_DESHIFTED=str(cr.ssim_deshifted),
                    REG_RELIABILITY=str(cr.shift_reliability),
                )
        except Exception as e:
            LOG.debug("Could not write registration tags: %s", e)
        LOG.info("AROSICS global: corrected file written to %s", outpath)
    except Exception as e:
        report.errors.append(f"AROSICS global failed: {e}")
        LOG.exception("AROSICS global failed")
        return False

    # Local refinement (best-effort; if it fails we keep global result)
    LOG.info(">>> AROSICS local refinement (best-effort)")
    try:
        local_out = outpath.with_name(outpath.stem + "_local.tif")
        crl = COREG_LOCAL(
            str(ref), str(outpath),
            path_out=str(local_out),
            fmt_out="GTIFF",
            grid_res=200,
            window_size=(128, 128),
            max_shift=50,
            r_b4match=band_match,
            s_b4match=band_match,
            mask_baddata_ref=str(ref_mask_path) if ref_mask_path else None,
            mask_baddata_tgt=str(mov_mask_path) if mov_mask_path else None,
            q=not debug,
            v=debug,
            ignore_errors=True,
            CPUs=None,
        )
        crl.correct_shifts()
        # Save tie-points geojson
        try:
            tp = crl.CoRegPoints_table.copy()
            # geopandas/fiona refuses bool columns -> cast to int
            for col in tp.columns:
                if tp[col].dtype == bool:
                    tp[col] = tp[col].astype("int64")
            geojson_path = Path(report.outdir) / "registration_vectors.geojson"
            tp.to_file(str(geojson_path), driver="GeoJSON")
            report.outputs["registration_vectors.geojson"] = str(geojson_path)
            n_total = len(tp)
            if "OUTLIER" in tp.columns:
                n_kept = int((tp["OUTLIER"] == 0).sum())
            else:
                n_kept = n_total
            report.arosics_local = {
                "success": True,
                "n_tiepoints_total": n_total,
                "n_tiepoints_kept": n_kept,
                "n_tiepoints_rejected": n_total - n_kept,
                "window_size": (128, 128),
                "grid_res_px": 200,
                "output": str(local_out),
            }
            for k, v in report.arosics_local.items():
                LOG.info("  local.%s = %s", k, v)
        except Exception as e:
            report.arosics_local = {"success": True,
                                    "tiepoints_export": f"failed: {e}"}
            LOG.warning("Tie-points export failed: %s", e)
        # Replace global output by local one
        if local_out.exists():
            try:
                outpath.unlink()
            except FileNotFoundError:
                pass
            local_out.replace(outpath)
            LOG.info("AROSICS local: refined file -> %s", outpath)
        return True
    except Exception as e:
        report.arosics_local = {"success": False, "error": str(e)}
        report.errors.append(f"AROSICS local failed (kept global result): {e}")
        LOG.warning("AROSICS local failed (kept global): %s", e)
        return True  # global already produced a valid output


# ---------------------------------------------------------------------------
# Fallback: ORB + RANSAC on the working band
# ---------------------------------------------------------------------------
def run_fallback(ref: Path, mov: Path, outpath: Path, report: Report,
                 band: Optional[int],
                 ref_bad_mask: Optional[np.ndarray] = None,
                 mov_bad_mask: Optional[np.ndarray] = None) -> bool:
    import cv2
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import Affine

    LOG.info(">>> Fallback ORB + RANSAC")
    ref_band, ref_desc = build_workband(ref, band)
    mov_band, mov_desc = build_workband(mov, band)
    LOG.info("Ref work band: %s | shape=%s", ref_desc, ref_band.shape)
    LOG.info("Mov work band: %s | shape=%s", mov_desc, mov_band.shape)
    # Apply bad-data masks
    if ref_bad_mask is not None:
        ref_band = ref_band.copy()
        ref_band[ref_bad_mask.astype(bool)] = np.nan
        LOG.info("Ref band masked: %.2f%% bad pixels excluded",
                 100.0 * ref_bad_mask.mean())
    if mov_bad_mask is not None:
        mov_band = mov_band.copy()
        mov_band[mov_bad_mask.astype(bool)] = np.nan
        LOG.info("Mov band masked: %.2f%% bad pixels excluded",
                 100.0 * mov_bad_mask.mean())

    # to uint8 for ORB
    def to_u8(a: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(a, nan=0.0)
        x = np.clip(x, 0.0, 1.0)
        return (x * 255.0).astype(np.uint8)

    r8 = to_u8(ref_band)
    m8 = to_u8(mov_band)

    # Downsample if very large to keep ORB fast
    max_dim = 4096
    scale = 1.0
    if max(r8.shape) > max_dim or max(m8.shape) > max_dim:
        scale = max_dim / float(max(max(r8.shape), max(m8.shape)))
        new_r = (int(r8.shape[1] * scale), int(r8.shape[0] * scale))
        new_m = (int(m8.shape[1] * scale), int(m8.shape[0] * scale))
        LOG.info("Downsampling for ORB: scale=%.4f", scale)
        r8s = cv2.resize(r8, new_r, interpolation=cv2.INTER_AREA)
        m8s = cv2.resize(m8, new_m, interpolation=cv2.INTER_AREA)
    else:
        r8s, m8s = r8, m8

    orb = cv2.ORB_create(nfeatures=8000, fastThreshold=10)
    kp_r, des_r = orb.detectAndCompute(r8s, None)
    kp_m, des_m = orb.detectAndCompute(m8s, None)
    LOG.info("ORB keypoints: ref=%d  mov=%d", len(kp_r), len(kp_m))
    if des_r is None or des_m is None or len(kp_r) < 30 or len(kp_m) < 30:
        report.errors.append("Fallback: not enough ORB keypoints")
        return False

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des_m, des_r, k=2)
    good = [m for m, n in knn if m.distance < 0.75 * n.distance]
    LOG.info("ORB matches: total=%d  good(Lowe)=%d", len(knn), len(good))
    if len(good) < 20:
        report.errors.append(f"Fallback: only {len(good)} good matches")
        return False

    pts_mov = np.float32([kp_m[g.queryIdx].pt for g in good])
    pts_ref = np.float32([kp_r[g.trainIdx].pt for g in good])

    # Conservative model: estimate Euclidean (rotation + translation) with RANSAC.
    # In downsampled pixel space (mov->ref).
    M, inliers = cv2.estimateAffinePartial2D(
        pts_mov, pts_ref,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=5000,
        confidence=0.999,
        refineIters=20,
    )
    if M is None:
        report.errors.append("Fallback: estimateAffinePartial2D returned None")
        return False
    n_in = int(inliers.sum())
    LOG.info("RANSAC inliers: %d / %d", n_in, len(good))
    if n_in < 15:
        report.errors.append(f"Fallback: only {n_in} RANSAC inliers")
        return False

    # Re-scale M from downsampled pixel space to full-resolution mov pixel space.
    if scale != 1.0:
        S = np.array([[1.0 / scale, 0, 0], [0, 1.0 / scale, 0]])
        Sinv = np.array([[scale, 0, 0], [0, scale, 0]])
        # M operates as: p_ref_small = M * [p_mov_small; 1]
        # p_mov_full = p_mov_small / scale ; p_ref_full = p_ref_small / scale
        # => M_full = (1/scale) * M * diag(scale, scale, 1) (translation also /scale)
        a, b, tx = M[0]
        c, d, ty = M[1]
        M_full = np.array([
            [a, b, tx / scale],
            [c, d, ty / scale],
        ])
    else:
        M_full = M.copy()

    # Decompose
    a, b, tx = M_full[0]
    c, d, ty = M_full[1]
    rot_deg = float(np.degrees(np.arctan2(c, a)))
    sx = float(np.hypot(a, c))
    LOG.info("Fallback model: rot=%.4f deg, scale=%.6f, tx=%.3f px, ty=%.3f px",
             rot_deg, sx, tx, ty)

    # If rotation is tiny, reduce to pure translation (most conservative).
    pure_translation = abs(rot_deg) < 0.05 and abs(sx - 1.0) < 1e-3
    if pure_translation:
        LOG.info("Rotation/scale negligible -> using pure translation")

    # Apply via rasterio: build a new transform for the moving image.
    # In mov pixel coords, the corrective map is: p_ref = M * p_mov
    # We want the registered raster aligned to ref grid spatially. Approach:
    # adjust mov's affine so that pixel (col,row) in mov maps to the same
    # geographic location it *should* map to in ref. With pure translation we
    # simply add (tx, ty) in pixels via mov's transform.
    with rasterio.open(mov) as ds_m:
        mov_transform = ds_m.transform
        mov_crs = ds_m.crs
        mov_count = ds_m.count
        mov_dtype = ds_m.dtypes[0]
        mov_nodata = ds_m.nodata
        mov_height = ds_m.height
        mov_width = ds_m.width
        mov_descriptions = ds_m.descriptions
        mov_data = ds_m.read()

    with rasterio.open(ref) as ds_r:
        ref_transform = ds_r.transform
        ref_crs = ds_r.crs

    # Build correction in geographic coords.
    # Mov pixel (col,row) -> mov geo via mov_transform.
    # Corrected mov pixel position = M_full * (col,row).
    # Corrected geo = mov_transform * corrected_pixel.
    # We can express this as a new affine: T_new = mov_transform * M_full_homog.
    A_M = np.array([[M_full[0, 0], M_full[0, 1], M_full[0, 2]],
                    [M_full[1, 0], M_full[1, 1], M_full[1, 2]],
                    [0.0, 0.0, 1.0]])
    A_T = np.array([[mov_transform.a, mov_transform.b, mov_transform.c],
                    [mov_transform.d, mov_transform.e, mov_transform.f],
                    [0.0, 0.0, 1.0]])
    A_new = A_T @ A_M
    new_transform = Affine(A_new[0, 0], A_new[0, 1], A_new[0, 2],
                           A_new[1, 0], A_new[1, 1], A_new[1, 2])
    LOG.info("Original mov transform: %s", mov_transform)
    LOG.info("New mov transform     : %s", new_transform)

    # Compute pixel shift in mov pixel units (just for the report)
    shift_px = float(np.hypot(tx, ty))
    # Approximate map shift via mov resolution
    res = abs(mov_transform.a)
    shift_map = shift_px * res

    profile = {
        "driver": "GTiff",
        "height": mov_height,
        "width": mov_width,
        "count": mov_count,
        "dtype": mov_dtype,
        "crs": mov_crs,
        "transform": new_transform,
        "tiled": True,
        "compress": "deflate",
        "predictor": 2,
        "blockxsize": 512,
        "blockysize": 512,
    }
    if mov_nodata is not None:
        profile["nodata"] = mov_nodata

    with rasterio.open(outpath, "w", **profile) as dst:
        dst.write(mov_data)
        for i, desc in enumerate(mov_descriptions, start=1):
            if desc:
                dst.set_band_description(i, desc)
        dst.update_tags(
            REG_METHOD="fallback_orb_ransac",
            REG_BAND=str(band if band is not None else 0),
            REG_MODEL="translation" if pure_translation else "euclidean",
            REG_SHIFT_PX_X=str(float(tx)),
            REG_SHIFT_PX_Y=str(float(ty)),
            REG_ROTATION_DEG=str(rot_deg),
            REG_SCALE=str(sx),
            REG_N_INLIERS=str(n_in),
            REG_N_MATCHES=str(len(good)),
        )

    report.fallback = {
        "n_keypoints_ref": len(kp_r),
        "n_keypoints_mov": len(kp_m),
        "n_matches_total": len(knn),
        "n_matches_good_lowe": len(good),
        "n_ransac_inliers": n_in,
        "n_ransac_rejected": len(good) - n_in,
        "downsample_scale_for_orb": scale,
        "model": "translation_only" if pure_translation else "euclidean",
        "rotation_deg": rot_deg,
        "scale_factor": sx,
        "shift_px_x": float(tx),
        "shift_px_y": float(ty),
        "shift_px_norm": shift_px,
        "shift_map_norm": shift_map,
        "old_transform": list(mov_transform)[:6],
        "new_transform": list(new_transform)[:6],
    }
    for k, v in report.fallback.items():
        LOG.info("  %s = %s", k, v)
    return True


# ---------------------------------------------------------------------------
# Quicklooks & differences
# ---------------------------------------------------------------------------
def quicklook(path: Path, png_out: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arr, _ = build_workband(path, band=None)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(png_out, dpi=120)
    plt.close(fig)


def difference_map_same_grid(a_path: Path, b_path: Path,
                             png_out: Path, title: str) -> None:
    """Inputs MUST share the same grid (width/height/transform).
    Compute mono-band of each (robust mean of valid bands) and save abs-diff PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a, _ = build_workband(a_path, band=None)
    b, _ = build_workband(b_path, band=None)
    if a.shape != b.shape:
        raise ValueError(f"Grid mismatch: {a.shape} vs {b.shape}")
    diff = np.abs(a - b)
    fig, ax = plt.subplots(figsize=(7, 7))
    valid = np.isfinite(diff)
    p99 = np.nanpercentile(diff[valid], 99) if valid.any() else 0.5
    im = ax.imshow(diff, cmap="magma", vmin=0, vmax=max(p99, 1e-6))
    n_valid = int(valid.sum())
    if n_valid:
        ax.set_title(f"{title}\nmean={np.nanmean(diff):.4f} "
                     f"median={np.nanmedian(diff):.4f} p99={p99:.4f}")
    else:
        ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(png_out, dpi=120)
    plt.close(fig)


def difference_map(ref: Path, other: Path, png_out: Path, title: str) -> None:
    """Resample 'other' onto ref grid (mono band) and save abs-diff PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.warp import reproject, Resampling

    ref_arr, _ = build_workband(ref, band=None)
    with rasterio.open(ref) as r, rasterio.open(other) as o:
        dst = np.full((r.height, r.width), np.nan, dtype=np.float32)
        # Build mono band of 'other' first
        other_arr, _ = build_workband(other, band=None)
        reproject(
            source=other_arr, destination=dst,
            src_transform=o.transform, src_crs=o.crs,
            dst_transform=r.transform, dst_crs=r.crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan, dst_nodata=np.nan,
        )
    diff = np.abs(ref_arr - dst)
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(diff, cmap="magma", vmin=0, vmax=0.5)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(png_out, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", required=True, help="Reference GeoTIFF")
    p.add_argument("--mov", required=True, help="Moving GeoTIFF")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--band", type=int, default=None,
                   help="1-indexed band for matching (default: AROSICS uses "
                        "band 1; fallback uses robust mean)")
    p.add_argument("--method", choices=["arosics", "fallback", "auto"],
                   default="auto")
    p.add_argument("--mask-dilate", type=int, default=64,
                   help="Pixels to dilate the saturation/border mask (def: 64)")
    p.add_argument("--no-mask", action="store_true",
                   help="Disable saturation/border masking")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.debug)

    ref = Path(args.ref)
    mov = Path(args.mov)
    outdir = Path(args.outdir)

    report = Report(
        ref_path=str(ref),
        mov_path=str(mov),
        outdir=str(outdir),
        method_requested=args.method,
        band_choice=(f"single band #{args.band}" if args.band
                     else "robust mean of valid bands (percentile 2-98)"),
        started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    t0 = time.time()

    LOG.info("=" * 70)
    LOG.info("phenolapseUAV - drone orthomosaic co-registration")
    LOG.info("=" * 70)
    LOG.info("ref     = %s (exists=%s)", ref, ref.exists())
    LOG.info("mov     = %s (exists=%s)", mov, mov.exists())
    LOG.info("outdir  = %s", outdir)
    LOG.info("method  = %s", args.method)
    LOG.info("band    = %s", args.band)
    LOG.info("debug   = %s", args.debug)
    LOG.info("overwrite = %s", args.overwrite)

    if not ref.exists():
        LOG.error("Reference not found: %s", ref); return 2
    if not mov.exists():
        LOG.error("Moving image not found: %s", mov); return 2
    outdir.mkdir(parents=True, exist_ok=True)

    # Define output paths
    out_tif = outdir / f"{mov.stem}_registered.tif"
    out_report = outdir / "registration_report.txt"
    out_ql_before = outdir / "quicklook_before.png"
    out_ql_after = outdir / "quicklook_after.png"
    out_diff_before = outdir / "difference_before.png"
    out_diff_after = outdir / "difference_after.png"
    out_mov_on_ref_before = outdir / "mov_on_refgrid_before.tif"
    out_mov_on_ref_after = outdir / "mov_on_refgrid_after.tif"
    out_ref_mask = outdir / "_ref_badmask.tif"
    out_mov_mask = outdir / "_mov_badmask.tif"

    # Overwrite checks
    for p in [out_tif, out_report, out_ql_before, out_ql_after,
              out_diff_before, out_diff_after,
              out_mov_on_ref_before, out_mov_on_ref_after]:
        if p.exists() and not args.overwrite:
            LOG.error("Output already exists (use --overwrite): %s", p)
            return 3

    # Inspect rasters
    try:
        report.ref_meta = inspect_raster(ref)
        report.mov_meta = inspect_raster(mov)
    except Exception as e:
        LOG.exception("Raster inspection failed")
        report.errors.append(f"inspect_raster failed: {e}")
        report.finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
        report.elapsed_s = time.time() - t0
        report.write(out_report)
        return 4

    log_meta("REFERENCE", report.ref_meta)
    log_meta("MOVING", report.mov_meta)

    report.overlap = compute_overlap(report.ref_meta, report.mov_meta)
    LOG.info("=== OVERLAP ===")
    for k, v in report.overlap.items():
        LOG.info("  %-22s : %s", k, v)

    if not report.overlap["same_crs"]:
        LOG.error("CRS mismatch: ref=%s mov=%s",
                  report.overlap["ref_crs"], report.overlap["mov_crs"])
        report.errors.append("CRS mismatch between ref and mov")
        report.finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
        report.elapsed_s = time.time() - t0
        report.write(out_report)
        return 5
    if not report.overlap["has_overlap"]:
        LOG.error("No geographic overlap between ref and mov.")
        report.errors.append("No geographic overlap")
        report.finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
        report.elapsed_s = time.time() - t0
        report.write(out_report)
        return 6
    if report.overlap["frac_of_ref"] < 0.05 and report.overlap["frac_of_mov"] < 0.05:
        LOG.warning("Very small overlap (<5%%). Registration may be unreliable.")

    # Quicklooks before
    LOG.info(">>> Quicklook BEFORE")
    try:
        quicklook(mov, out_ql_before, "mov BEFORE registration (mono-band)")
        report.outputs["quicklook_before.png"] = str(out_ql_before)
    except Exception as e:
        LOG.warning("quicklook_before failed: %s", e)
        report.errors.append(f"quicklook_before: {e}")

    # Reproject the *original* moving image onto the reference grid (BEFORE).
    LOG.info(">>> Reproject mov BEFORE on ref grid -> %s", out_mov_on_ref_before)
    try:
        reproject_to_ref_grid(mov, ref, out_mov_on_ref_before,
                              resampling="cubic")
        report.outputs["mov_on_refgrid_before.tif"] = str(out_mov_on_ref_before)
    except Exception as e:
        LOG.warning("reproject BEFORE failed: %s", e)
        report.errors.append(f"reproject_before: {e}")

    try:
        difference_map_same_grid(
            ref, out_mov_on_ref_before, out_diff_before,
            "|ref - mov| BEFORE (on ref grid)")
        report.outputs["difference_before.png"] = str(out_diff_before)
    except Exception as e:
        LOG.warning("difference_before failed: %s", e)
        report.errors.append(f"difference_before: {e}")

    # Build saturation/border masks
    ref_mask_arr = mov_mask_arr = None
    ref_mask_path_used = mov_mask_path_used = None
    if not args.no_mask:
        LOG.info(">>> Building saturation/border masks (dilate=%d px)",
                 args.mask_dilate)
        try:
            ref_mask_arr = saturation_mask(ref, dilate_px=args.mask_dilate)
            mov_mask_arr = saturation_mask(mov, dilate_px=args.mask_dilate)
            write_mask_tif(ref, ref_mask_arr, out_ref_mask)
            write_mask_tif(mov, mov_mask_arr, out_mov_mask)
            ref_mask_path_used = out_ref_mask
            mov_mask_path_used = out_mov_mask
            report.outputs["_ref_badmask.tif"] = str(out_ref_mask)
            report.outputs["_mov_badmask.tif"] = str(out_mov_mask)
        except Exception as e:
            LOG.warning("Could not build masks: %s (continuing without)", e)
            report.errors.append(f"mask build failed: {e}")

    # Run method
    success = False
    used = ""
    if args.method in ("arosics", "auto"):
        ok = run_arosics(ref, mov, out_tif, report, args.debug,
                         args.band, ref_mask_path_used, mov_mask_path_used)
        if ok:
            success = True; used = "arosics"
        elif args.method == "arosics":
            LOG.error("AROSICS requested but failed.")
        else:
            LOG.warning("AROSICS failed, falling back.")

    if not success and args.method in ("fallback", "auto"):
        try:
            ok = run_fallback(ref, mov, out_tif, report, args.band,
                              ref_mask_arr, mov_mask_arr)
            if ok:
                success = True; used = "fallback"
        except Exception as e:
            LOG.exception("Fallback crashed")
            report.errors.append(f"fallback crashed: {e}")

    report.method_used = used or "none"

    if success:
        report.outputs[out_tif.name] = str(out_tif)
        # Reproject registered onto ref grid (AFTER)
        LOG.info(">>> Reproject mov AFTER on ref grid -> %s",
                 out_mov_on_ref_after)
        try:
            reproject_to_ref_grid(out_tif, ref, out_mov_on_ref_after,
                                  resampling="cubic")
            report.outputs["mov_on_refgrid_after.tif"] = str(out_mov_on_ref_after)
        except Exception as e:
            LOG.warning("reproject AFTER failed: %s", e)
            report.errors.append(f"reproject_after: {e}")
        # Quicklook & diff after (using on-ref-grid file for diff)
        LOG.info(">>> Quicklook AFTER")
        try:
            quicklook(out_tif, out_ql_after,
                      f"mov AFTER registration ({used})")
            report.outputs["quicklook_after.png"] = str(out_ql_after)
        except Exception as e:
            LOG.warning("quicklook_after failed: %s", e)
            report.errors.append(f"quicklook_after: {e}")
        try:
            difference_map_same_grid(
                ref, out_mov_on_ref_after, out_diff_after,
                f"|ref - mov| AFTER ({used}) on ref grid")
            report.outputs["difference_after.png"] = str(out_diff_after)
        except Exception as e:
            LOG.warning("difference_after failed: %s", e)
            report.errors.append(f"difference_after: {e}")

    report.finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
    report.elapsed_s = time.time() - t0
    report.outputs["registration_report.txt"] = str(out_report)
    report.write(out_report)

    LOG.info("=" * 70)
    LOG.info("DONE. method=%s success=%s elapsed=%.2fs",
             report.method_used, success, report.elapsed_s)
    LOG.info("Outputs:")
    for k, v in report.outputs.items():
        LOG.info("  - %s : %s", k, v)
    LOG.info("Report: %s", out_report)
    LOG.info("=" * 70)

    return 0 if success else 7


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        LOG.error("Interrupted")
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
