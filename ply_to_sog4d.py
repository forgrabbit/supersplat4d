#!/usr/bin/env python3
"""
Convert dynamic gaussian splatting PLY to compressed .sog4d format.

This script reads a dynamic gaussian PLY file and produces a single .sog4d file
containing all compressed data (WebP images) and metadata.

Dependencies:
    pip install numpy plyfile pillow scikit-learn

Optional (for segment computation):
    pip install torch
"""

import argparse
import io
import json
import os
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from plyfile import PlyData
from sklearn.cluster import KMeans

try:
    import torch
except ImportError:
    torch = None


# =============================================================================
# Constants
# =============================================================================



# =============================================================================
# cfg_args parsing
# =============================================================================

def parse_cfg_args_text(text: str) -> Dict[str, float]:
    """Parse strings like: Namespace(duration=2.0, start=2.0, fps=30.0, sh_degree=0)"""
    text = text.strip()
    if not text.startswith('Namespace(') or not text.endswith(')'):
        raise ValueError(f"Unrecognized cfg_args format: {text[:200]}")

    content = text[len('Namespace('):-1]

    out: Dict[str, float] = {}
    parts = content.split(',')
    for part in parts:
        part = part.strip()
        if '=' not in part:
            continue

        key, value = part.split('=', 1)
        key = key.strip()
        value = value.strip()

        try:
            if '.' in value or 'e' in value.lower():
                out[key] = float(value)
            else:
                out[key] = int(value)
        except ValueError:
            print(f"Warning: Could not parse value for key '{key}': {value}")

    required_keys = ['start', 'duration', 'fps']
    if not all(k in out for k in required_keys):
        missing = [k for k in required_keys if k not in out]
        raise ValueError(f"Could not parse required keys {missing} from cfg_args. Found: {list(out.keys())}")

    return out


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class SplatData:
    num: int
    sh_degree: int
    fields: Dict[str, np.ndarray]


# =============================================================================
# PLY loading
# =============================================================================

def load_ply_dynamic(ply_path: str, sh_degree: int, max_splats: Optional[int] = None, seed: int = 0) -> SplatData:
    """Load dynamic gaussian PLY file."""
    print(f"Loading PLY: {ply_path}")
    ply = PlyData.read(ply_path)
    v = ply.elements[0]
    names = v.data.dtype.names

    # Check required fields
    base_req = [
        'x', 'y', 'z',
        'scale_0', 'scale_1', 'scale_2',
        'rot_0', 'rot_1', 'rot_2', 'rot_3',
        'f_dc_0', 'f_dc_1', 'f_dc_2',
        'opacity',
        'trbf_center', 'trbf_scale'
    ]

    has_motion = all(n in names for n in ['motion_0', 'motion_1', 'motion_2'])
    if not has_motion:
        raise ValueError('PLY missing motion_0, motion_1, motion_2')

    missing = [n for n in base_req if n not in names]
    if missing:
        raise ValueError(f"PLY missing properties: {missing}")

    num_full = v.count

    def f32(name: str) -> np.ndarray:
        return np.asarray(v.data[name], dtype=np.float32)

    fields: Dict[str, np.ndarray] = {}

    # Position
    fields['x'] = f32('x')
    fields['y'] = f32('y')
    fields['z'] = f32('z')

    # Scale (keep as log scale for encoding)
    fields['scale_0'] = f32('scale_0')
    fields['scale_1'] = f32('scale_1')
    fields['scale_2'] = f32('scale_2')

    # Rotation
    fields['rot_0'] = f32('rot_0')
    fields['rot_1'] = f32('rot_1')
    fields['rot_2'] = f32('rot_2')
    fields['rot_3'] = f32('rot_3')

    # Color (SH DC)
    fields['f_dc_0'] = f32('f_dc_0')
    fields['f_dc_1'] = f32('f_dc_1')
    fields['f_dc_2'] = f32('f_dc_2')

    # Opacity (logit)
    fields['opacity'] = f32('opacity')

    # TRBF parameters
    fields['trbf_center'] = f32('trbf_center')
    # Apply exp to trbf_scale (stored as log in PLY)
    fields['trbf_scale'] = np.exp(f32('trbf_scale'))

    # Motion
    fields['motion_0'] = f32('motion_0')
    fields['motion_1'] = f32('motion_1')
    fields['motion_2'] = f32('motion_2')

    # SH rest coefficients (if any)
    if sh_degree > 0:
        sh_coeffs = (sh_degree + 1) ** 2
        rest = sh_coeffs - 1
        rest_names = [f'f_rest_{i}' for i in range(rest * 3)]
        if all(n in names for n in rest_names):
            rest_data = np.stack([f32(n) for n in rest_names], axis=1)
            fields['sh_rest'] = rest_data.reshape(num_full, rest * 3)
        else:
            print(f"Warning: SH degree {sh_degree} requested but f_rest_* not found, using zeros")
            fields['sh_rest'] = np.zeros((num_full, rest * 3), dtype=np.float32)

    # Subsample if requested
    if max_splats is not None and 0 < max_splats < num_full:
        print(f"Subsampling {max_splats} splats from {num_full}")
        rng = np.random.default_rng(seed)
        idx = rng.choice(num_full, size=max_splats, replace=False)
        idx.sort()
        fields = {k: v[idx] for k, v in fields.items()}
        num = max_splats
    else:
        num = num_full

    print(f"Loaded {num} splats")
    return SplatData(num=num, sh_degree=sh_degree, fields=fields)


# =============================================================================
# Filter low opacity splats
# =============================================================================

def filter_low_opacity_splats(data: SplatData, cfg: Dict[str, float], opacity_threshold: float) -> SplatData:
    """
    Remove splats that have max dynamic opacity < opacity_threshold across all frames.
    
    Dynamic opacity = sigmoid(opacity_logit) * exp(-((t - trbf_center) / trbf_scale)^2)
    
    Uses PyTorch for vectorized computation.
    """
    if torch is None:
        print("\nWARNING: torch not found. Skipping low opacity filtering.")
        print("Install torch for this feature: pip install torch")
        return data
    
    print(f"\nFiltering splats with max opacity < {opacity_threshold}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    
    # Get parameters
    opacity_logit = torch.from_numpy(data.fields['opacity']).to(device).squeeze()
    tc = torch.from_numpy(data.fields['trbf_center']).to(device).squeeze()
    ts = torch.from_numpy(data.fields['trbf_scale']).to(device).squeeze()
    
    start = float(cfg.get('start', 0.0))
    duration = float(cfg.get('duration', 0.0))
    fps = float(cfg.get('fps', 30.0))
    
    # Sample all time points
    num_frames = int(duration * fps)
    sample_times = torch.linspace(start, start + duration, num_frames, device=device)
    
    print(f"  Sampling {num_frames} frames...")
    
    # Compute max opacity for each splat across all frames
    # Shape: [num_frames, num_splats]
    dt = sample_times.view(-1, 1) - tc.view(1, -1)  # [F, N]
    dt_scaled = dt / torch.clamp(ts.view(1, -1), min=1e-6)  # [F, N]
    trbf_gauss = torch.exp(-dt_scaled * dt_scaled)  # [F, N]
    
    base_opacity = torch.sigmoid(opacity_logit).view(1, -1)  # [1, N]
    dyn_opacity = base_opacity * trbf_gauss  # [F, N]
    
    # Get max opacity for each splat
    max_opacity, _ = torch.max(dyn_opacity, dim=0)  # [N]
    
    # Find splats to keep
    keep_mask = max_opacity >= opacity_threshold
    keep_indices = torch.where(keep_mask)[0].cpu().numpy()
    
    num_removed = data.num - len(keep_indices)
    print(f"  Keeping {len(keep_indices)} splats, removing {num_removed} ({100*num_removed/data.num:.1f}%)")
    
    if num_removed == 0:
        return data
    
    # Filter all fields
    filtered_fields = {k: v[keep_indices] for k, v in data.fields.items()}
    
    return SplatData(
        num=len(keep_indices),
        sh_degree=data.sh_degree,
        fields=filtered_fields
    )


# =============================================================================
# Morton order sorting (for better compression)
# =============================================================================

def morton_encode_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Encode 3D coordinates to Morton code (Z-order curve)."""

    def spread_bits(v: np.ndarray) -> np.ndarray:
        """Spread bits for 21-bit input to 63-bit output."""
        v = v.astype(np.uint64)
        v = (v | (v << 32)) & 0x1f00000000ffff
        v = (v | (v << 16)) & 0x1f0000ff0000ff
        v = (v | (v << 8)) & 0x100f00f00f00f00f
        v = (v | (v << 4)) & 0x10c30c30c30c30c3
        v = (v | (v << 2)) & 0x1249249249249249
        return v

    return spread_bits(x) | (spread_bits(y) << 1) | (spread_bits(z) << 2)


def sort_morton_order(data: SplatData) -> np.ndarray:
    """Sort splats by Morton order and return indices."""
    x = data.fields['x']
    y = data.fields['y']
    z = data.fields['z']

    # Normalize to [0, 2^21-1] range
    xyz = np.stack([x, y, z], axis=1)
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    scale = maxs - mins
    scale[scale == 0] = 1.0

    norm = (xyz - mins) / scale
    quantized = (norm * ((1 << 21) - 1)).astype(np.uint32)

    # Compute Morton codes
    morton = morton_encode_3d(quantized[:, 0], quantized[:, 1], quantized[:, 2])

    # Sort by Morton code
    indices = np.argsort(morton).astype(np.uint32)
    return indices


# =============================================================================
# Compression utilities
# =============================================================================

def log_transform(value: np.ndarray) -> np.ndarray:
    """Apply log transform: sign(x) * ln(|x| + 1)"""
    return np.sign(value) * np.log(np.abs(value) + 1)


def quantize_16bit(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Quantize float values to 16-bit integers."""
    scale = vmax - vmin
    if scale == 0:
        scale = 1.0
    normalized = (values - vmin) / scale
    return (np.clip(normalized, 0, 1) * 65535).astype(np.uint16)


def encode_webp_lossless(rgba: np.ndarray, width: int, height: int) -> bytes:
    """Encode RGBA data to lossless WebP."""
    img = Image.frombytes('RGBA', (width, height), rgba.tobytes())
    buf = io.BytesIO()
    img.save(buf, format='WEBP', lossless=True)
    return buf.getvalue()


def compute_texture_size(num_splats: int) -> Tuple[int, int]:
    """Compute texture dimensions for given number of splats."""
    width = ((int(np.ceil(np.sqrt(num_splats))) + 3) // 4) * 4
    height = ((num_splats + width - 1) // width + 3) // 4 * 4
    return width, height


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


# =============================================================================
# Encoding functions for each attribute
# =============================================================================

def encode_means(data: SplatData, indices: np.ndarray, width: int, height: int) -> Tuple[bytes, bytes, Dict]:
    """Encode positions (x, y, z) using 16-bit quantization with log transform."""
    print("  Encoding means (positions)...")

    x = data.fields['x'][indices]
    y = data.fields['y'][indices]
    z = data.fields['z'][indices]

    # Apply log transform
    x_log = log_transform(x)
    y_log = log_transform(y)
    z_log = log_transform(z)

    # Compute min/max for each axis
    mins = [float(x_log.min()), float(y_log.min()), float(z_log.min())]
    maxs = [float(x_log.max()), float(y_log.max()), float(z_log.max())]

    # Quantize to 16-bit
    x_q = quantize_16bit(x_log, mins[0], maxs[0])
    y_q = quantize_16bit(y_log, mins[1], maxs[1])
    z_q = quantize_16bit(z_log, mins[2], maxs[2])

    # Pack into two RGBA textures (low and high bytes)
    tex_size = width * height
    means_l = np.zeros((tex_size, 4), dtype=np.uint8)
    means_u = np.zeros((tex_size, 4), dtype=np.uint8)

    n = len(indices)
    means_l[:n, 0] = x_q & 0xFF
    means_l[:n, 1] = y_q & 0xFF
    means_l[:n, 2] = z_q & 0xFF
    means_l[:n, 3] = 255

    means_u[:n, 0] = (x_q >> 8) & 0xFF
    means_u[:n, 1] = (y_q >> 8) & 0xFF
    means_u[:n, 2] = (z_q >> 8) & 0xFF
    means_u[:n, 3] = 255

    webp_l = encode_webp_lossless(means_l.flatten(), width, height)
    webp_u = encode_webp_lossless(means_u.flatten(), width, height)

    meta = {
        'mins': mins,
        'maxs': maxs,
        'files': ['means_l.webp', 'means_u.webp']
    }

    return webp_l, webp_u, meta


def encode_quats(data: SplatData, indices: np.ndarray, width: int, height: int) -> Tuple[bytes, Dict]:
    """Encode quaternions using smallest-component compression."""
    print("  Encoding quaternions...")

    r0 = data.fields['rot_0'][indices]
    r1 = data.fields['rot_1'][indices]
    r2 = data.fields['rot_2'][indices]
    r3 = data.fields['rot_3'][indices]

    n = len(indices)
    tex_size = width * height
    quats = np.zeros((tex_size, 4), dtype=np.uint8)

    sqrt2 = np.sqrt(2.0)

    for i in range(n):
        q = np.array([r0[i], r1[i], r2[i], r3[i]], dtype=np.float32)

        # Normalize
        length = np.sqrt(np.sum(q * q))
        if length > 0:
            q /= length

        # Find max component
        max_comp = np.argmax(np.abs(q))

        # Make max component positive
        if q[max_comp] < 0:
            q = -q

        # Scale by sqrt(2) to fit in [-1, 1]
        q *= sqrt2

        # Get indices of non-max components
        idx_map = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
        idx = idx_map[max_comp]

        # Quantize to 8-bit
        quats[i, 0] = int(np.clip((q[idx[0]] * 0.5 + 0.5) * 255, 0, 255))
        quats[i, 1] = int(np.clip((q[idx[1]] * 0.5 + 0.5) * 255, 0, 255))
        quats[i, 2] = int(np.clip((q[idx[2]] * 0.5 + 0.5) * 255, 0, 255))
        quats[i, 3] = 252 + max_comp  # Tag: 252-255 indicates which component is max

    webp = encode_webp_lossless(quats.flatten(), width, height)

    meta = {
        'files': ['quats.webp']
    }

    return webp, meta


def encode_scales(data: SplatData, indices: np.ndarray, width: int, height: int,
                  n_clusters: int = 256) -> Tuple[bytes, Dict]:
    """Encode scales using k-means clustering."""
    print("  Encoding scales (k-means clustering)...")

    s0 = data.fields['scale_0'][indices]
    s1 = data.fields['scale_1'][indices]
    s2 = data.fields['scale_2'][indices]

    # Stack all scale values for joint clustering
    all_scales = np.concatenate([s0, s1, s2]).reshape(-1, 1)

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    kmeans.fit(all_scales)

    # Sort centroids for better compression
    centroids = kmeans.cluster_centers_.flatten()
    sort_order = np.argsort(centroids)
    centroids_sorted = centroids[sort_order]

    # Create inverse mapping
    inv_order = np.empty_like(sort_order)
    inv_order[sort_order] = np.arange(len(sort_order))

    # Assign labels
    n = len(indices)
    labels = kmeans.predict(np.concatenate([s0, s1, s2]).reshape(-1, 1))
    labels = inv_order[labels]  # Remap to sorted order

    labels_0 = labels[:n].astype(np.uint8)
    labels_1 = labels[n:2*n].astype(np.uint8)
    labels_2 = labels[2*n:].astype(np.uint8)

    # Pack into RGBA texture
    tex_size = width * height
    scales_tex = np.zeros((tex_size, 4), dtype=np.uint8)
    scales_tex[:n, 0] = labels_0
    scales_tex[:n, 1] = labels_1
    scales_tex[:n, 2] = labels_2
    scales_tex[:n, 3] = 255

    webp = encode_webp_lossless(scales_tex.flatten(), width, height)

    meta = {
        'codebook': centroids_sorted.tolist(),
        'files': ['scales.webp']
    }

    return webp, meta


def encode_sh0(data: SplatData, indices: np.ndarray, width: int, height: int,
               n_clusters: int = 256) -> Tuple[bytes, Dict]:
    """Encode SH DC coefficients (color) and opacity using k-means clustering."""
    print("  Encoding colors and opacity (k-means clustering)...")

    c0 = data.fields['f_dc_0'][indices]
    c1 = data.fields['f_dc_1'][indices]
    c2 = data.fields['f_dc_2'][indices]
    opacity = data.fields['opacity'][indices]

    # Stack all color values for joint clustering
    all_colors = np.concatenate([c0, c1, c2]).reshape(-1, 1)

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    kmeans.fit(all_colors)

    # Sort centroids
    centroids = kmeans.cluster_centers_.flatten()
    sort_order = np.argsort(centroids)
    centroids_sorted = centroids[sort_order]

    # Create inverse mapping
    inv_order = np.empty_like(sort_order)
    inv_order[sort_order] = np.arange(len(sort_order))

    # Assign labels
    n = len(indices)
    labels = kmeans.predict(np.concatenate([c0, c1, c2]).reshape(-1, 1))
    labels = inv_order[labels]

    labels_0 = labels[:n].astype(np.uint8)
    labels_1 = labels[n:2*n].astype(np.uint8)
    labels_2 = labels[2*n:].astype(np.uint8)

    # Convert opacity logit to [0, 255]
    opacity_sigmoid = sigmoid(opacity)
    opacity_u8 = (np.clip(opacity_sigmoid, 0, 1) * 255).astype(np.uint8)

    # Pack into RGBA texture
    tex_size = width * height
    sh0_tex = np.zeros((tex_size, 4), dtype=np.uint8)
    sh0_tex[:n, 0] = labels_0
    sh0_tex[:n, 1] = labels_1
    sh0_tex[:n, 2] = labels_2
    sh0_tex[:n, 3] = opacity_u8

    webp = encode_webp_lossless(sh0_tex.flatten(), width, height)

    meta = {
        'codebook': centroids_sorted.tolist(),
        'files': ['sh0.webp']
    }

    return webp, meta


def encode_motion(data: SplatData, indices: np.ndarray, width: int, height: int) -> Tuple[bytes, bytes, Dict]:
    """Encode motion vectors using 16-bit quantization (similar to means)."""
    print("  Encoding motion vectors...")

    m0 = data.fields['motion_0'][indices]
    m1 = data.fields['motion_1'][indices]
    m2 = data.fields['motion_2'][indices]

    # Apply log transform (same as position)
    m0_log = log_transform(m0)
    m1_log = log_transform(m1)
    m2_log = log_transform(m2)

    # Compute min/max
    mins = [float(m0_log.min()), float(m1_log.min()), float(m2_log.min())]
    maxs = [float(m0_log.max()), float(m1_log.max()), float(m2_log.max())]

    # Quantize to 16-bit
    m0_q = quantize_16bit(m0_log, mins[0], maxs[0])
    m1_q = quantize_16bit(m1_log, mins[1], maxs[1])
    m2_q = quantize_16bit(m2_log, mins[2], maxs[2])

    # Pack into two RGBA textures
    n = len(indices)
    tex_size = width * height
    motion_l = np.zeros((tex_size, 4), dtype=np.uint8)
    motion_u = np.zeros((tex_size, 4), dtype=np.uint8)

    motion_l[:n, 0] = m0_q & 0xFF
    motion_l[:n, 1] = m1_q & 0xFF
    motion_l[:n, 2] = m2_q & 0xFF
    motion_l[:n, 3] = 255

    motion_u[:n, 0] = (m0_q >> 8) & 0xFF
    motion_u[:n, 1] = (m1_q >> 8) & 0xFF
    motion_u[:n, 2] = (m2_q >> 8) & 0xFF
    motion_u[:n, 3] = 255

    webp_l = encode_webp_lossless(motion_l.flatten(), width, height)
    webp_u = encode_webp_lossless(motion_u.flatten(), width, height)

    meta = {
        'mins': mins,
        'maxs': maxs,
        'files': ['motion_l.webp', 'motion_u.webp']
    }

    return webp_l, webp_u, meta


def encode_trbf(data: SplatData, indices: np.ndarray, width: int, height: int) -> Tuple[bytes, bytes, Dict]:
    """Encode TRBF parameters (trbf_center, trbf_scale) using 16-bit quantization."""
    print("  Encoding TRBF parameters...")

    tc = data.fields['trbf_center'][indices]
    ts = data.fields['trbf_scale'][indices]

    # Apply log transform to trbf_scale (it's already exp'd, so log it back for better distribution)
    ts_log = np.log(np.maximum(ts, 1e-8))

    # Compute min/max
    tc_min, tc_max = float(tc.min()), float(tc.max())
    ts_min, ts_max = float(ts_log.min()), float(ts_log.max())

    # Quantize to 16-bit
    tc_q = quantize_16bit(tc, tc_min, tc_max)
    ts_q = quantize_16bit(ts_log, ts_min, ts_max)

    # Pack into two RGBA textures (RG channels used)
    n = len(indices)
    tex_size = width * height
    trbf_l = np.zeros((tex_size, 4), dtype=np.uint8)
    trbf_u = np.zeros((tex_size, 4), dtype=np.uint8)

    trbf_l[:n, 0] = tc_q & 0xFF
    trbf_l[:n, 1] = ts_q & 0xFF
    trbf_l[:n, 2] = 0
    trbf_l[:n, 3] = 255

    trbf_u[:n, 0] = (tc_q >> 8) & 0xFF
    trbf_u[:n, 1] = (ts_q >> 8) & 0xFF
    trbf_u[:n, 2] = 0
    trbf_u[:n, 3] = 255

    webp_l = encode_webp_lossless(trbf_l.flatten(), width, height)
    webp_u = encode_webp_lossless(trbf_u.flatten(), width, height)

    meta = {
        'encoding': 'quantize16',
        'center_min': tc_min,
        'center_max': tc_max,
        'scale_min': ts_min,  # This is log(trbf_scale)
        'scale_max': ts_max,
        'files': ['trbf_l.webp', 'trbf_u.webp']
    }

    return webp_l, webp_u, meta


def encode_trbf_kmeans(data: SplatData, indices: np.ndarray, width: int, height: int,
                       n_clusters: int = 256) -> Tuple[bytes, Dict]:
    """Encode TRBF parameters using k-means clustering (higher compression ratio)."""
    print("  Encoding TRBF parameters (k-means clustering)...")

    tc = data.fields['trbf_center'][indices]
    ts = data.fields['trbf_scale'][indices]

    # Apply log transform to trbf_scale for better distribution
    ts_log = np.log(np.maximum(ts, 1e-8))

    # K-means clustering for trbf_center
    tc_reshaped = tc.reshape(-1, 1)
    kmeans_tc = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    kmeans_tc.fit(tc_reshaped)
    
    # Sort centroids for better compression
    centroids_tc = kmeans_tc.cluster_centers_.flatten()
    sort_order_tc = np.argsort(centroids_tc)
    centroids_tc_sorted = centroids_tc[sort_order_tc]
    
    # Create inverse mapping
    inv_order_tc = np.empty_like(sort_order_tc)
    inv_order_tc[sort_order_tc] = np.arange(len(sort_order_tc))
    
    # Assign labels
    labels_tc = kmeans_tc.predict(tc_reshaped)
    labels_tc = inv_order_tc[labels_tc].astype(np.uint8)

    # K-means clustering for trbf_scale (log space)
    ts_reshaped = ts_log.reshape(-1, 1)
    kmeans_ts = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    kmeans_ts.fit(ts_reshaped)
    
    # Sort centroids
    centroids_ts = kmeans_ts.cluster_centers_.flatten()
    sort_order_ts = np.argsort(centroids_ts)
    centroids_ts_sorted = centroids_ts[sort_order_ts]
    
    # Create inverse mapping
    inv_order_ts = np.empty_like(sort_order_ts)
    inv_order_ts[sort_order_ts] = np.arange(len(sort_order_ts))
    
    # Assign labels
    labels_ts = kmeans_ts.predict(ts_reshaped)
    labels_ts = inv_order_ts[labels_ts].astype(np.uint8)

    # Pack into single RGBA texture (RG channels used, BA unused)
    n = len(indices)
    tex_size = width * height
    trbf_tex = np.zeros((tex_size, 4), dtype=np.uint8)
    trbf_tex[:n, 0] = labels_tc
    trbf_tex[:n, 1] = labels_ts
    trbf_tex[:n, 2] = 0
    trbf_tex[:n, 3] = 255

    webp = encode_webp_lossless(trbf_tex.flatten(), width, height)

    meta = {
        'encoding': 'kmeans',
        'center_codebook': centroids_tc_sorted.tolist(),
        'scale_codebook': centroids_ts_sorted.tolist(),  # Values are log(trbf_scale)
        'files': ['trbf.webp']
    }

    return webp, meta


# =============================================================================
# Segment computation
# =============================================================================

def compute_segments(data: SplatData, cfg: Dict[str, float], segment_duration: float,
                     opacity_threshold: float, morton_indices: np.ndarray) -> List[Tuple[Dict, bytes]]:
    """Compute time segments and their active indices.
    
    Args:
        data: Splat data (original order)
        cfg: Config with start, duration, fps
        segment_duration: Duration of each segment
        opacity_threshold: Threshold for considering a splat active
        morton_indices: Morton sorted indices (morton_indices[morton_idx] = original_idx)
    
    Returns:
        List of (segment_info, indices_bytes) tuples where indices are in Morton order
    """
    if torch is None:
        print("\nWARNING: torch not found. Skipping segment generation.")
        print("Install torch for segment computation: pip install torch")
        return []

    print("\nComputing segments...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    # Create reverse mapping: original_idx -> morton_idx
    # morton_indices[morton_idx] = original_idx
    # So: reverse_map[original_idx] = morton_idx
    reverse_map = np.empty(data.num, dtype=np.uint32)
    reverse_map[morton_indices] = np.arange(data.num, dtype=np.uint32)

    opacity_logit = torch.from_numpy(data.fields['opacity']).to(device).squeeze()
    tc = torch.from_numpy(data.fields['trbf_center']).to(device).squeeze()
    ts = torch.from_numpy(data.fields['trbf_scale']).to(device).squeeze()

    start = float(cfg.get('start', 0.0))
    duration = float(cfg.get('duration', 0.0))
    fps = float(cfg.get('fps', 30.0))

    num_segments = int(np.ceil(duration / segment_duration))
    segments = []

    for i in range(num_segments):
        t0 = start + i * segment_duration
        t1 = min(start + (i + 1) * segment_duration, start + duration)

        sample_times = torch.arange(t0, t1, 1.0 / fps, device=device)

        dt = sample_times.view(-1, 1) - tc.view(1, -1)
        dt_scaled = dt / torch.clamp(ts.view(1, -1), min=1e-6)
        gauss = torch.exp(-dt_scaled * dt_scaled)

        base_opacity = torch.sigmoid(opacity_logit).view(1, -1)
        dyn_alpha = base_opacity * gauss

        is_visible = torch.any(dyn_alpha > opacity_threshold, dim=0)
        # Get original indices that are visible
        original_indices = torch.where(is_visible)[0].cpu().numpy().astype(np.uint32)
        
        # Map original indices to Morton order indices
        morton_order_indices = reverse_map[original_indices]
        # Sort for better compression and access patterns
        morton_order_indices.sort()

        segment_info = {
            't0': t0 - start,
            't1': t1 - start,
            'count': len(morton_order_indices)
        }

        # Convert indices to bytes
        indices_bytes = morton_order_indices.tobytes()

        segments.append((segment_info, indices_bytes))
        print(f"  Segment {i}: [{t0-start:.2f}s - {t1-start:.2f}s], {len(morton_order_indices)} active splats")

    return segments


# =============================================================================
# Main export function
# =============================================================================

def write_sog4d(output_path: str, data: SplatData, cfg: Dict[str, float],
                segment_duration: float, opacity_threshold: float, trbf_kmeans: bool = True):
    """Write compressed .sog4d file.
    
    Args:
        output_path: Output file path
        data: Splat data
        cfg: Config with start, duration, fps
        segment_duration: Duration of each segment in seconds
        opacity_threshold: Opacity threshold for segment computation
        trbf_kmeans: If True, use k-means clustering for TRBF (better compression).
                     If False, use 16-bit quantization (higher precision).
    """

    print("\nSorting by Morton order...")
    indices = sort_morton_order(data)

    width, height = compute_texture_size(data.num)
    print(f"Texture size: {width} x {height} ({width * height} pixels for {data.num} splats)")

    print("\nEncoding attributes...")

    # Encode all attributes
    means_l, means_u, means_meta = encode_means(data, indices, width, height)
    quats, quats_meta = encode_quats(data, indices, width, height)
    scales, scales_meta = encode_scales(data, indices, width, height)
    sh0, sh0_meta = encode_sh0(data, indices, width, height)
    motion_l, motion_u, motion_meta = encode_motion(data, indices, width, height)
    
    # TRBF encoding: k-means (default) or 16-bit quantization
    if trbf_kmeans:
        trbf_data, trbf_meta = encode_trbf_kmeans(data, indices, width, height)
    else:
        trbf_l, trbf_u, trbf_meta = encode_trbf(data, indices, width, height)

    # Compute segments (pass morton indices for correct index mapping)
    segments = compute_segments(data, cfg, segment_duration, opacity_threshold, indices)

    # Build meta.json
    meta = {
        'version': 1,
        'type': 'sog4d',
        'generator': 'ply_to_sog4d.py',
        'count': data.num,
        'width': width,
        'height': height,
        'sh_degree': data.sh_degree,

        # Dynamic gaussian parameters
        'start': float(cfg.get('start', 0.0)),
        'duration': float(cfg.get('duration', 0.0)),
        'fps': float(cfg.get('fps', 30.0)),

        # Static gaussian attributes (SOG format)
        'means': means_meta,
        'quats': quats_meta,
        'scales': scales_meta,
        'sh0': sh0_meta,

        # Dynamic gaussian attributes (new for sog4d)
        'motion': motion_meta,
        'trbf': trbf_meta,

        # Segments
        'segments': [{'t0': s[0]['t0'], 't1': s[0]['t1'], 'url': f'segments/seg_{i:03d}.act', 'count': s[0]['count']}
                     for i, s in enumerate(segments)]
    }

    # Write ZIP file
    print(f"\nWriting {output_path}...")

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Meta
        zf.writestr('meta.json', json.dumps(meta, indent=2))

        # Static gaussian textures
        zf.writestr('means_l.webp', means_l)
        zf.writestr('means_u.webp', means_u)
        zf.writestr('quats.webp', quats)
        zf.writestr('scales.webp', scales)
        zf.writestr('sh0.webp', sh0)

        # Dynamic gaussian textures
        zf.writestr('motion_l.webp', motion_l)
        zf.writestr('motion_u.webp', motion_u)
        
        # TRBF textures (different files depending on encoding mode)
        if trbf_kmeans:
            zf.writestr('trbf.webp', trbf_data)
        else:
            zf.writestr('trbf_l.webp', trbf_l)
            zf.writestr('trbf_u.webp', trbf_u)

        # Segments
        for i, (_, indices_bytes) in enumerate(segments):
            zf.writestr(f'segments/seg_{i:03d}.act', indices_bytes)

    # Print size info
    file_size = os.path.getsize(output_path)
    print(f"\nOutput: {output_path}")
    print(f"Size: {file_size / 1024 / 1024:.2f} MB")
    print(f"Splats: {data.num}")
    print(f"Duration: {cfg.get('duration', 0):.2f}s @ {cfg.get('fps', 30)} fps")
    print(f"Segments: {len(segments)}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Convert dynamic gaussian PLY to compressed .sog4d format'
    )
    parser.add_argument('--cfg_args', required=True,
                        help='Path to cfg_args file (contains start, duration, fps, sh_degree)')
    parser.add_argument('--ply', required=True,
                        help='Path to point_cloud.ply (dynamic gaussian)')
    parser.add_argument('--output', '-o', required=True,
                        help='Output .sog4d file path')
    parser.add_argument('--max_splats', type=int, default=0,
                        help='If >0, randomly sample this many splats (for debugging)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for --max_splats sampling')
    parser.add_argument('--segment_duration', type=float, default=0.5,
                        help='Duration of each segment in seconds (default: 0.5)')
    parser.add_argument('--opacity_threshold', type=float, default=0.005,
                        help='Opacity threshold to consider a splat active (default: 0.005)')
    parser.add_argument('--no-trbf-kmeans', action='store_true',
                        help='Disable k-means clustering for TRBF (use 16-bit quantization instead)')
    parser.add_argument('--filter_opacity', action='store_true',
                        help='Disable filtering of low opacity splats')

    args = parser.parse_args()

    # Parse cfg_args
    with open(args.cfg_args, 'r', encoding='utf-8') as f:
        cfg_text = f.read()
    cfg = parse_cfg_args_text(cfg_text)

    sh_degree = int(cfg.get('sh_degree', 0))

    # Load PLY
    max_splats = args.max_splats if args.max_splats > 0 else None
    data = load_ply_dynamic(args.ply, sh_degree=sh_degree, max_splats=max_splats, seed=args.seed)

    # Filter out low opacity splats (those invisible across all frames)
    if args.filter_opacity:
        opacity_threshold = args.opacity_threshold
        data = filter_low_opacity_splats(data, cfg, opacity_threshold)

    # Ensure output has .sog4d extension
    output_path = args.output
    if not output_path.lower().endswith('.sog4d'):
        output_path += '.sog4d'

    # Write sog4d
    trbf_kmeans = not getattr(args, 'no_trbf_kmeans', False)
    write_sog4d(output_path, data, cfg, args.segment_duration, args.opacity_threshold, trbf_kmeans)

    print("\nDone!")


if __name__ == '__main__':
    main()

# Example usage:
# python ply_to_sog4d.py --cfg_args "path/to/cfg_args" --ply "path/to/point_cloud.ply" -o "output.sog4d"
