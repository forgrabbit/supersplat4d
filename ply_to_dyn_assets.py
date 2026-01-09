import argparse
import json
import os
import re
import struct
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from plyfile import PlyData

# -----------------------------
# cfg_args parsing
# -----------------------------

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
            pass
            
    required_keys = ['start', 'duration', 'fps']
    if not all(k in out for k in required_keys):
        missing = [k for k in required_keys if k not in out]
        raise ValueError(f"Could not parse required keys {missing} from cfg_args. Found: {list(out.keys())}")

    return out

# -----------------------------
# dyn assets format
# -----------------------------

MAGIC = b"DYGS"
VERSION = 1

@dataclass
class GlobalArrays:
    num: int
    sh_degree: int
    fields: Dict[str, np.ndarray]

def _require(p: PlyData, names: List[str]):
    missing = [n for n in names if n not in p.elements[0].data.dtype.names]
    if missing:
        raise ValueError(f"PLY missing properties: {missing}")

def _apply_subset(fields: Dict[str, np.ndarray], idx: np.ndarray) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k, v in fields.items():
        out[k] = v[idx]
    return out

def load_ply_dynamic(ply_path: str, sh_degree: int, max_splats: int | None = None, seed: int = 0) -> GlobalArrays:
    ply = PlyData.read(ply_path)
    v = ply.elements[0]
    names = v.data.dtype.names

    base_req = [
        'x', 'y', 'z',
        'scale_0', 'scale_1', 'scale_2',
        'rot_0', 'rot_1', 'rot_2', 'rot_3',
        'f_dc_0', 'f_dc_1', 'f_dc_2',
        'opacity',
        'trbf_center', 'trbf_scale'
    ]

    has_motion3 = all(n in names for n in ['motion_0', 'motion_1', 'motion_2'])
    if not has_motion3:
        raise ValueError('PLY missing motion_0..2')

    _require(ply, base_req)

    num_full = v.count

    def f32(name: str) -> np.ndarray:
        return np.asarray(v.data[name], dtype=np.float32)

    fields: Dict[str, np.ndarray] = {}
    for k in ['x', 'y', 'z', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'trbf_center']:
        fields[k] = f32(k)

    fields['trbf_scale'] = np.exp(f32('trbf_scale'))
    fields['opacity_logit'] = f32('opacity')

    fields['motion_0'] = f32('motion_0')
    fields['motion_1'] = f32('motion_1')
    fields['motion_2'] = f32('motion_2')

    if sh_degree > 0:
        sh_coeffs = (sh_degree + 1) ** 2
        rest = sh_coeffs - 1
        rest_names = [f'f_rest_{i}' for i in range(rest * 3)]
        if all(n in names for n in rest_names):
            rest_data = np.stack([f32(n) for n in rest_names], axis=1)
            fields['sh_rest'] = rest_data.reshape(num_full, rest * 3)
        else:
            fields['sh_rest'] = np.zeros((num_full, rest * 3), dtype=np.float32)

    if max_splats is not None and 0 < max_splats < num_full:
        rng = np.random.default_rng(seed)
        idx = rng.choice(num_full, size=max_splats, replace=False)
        idx.sort()
        fields = _apply_subset(fields, idx)
        num = max_splats
    else:
        num = num_full

    return GlobalArrays(num=num, sh_degree=sh_degree, fields=fields)

def write_global_bin(out_path: str, g: GlobalArrays):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    header = struct.pack('<4sIIII', MAGIC, VERSION, g.num, g.sh_degree, 0)
    with open(out_path, 'wb') as f:
        f.write(header)
        order = ['x', 'y', 'z', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity_logit', 'motion_0', 'motion_1', 'motion_2', 'trbf_center', 'trbf_scale']
        for name in order:
            arr = g.fields[name].astype(np.float32, copy=False)
            f.write(arr.tobytes(order='C'))
        if g.sh_degree > 0:
            f.write(g.fields['sh_rest'].astype(np.float32, copy=False).tobytes(order='C'))

def compute_and_write_segments(out_dir: str, g: GlobalArrays, cfg: Dict[str, float], segment_duration: float, opacity_threshold: float) -> List[Dict]:
    if torch is None:
        print("\nWARNING: torch not found. Skipping segment generation. Install torch for this feature.")
        return []

    print("\nComputing segments...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    opacity_logit = torch.from_numpy(g.fields['opacity_logit']).to(device).squeeze()
    tc = torch.from_numpy(g.fields['trbf_center']).to(device).squeeze()
    ts = torch.from_numpy(g.fields['trbf_scale']).to(device).squeeze()

    start = float(cfg.get('start', 0.0))
    duration = float(cfg.get('duration', 0.0))
    fps = float(cfg.get('fps', 30.0))

    num_segments = int(np.ceil(duration / segment_duration))
    segment_list = []

    segments_dir = os.path.join(out_dir, 'segments')
    os.makedirs(segments_dir, exist_ok=True)

    for i in range(num_segments):
        t0 = start + i * segment_duration
        t1 = min(start + (i + 1) * segment_duration, start + duration)

        sample_times = torch.arange(t0, t1, 1.0 / fps, device=device)

        dt = sample_times.view(-1, 1) - tc.view(1, -1)
        dt_scaled = dt / torch.clamp(ts.view(1, -1), min=1e-6)
        # Match SIBR viewer CUDA: exp(-dt_scaled^2)
        gauss = torch.exp(-dt_scaled * dt_scaled)

        base_opacity = torch.sigmoid(opacity_logit).view(1, -1)
        dyn_alpha = base_opacity * gauss

        is_visible_at_any_time = torch.any(dyn_alpha > opacity_threshold, dim=0)
        active_indices = torch.where(is_visible_at_any_time)[0].cpu().numpy().astype(np.uint32)

        act_filename = f"seg_{i:03d}.act"
        act_path = os.path.join(segments_dir, act_filename)
        with open(act_path, 'wb') as f:
            f.write(active_indices.tobytes())

        segment_info = {
            "t0": t0 - start,
            "t1": t1 - start,
            "url": f"segments/{act_filename}",
            "count": len(active_indices)
        }
        segment_list.append(segment_info)
        print(f"  Segment {i}: [{t0:.2f}s - {t1:.2f}s], {len(active_indices)} active splats")

    return segment_list

def write_manifest(out_path: str, cfg: Dict[str, float], global_bin_name: str, num_splats: int, sh_degree: int, segments: List[Dict]):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    manifest = {
        'version': 1,
        'type': 'dyn',
        'start': float(cfg.get('start', 0.0)),
        'duration': float(cfg.get('duration', 0.0)),
        'fps': float(cfg.get('fps', 30.0)),
        'sh_degree': int(sh_degree),
        'global': {
            'url': global_bin_name,
            'numSplats': int(num_splats)
        },
        'segments': segments
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg_args', required=True, help='Path to cfg_args file')
    ap.add_argument('--ply', required=True, help='Path to point_cloud.ply (dynamic)')
    ap.add_argument('--out_dir', required=True, help='Output directory for dyn assets')
    ap.add_argument('--out_name', default='scene.dyn.json', help='Manifest filename')
    ap.add_argument('--max_splats', type=int, default=0, help='If >0, randomly sample this many splats for debugging')
    ap.add_argument('--seed', type=int, default=0, help='Random seed for --max_splats sampling')
    ap.add_argument('--segment_duration', type=float, default=0.5, help='Duration of each segment in seconds.')
    ap.add_argument('--opacity_threshold', type=float, default=0.005, help='Opacity threshold to consider a splat active.')
    args = ap.parse_args()

    with open(args.cfg_args, 'r', encoding='utf-8') as f:
        cfg_text = f.read()
    cfg = parse_cfg_args_text(cfg_text)

    sh_degree = int(cfg.get('sh_degree', 0))

    max_splats = args.max_splats if args.max_splats and args.max_splats > 0 else None
    g = load_ply_dynamic(args.ply, sh_degree=sh_degree, max_splats=max_splats, seed=args.seed)

    out_dir = args.out_dir
    global_bin = os.path.join(out_dir, 'global.bin')
    manifest_path = os.path.join(out_dir, args.out_name)

    write_global_bin(global_bin, g)
    segments = compute_and_write_segments(out_dir, g, cfg, args.segment_duration, args.opacity_threshold)
    write_manifest(manifest_path, cfg, 'global.bin', g.num, sh_degree, segments)

    print('\nWrote:')
    print(' ', manifest_path)
    print(' ', global_bin)
    print('numSplats:', g.num)
    print('start/duration/fps:', cfg.get('start'), cfg.get('duration'), cfg.get('fps'))
    if max_splats is not None:
        print('sampled:', max_splats, 'seed:', args.seed)

if __name__ == '__main__':
    main()

# python .\ply_to_dyn_assets.py --cfg_args "E:\3dgs\stg_lite\output\JD_test\ruan_60_180000\cfg_args" --ply      "E:\3dgs\stg_lite\output\JD_test\ruan_60_180000\point_cloud\iteration_79\point_cloud.ply" --out_dir  "E:\3dgs\splat4d_viewer\dyn_pkg" --max_splats -1 --segment_duration 0.2
