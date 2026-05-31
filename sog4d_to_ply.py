#!/usr/bin/env python3
"""
Convert compressed .sog4d/.sog archives back to binary little-endian PLY.

The reconstruction is necessarily approximate: SOG4D stores quantized positions,
compressed quaternions, and K-means labels/codebooks for several attributes.
This script decodes exactly what is present in the archive.
"""

import argparse
import io
import json
import math
import os
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


ArrayMap = Dict[str, np.ndarray]


STATIC_PROPERTY_ORDER = [
    "x", "y", "z",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity",
]

DYNAMIC_PROPERTY_ORDER = [
    "x", "y", "z",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity",
    "trbf_center", "trbf_scale",
    "motion_0", "motion_1", "motion_2",
]


def zip_join(prefix: str, name: str) -> str:
    prefix = prefix.replace("\\", "/").strip("/")
    name = name.replace("\\", "/").lstrip("/")
    return f"{prefix}/{name}" if prefix else name


def read_json(zf: zipfile.ZipFile, path: str) -> Dict:
    return json.loads(zf.read(path).decode("utf-8"))


def decode_rgba(zf: zipfile.ZipFile, path: str) -> Tuple[np.ndarray, int, int]:
    with zf.open(path, "r") as f:
        data = f.read()

    img = Image.open(io.BytesIO(data))
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    rgba = np.array(img, dtype=np.uint8, copy=True)
    return rgba.reshape(-1, 4), img.width, img.height


def inv_log_transform(values: np.ndarray) -> np.ndarray:
    return (np.sign(values) * np.expm1(np.abs(values))).astype(np.float32)


def dequantize16(lo: np.ndarray, hi: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    q = lo.astype(np.uint16) | (hi.astype(np.uint16) << 8)
    scale = float(vmax) - float(vmin)
    if scale == 0.0:
        scale = 1.0
    return (float(vmin) + (q.astype(np.float32) / 65535.0) * scale).astype(np.float32)


def sigmoid_inv(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(np.float32), 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values)).astype(np.float32)


def field_files(meta_section: Dict, fallback: Sequence[str]) -> Sequence[str]:
    return meta_section.get("files") or fallback


def decode_means(zf: zipfile.ZipFile, prefix: str, meta: Dict, count: int) -> ArrayMap:
    files = field_files(meta["means"], ["means_l.webp", "means_u.webp"])
    means_l, _, _ = decode_rgba(zf, zip_join(prefix, files[0]))
    means_u, _, _ = decode_rgba(zf, zip_join(prefix, files[1]))
    means_l = means_l[:count]
    means_u = means_u[:count]

    mins = meta["means"]["mins"]
    maxs = meta["means"]["maxs"]

    x_log = dequantize16(means_l[:, 0], means_u[:, 0], mins[0], maxs[0])
    y_log = dequantize16(means_l[:, 1], means_u[:, 1], mins[1], maxs[1])
    z_log = dequantize16(means_l[:, 2], means_u[:, 2], mins[2], maxs[2])

    return {
        "x": inv_log_transform(x_log),
        "y": inv_log_transform(y_log),
        "z": inv_log_transform(z_log),
    }


def decode_quats(zf: zipfile.ZipFile, prefix: str, meta: Dict, count: int) -> ArrayMap:
    files = field_files(meta["quats"], ["quats.webp"])
    quats, _, _ = decode_rgba(zf, zip_join(prefix, files[0]))
    quats = quats[:count]

    out = np.zeros((count, 4), dtype=np.float32)
    out[:, 0] = 1.0

    tags = quats[:, 3].astype(np.int16)
    modes = tags - 252
    values = ((quats[:, :3].astype(np.float32) / 255.0) * 2.0 - 1.0) / math.sqrt(2.0)
    idx_map = (
        (1, 2, 3),
        (0, 2, 3),
        (0, 1, 3),
        (0, 1, 2),
    )

    for mode, idx in enumerate(idx_map):
        mask = modes == mode
        if not np.any(mask):
            continue

        out[mask] = 0.0
        out[mask, idx[0]] = values[mask, 0]
        out[mask, idx[1]] = values[mask, 1]
        out[mask, idx[2]] = values[mask, 2]
        sum_sq = np.sum(out[mask] * out[mask], axis=1)
        out[mask, mode] = np.sqrt(np.maximum(0.0, 1.0 - sum_sq))

    return {
        "rot_0": out[:, 0],
        "rot_1": out[:, 1],
        "rot_2": out[:, 2],
        "rot_3": out[:, 3],
    }


def decode_scales(zf: zipfile.ZipFile, prefix: str, meta: Dict, count: int) -> ArrayMap:
    files = field_files(meta["scales"], ["scales.webp"])
    scales, _, _ = decode_rgba(zf, zip_join(prefix, files[0]))
    scales = scales[:count]

    if "codebook" not in meta["scales"]:
        mins = meta["scales"]["mins"]
        maxs = meta["scales"]["maxs"]
        return {
            "scale_0": (mins[0] + (scales[:, 0].astype(np.float32) / 255.0) * (maxs[0] - mins[0])).astype(np.float32),
            "scale_1": (mins[1] + (scales[:, 1].astype(np.float32) / 255.0) * (maxs[1] - mins[1])).astype(np.float32),
            "scale_2": (mins[2] + (scales[:, 2].astype(np.float32) / 255.0) * (maxs[2] - mins[2])).astype(np.float32),
        }

    codebook = np.asarray(meta["scales"]["codebook"], dtype=np.float32)
    return {
        "scale_0": codebook[scales[:, 0]],
        "scale_1": codebook[scales[:, 1]],
        "scale_2": codebook[scales[:, 2]],
    }


def decode_sh0(zf: zipfile.ZipFile, prefix: str, meta: Dict, count: int) -> ArrayMap:
    files = field_files(meta["sh0"], ["sh0.webp"])
    sh0, _, _ = decode_rgba(zf, zip_join(prefix, files[0]))
    sh0 = sh0[:count]

    if "codebook" not in meta["sh0"]:
        mins = meta["sh0"]["mins"]
        maxs = meta["sh0"]["maxs"]
        c0 = mins[0] + (sh0[:, 0].astype(np.float32) / 255.0) * (maxs[0] - mins[0])
        c1 = mins[1] + (sh0[:, 1].astype(np.float32) / 255.0) * (maxs[1] - mins[1])
        c2 = mins[2] + (sh0[:, 2].astype(np.float32) / 255.0) * (maxs[2] - mins[2])
        opacity = mins[3] + (sh0[:, 3].astype(np.float32) / 255.0) * (maxs[3] - mins[3])
        return {
            "f_dc_0": c0.astype(np.float32),
            "f_dc_1": c1.astype(np.float32),
            "f_dc_2": c2.astype(np.float32),
            "opacity": opacity.astype(np.float32),
        }

    codebook = np.asarray(meta["sh0"]["codebook"], dtype=np.float32)
    alpha = sh0[:, 3].astype(np.float32) / 255.0
    return {
        "f_dc_0": codebook[sh0[:, 0]],
        "f_dc_1": codebook[sh0[:, 1]],
        "f_dc_2": codebook[sh0[:, 2]],
        "opacity": sigmoid_inv(alpha),
    }


def decode_motion(zf: zipfile.ZipFile, prefix: str, meta: Dict, count: int) -> ArrayMap:
    files = field_files(meta["motion"], ["motion_l.webp", "motion_u.webp"])
    motion_l, _, _ = decode_rgba(zf, zip_join(prefix, files[0]))
    motion_u, _, _ = decode_rgba(zf, zip_join(prefix, files[1]))
    motion_l = motion_l[:count]
    motion_u = motion_u[:count]

    mins = meta["motion"]["mins"]
    maxs = meta["motion"]["maxs"]
    m0_log = dequantize16(motion_l[:, 0], motion_u[:, 0], mins[0], maxs[0])
    m1_log = dequantize16(motion_l[:, 1], motion_u[:, 1], mins[1], maxs[1])
    m2_log = dequantize16(motion_l[:, 2], motion_u[:, 2], mins[2], maxs[2])

    return {
        "motion_0": inv_log_transform(m0_log),
        "motion_1": inv_log_transform(m1_log),
        "motion_2": inv_log_transform(m2_log),
    }


def decode_trbf(zf: zipfile.ZipFile, prefix: str, meta: Dict, count: int) -> ArrayMap:
    trbf_meta = meta["trbf"]
    encoding = trbf_meta.get("encoding", "quantize16")

    if encoding == "kmeans":
        files = field_files(trbf_meta, ["trbf.webp"])
        trbf, _, _ = decode_rgba(zf, zip_join(prefix, files[0]))
        trbf = trbf[:count]
        center_codebook = np.asarray(trbf_meta["center_codebook"], dtype=np.float32)
        scale_codebook = np.asarray(trbf_meta["scale_codebook"], dtype=np.float32)
        return {
            "trbf_center": center_codebook[trbf[:, 0]],
            # PLY stores log(trbf_scale). The viewer expands it at load time.
            "trbf_scale": scale_codebook[trbf[:, 1]],
        }

    files = field_files(trbf_meta, ["trbf_l.webp", "trbf_u.webp"])
    trbf_l, _, _ = decode_rgba(zf, zip_join(prefix, files[0]))
    trbf_u, _, _ = decode_rgba(zf, zip_join(prefix, files[1]))
    trbf_l = trbf_l[:count]
    trbf_u = trbf_u[:count]

    return {
        "trbf_center": dequantize16(trbf_l[:, 0], trbf_u[:, 0], trbf_meta["center_min"], trbf_meta["center_max"]),
        "trbf_scale": dequantize16(trbf_l[:, 1], trbf_u[:, 1], trbf_meta["scale_min"], trbf_meta["scale_max"]),
    }


def sh_rest_count_from_degree(sh_degree: int) -> int:
    return ((int(sh_degree) + 1) ** 2 - 1) * 3


def add_zero_sh_rest(fields: ArrayMap, count: int, sh_degree: int) -> None:
    for idx in range(sh_rest_count_from_degree(sh_degree)):
        fields[f"f_rest_{idx}"] = np.zeros(count, dtype=np.float32)


def decode_shN(zf: zipfile.ZipFile, prefix: str, meta: Dict, count: int) -> ArrayMap:
    shn_meta = meta.get("shN")
    if not shn_meta:
        return {}

    bands = int(shn_meta.get("bands", 0))
    rest_coeffs = (bands + 1) ** 2 - 1
    if rest_coeffs <= 0:
        return {}

    files = field_files(shn_meta, ["shN_centroids.webp", "shN_labels.webp"])
    centroids, centroids_width, _ = decode_rgba(zf, zip_join(prefix, files[0]))
    labels, _, _ = decode_rgba(zf, zip_join(prefix, files[1]))
    labels = labels[:count]

    if "codebook" not in shn_meta:
        raise ValueError("Only SOG v2 shN codebook decoding is supported")

    codebook = np.asarray(shn_meta["codebook"], dtype=np.float32)
    label_ids = labels[:, 0].astype(np.uint16) | (labels[:, 1].astype(np.uint16) << 8)
    label_ids = label_ids.astype(np.int64)

    base_u = (label_ids % 64) * rest_coeffs
    v = label_ids // 64
    sh_rest = np.empty((count, rest_coeffs * 3), dtype=np.float32)

    for coeff in range(rest_coeffs):
        centroid_idx = v * centroids_width + base_u + coeff
        texels = centroids[centroid_idx]
        sh_rest[:, coeff * 3 + 0] = codebook[texels[:, 0]]
        sh_rest[:, coeff * 3 + 1] = codebook[texels[:, 1]]
        sh_rest[:, coeff * 3 + 2] = codebook[texels[:, 2]]

    return {f"f_rest_{idx}": sh_rest[:, idx] for idx in range(rest_coeffs * 3)}


def decode_static_sog(zf: zipfile.ZipFile, prefix: str, meta: Dict) -> ArrayMap:
    count = int(meta["count"])
    fields: ArrayMap = {}
    fields.update(decode_means(zf, prefix, meta, count))
    fields.update(decode_scales(zf, prefix, meta, count))
    fields.update(decode_quats(zf, prefix, meta, count))
    fields.update(decode_sh0(zf, prefix, meta, count))
    fields.update(decode_shN(zf, prefix, meta, count))
    return fields


def decode_dynamic_sog4d(zf: zipfile.ZipFile, prefix: str, meta: Dict) -> ArrayMap:
    count = int(meta["count"])
    fields = decode_static_sog(zf, prefix, meta)
    fields.update(decode_motion(zf, prefix, meta, count))
    fields.update(decode_trbf(zf, prefix, meta, count))

    if "shN" not in meta and int(meta.get("sh_degree", 0)) > 0:
        add_zero_sh_rest(fields, count, int(meta["sh_degree"]))

    return fields


def property_names(fields: ArrayMap, dynamic: bool) -> List[str]:
    order = DYNAMIC_PROPERTY_ORDER if dynamic else STATIC_PROPERTY_ORDER
    names = [name for name in order if name in fields]

    rest_names = [
        name for name in fields
        if name.startswith("f_rest_")
    ]
    rest_names.sort(key=lambda x: int(x.rsplit("_", 1)[1]))
    names.extend(rest_names)

    extras = sorted(name for name in fields if name not in set(names))
    names.extend(extras)
    return names


def write_binary_ply(
    path: Path,
    fields: ArrayMap,
    dynamic: bool,
    comments: Optional[Sequence[str]] = None,
    overwrite: bool = False,
    chunk_size: int = 65536,
) -> None:
    if path.exists() and not overwrite:
        print(f"  skip existing: {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    names = property_names(fields, dynamic)
    if not names:
        raise ValueError("No fields to write")

    count = int(len(fields[names[0]]))
    for name in names:
        if len(fields[name]) != count:
            raise ValueError(f"Field {name} has {len(fields[name])} values, expected {count}")

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
    ]
    for comment in comments or []:
        comment = comment.strip()
        if comment:
            header_lines.append(f"comment {comment}")

    header_lines.extend([
        f"element vertex {count}",
        *[f"property float {name}" for name in names],
        "end_header",
        "",
    ])

    arrays = [np.asarray(fields[name], dtype="<f4") for name in names]

    with path.open("wb") as f:
        f.write("\n".join(header_lines).encode("ascii"))

        for start in range(0, count, chunk_size):
            end = min(start + chunk_size, count)
            block = np.empty((end - start, len(arrays)), dtype="<f4")
            for col, arr in enumerate(arrays):
                block[:, col] = arr[start:end]
            f.write(block.tobytes(order="C"))

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  wrote: {path} ({count} splats, {size_mb:.2f} MB)")


def cfg_comment(meta: Dict) -> str:
    parts = [
        f"start={float(meta.get('start', 0.0))}",
        f"duration={float(meta.get('duration', 0.0))}",
        f"fps={float(meta.get('fps', 30.0))}",
        f"sh_degree={int(meta.get('sh_degree', 0))}",
    ]
    if "culling" in meta:
        parts.append(f"culling={float(meta['culling'])}")
    return "cfg_args: " + " ".join(parts)


def output_root_for(input_path: Path, output_dir: Path) -> Path:
    resolved_input = input_path.resolve()
    resolved_cwd = Path.cwd().resolve()
    try:
        rel_parent = resolved_input.parent.relative_to(resolved_cwd)
    except ValueError:
        rel_parent = Path()
    return output_dir / rel_parent / input_path.stem


def convert_archive(input_path: Path, output_dir: Path, overwrite: bool = False) -> List[Path]:
    written: List[Path] = []
    base_out = output_root_for(input_path, output_dir)
    print(f"Converting: {input_path}")

    with zipfile.ZipFile(input_path, "r") as zf:
        main_meta = read_json(zf, "meta.json")
        meta_type = main_meta.get("type")

        if meta_type == "sog4d_multi":
            dynamic_info = main_meta.get("dynamic")
            if dynamic_info:
                prefix = dynamic_info.get("path", "dynamic/")
                meta_path = dynamic_info.get("meta", zip_join(prefix, "meta.json"))
                dynamic_meta = read_json(zf, meta_path)
                fields = decode_dynamic_sog4d(zf, prefix, dynamic_meta)
                out_path = base_out / f"{input_path.stem}_dynamic.ply"
                write_binary_ply(out_path, fields, dynamic=True, comments=[cfg_comment(dynamic_meta)], overwrite=overwrite)
                written.append(out_path)

            for name, info in sorted((main_meta.get("static") or {}).items()):
                prefix = info.get("path", f"{name}/")
                meta_path = info.get("meta", zip_join(prefix, "meta.json"))
                static_meta = read_json(zf, meta_path)
                fields = decode_static_sog(zf, prefix, static_meta)
                out_path = base_out / f"{input_path.stem}_{name}.ply"
                write_binary_ply(out_path, fields, dynamic=False, overwrite=overwrite)
                written.append(out_path)

        elif meta_type == "sog4d":
            fields = decode_dynamic_sog4d(zf, "", main_meta)
            out_path = base_out / f"{input_path.stem}.ply"
            write_binary_ply(out_path, fields, dynamic=True, comments=[cfg_comment(main_meta)], overwrite=overwrite)
            written.append(out_path)

        elif "means" in main_meta and "quats" in main_meta:
            fields = decode_static_sog(zf, "", main_meta)
            out_path = base_out / f"{input_path.stem}.ply"
            write_binary_ply(out_path, fields, dynamic=False, overwrite=overwrite)
            written.append(out_path)

        else:
            raise ValueError(f"Unsupported archive type in {input_path}: {meta_type!r}")

    return written


def is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def discover_inputs(paths: Sequence[str], output_dir: Path, include_sog: bool) -> List[Path]:
    suffixes = {".sog4d"}
    if include_sog:
        suffixes.add(".sog")

    found: List[Path] = []
    search_paths = [Path(p) for p in paths] if paths else [Path.cwd()]

    for path in search_paths:
        if path.is_dir():
            for suffix in suffixes:
                found.extend(p for p in path.rglob(f"*{suffix}") if not is_under(p, output_dir))
        elif path.is_file() and path.suffix.lower() in suffixes:
            if not is_under(path, output_dir):
                found.append(path)
        else:
            print(f"Skipping unsupported path: {path}")

    unique: Dict[Path, Path] = {}
    for path in found:
        unique[path.resolve()] = path
    return [unique[key] for key in sorted(unique)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode .sog4d/.sog archives into binary little-endian PLY files."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Input .sog4d/.sog files or directories. Defaults to recursively finding .sog4d under the current directory.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="unpacked_ply",
        help="Directory for generated PLY files (default: unpacked_ply).",
    )
    parser.add_argument(
        "--include-sog",
        action="store_true",
        help="Also decode standalone .sog files when scanning directories.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PLY outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    inputs = discover_inputs(args.inputs, output_dir, args.include_sog)

    if not inputs:
        raise SystemExit("No .sog4d files found.")

    print(f"Found {len(inputs)} archive(s). Output directory: {output_dir}")
    total_written = 0

    for input_path in inputs:
        try:
            written = convert_archive(input_path, output_dir, overwrite=args.overwrite)
            total_written += len(written)
        except Exception as exc:
            print(f"ERROR converting {input_path}: {exc}")

    print(f"Done. Generated/skipped {total_written} PLY target(s).")


if __name__ == "__main__":
    main()
