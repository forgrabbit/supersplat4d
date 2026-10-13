# SuperSplat 4D

**Dynamic Gaussian Splatting editing, compression, and browser rendering.**

基于 [PlayCanvas SuperSplat](https://github.com/playcanvas/supersplat) 扩展的 4D Gaussian Splatting 网页工具，支持动态场景的加载、播放、编辑和导出，并提供 `.sog4d` 压缩工具。

[Online editor](https://supersplat4d.netlify.app/) · [Dynamic scene viewer](https://forgrabbit.github.io/splat4d-viewer-pages/) · [4D source branch](https://github.com/forgrabbit/supersplat4d/tree/splat4d) · [MIT license](LICENSE)

> **Start with the `splat4d` branch.** The 4D implementation, Python conversion tools, and sample models live there. The repository's default `main` branch is not the 4D development branch.

## What this extension adds

SuperSplat provides the original editor and static Gaussian tooling. This fork extends that foundation with:

| Area | 4D functionality |
| :--- | :--- |
| Dynamic representation | Time-Radial Basis Function (TRBF) playback with time-dependent motion and opacity. |
| Loading | Dynamic PLY files with `trbf_center`, `trbf_scale`, and motion attributes; compressed `.sog4d` scenes. |
| Editing | Inspect, select, and delete dynamic splats; filter by dynamic opacity. |
| Export | Export dynamic scenes to PLY or `.sog4d`, including selected time ranges. |
| Compression | Python conversion using attribute quantization, k-means clustering, and WebP encoding. |

## Quick start

Use Node.js 22 or later and npm. Clone **this fork and its 4D branch**:

```sh
git clone --branch splat4d --single-branch https://github.com/forgrabbit/supersplat4d.git
cd supersplat4d
npm ci
npm run develop
```

Open `http://localhost:3000` after the build finishes. The [`public/`](https://github.com/forgrabbit/supersplat4d/tree/splat4d/public) directory contains sample `.sog4d` models for trying dynamic playback.

For a production build:

```sh
npm run build
npm run serve
```

The generated web application is written to `dist/`.

## Convert a dynamic PLY to `.sog4d`

From the `splat4d` branch, install the conversion dependencies:

```sh
python -m pip install numpy plyfile pillow scikit-learn
```

For temporal segment generation and optional opacity filtering, also install PyTorch appropriate for your environment. The converter can use CUDA when available.

```sh
python ply_to_sog4d.py --ply input.ply -o output.sog4d
```

Dynamic conversion requires the scene's actual timing metadata. The converter reads it from a PLY header comment such as:

```text
comment cfg_args: duration=2.0 start=0.0 fps=30.0 sh_degree=0
```

If the PLY does not include this comment, provide a separate `cfg_args` file:

```sh
python ply_to_sog4d.py --cfg_args path/to/cfg_args --ply input.ply -o output.sog4d
```

Use timing values that match your training/export pipeline. See `python ply_to_sog4d.py --help` for compression and filtering options. The Python conversion tools operate separately from the browser editor.

## Implementation guide

All paths below refer to the **`splat4d` branch**:

| Entry point | Purpose |
| :--- | :--- |
| [`src/loaders/dynamic-ply.ts`](https://github.com/forgrabbit/supersplat4d/blob/splat4d/src/loaders/dynamic-ply.ts) | Dynamic PLY loading. |
| [`src/loaders/sog4d.ts`](https://github.com/forgrabbit/supersplat4d/blob/splat4d/src/loaders/sog4d.ts) | Compressed scene loading. |
| [`src/serialize-dynamic.ts`](https://github.com/forgrabbit/supersplat4d/blob/splat4d/src/serialize-dynamic.ts) | Dynamic scene serialization. |
| [`src/ui/dynamic-export-dialog.ts`](https://github.com/forgrabbit/supersplat4d/blob/splat4d/src/ui/dynamic-export-dialog.ts) | Export controls. |
| [`ply_to_sog4d.py`](https://github.com/forgrabbit/supersplat4d/blob/splat4d/ply_to_sog4d.py) | Python compression and conversion. |

## Compatibility and scope

- Dynamic PLY input must follow the TRBF/motion schema supported by this implementation; arbitrary 4DGS formats may need conversion.
- Playback speed and memory use depend on model size, device, and browser. This repository does not claim a uniform frame rate across devices.
- This is an editing and rendering tool; train or reconstruct a scene with a compatible pipeline before loading it here.

## Localization

Translations live in `static/locales` on the `splat4d` branch. For example, use `http://localhost:3000/?lng=zh-CN` for Simplified Chinese.

## Acknowledgments and license

The original SuperSplat editor is developed by the PlayCanvas team and the open-source community. This repository builds on their work and adds support for dynamic Gaussian scenes. See the [upstream project](https://github.com/playcanvas/supersplat) and the [MIT license](LICENSE) for attribution and license terms.
