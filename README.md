# SuperSplat 4D Editor


[![License](https://img.shields.io/github/license/playcanvas/supersplat)](https://github.com/playcanvas/supersplat/blob/main/LICENSE)

SuperSplat 4D is an enhanced version of the SuperSplat Editor, extending support to **Dynamic Gaussian Splatting (4D)**. It is a free and open-source tool for inspecting, editing, optimizing, and publishing both static and dynamic Gaussian Splats. Built on web technologies, it runs directly in your browser.

A live version of this tool is available at: https://supersplat4d.netlify.app/

## Key Features

- **Dynamic Gaussian Support**: Load, play, and visualize dynamic splats with Time-Radial Basis Functions (TRBF).
- **Format Support**:
  - **.sog4d**: Highly compressed dynamic splat format using k-means clustering and WebP.
  - **.ply**: Direct loading of dynamic PLY files containing `trbf_center`, `trbf_scale`, and `motion` attributes.
- **Dynamic Editing**:
  - Select and delete瑕疵 (artifacts) or unwanted splats across the entire animation.
  - Filter splats based on dynamic opacity.
- **Export Capabilities**:
  - Export edited dynamic scenes back to `.ply` or `.sog4d`.
  - Selective time-range export (Scheme A: keeping original `trbf_center`).

## Compression Tool

The repository includes a Python script to compress raw dynamic PLY files into the efficient `.sog4d` format:

```bash
python ply_to_sog4d.py --ply input.ply -o output.sog4d
```

The script can read `cfg_args` directly from the PLY header if it contains a line like:
`comment cfg_args: duration=2.0 start=0.0 fps=30.0 sh_degree=0`

## Local Development

To initialize a local development environment, ensure you have [Node.js](https://nodejs.org/) 18 or later installed.

1. Clone and switch branch:
   ```sh
   git clone https://github.com/playcanvas/supersplat.git
   cd supersplat
   git checkout splat4d
   ```

2. Install dependencies:
   ```sh
   npm install
   ```

3. Run development server:
   ```sh
   npm run develop
   ```

3. Navigate to `http://localhost:3000`

## Localizing the Editor

Supported languages are located in `static/locales`. To test a specific language:
`http://localhost:3000/?lng=<locale>` (e.g., `zh`, `en`).

## Contributors

SuperSplat is made possible by the open source community and the original PlayCanvas team.

<a href="https://github.com/playcanvas/supersplat/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=playcanvas/supersplat" />
</a>
