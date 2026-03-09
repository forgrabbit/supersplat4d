# Visibility-Trained 3D Gaussian PLY: Renderer Integration Spec

This document describes how visibility post-training works in this project, how visibility is computed, and how a renderer must read and render visibility-trained Gaussian PLY files correctly. It is intended for implementers who need to modify an existing 3DGS renderer to support these PLY files.

---

## 1. What Visibility Post-Training Does

- **Input**: A pre-trained 3D Gaussian model (e.g. after 30k steps of standard training).
- **Goal**: Add **view-dependent visibility** so that each Gaussian has an extra scalar “visibility” in [0, 1] per view, modulating its opacity. This reduces floaters and improves consistency across views.
- **Mechanism**:
  - **New parameters**: Each Gaussian gets a **visibility spherical harmonics (SH)** vector (1 channel, degree 3 → 16 coefficients). Stored in PLY as `v_sh_0` … `v_sh_15`.
  - **Training**: In a “visibility phase” only these visibility SH coefficients are trained (other Gaussian parameters can be frozen or updated in alternating blocks). The effective opacity used in rendering is `opacity_after = visibility * opacity_before`, and the training loss (L1, SSIM, etc.) is computed on the rendered image using `opacity_after`.
  - **Optional culling**: During training, points with `opacity_after < threshold` (e.g. 0.005) can be excluded from the rasterizer (opacity set to 0) in certain phases. At **inference**, the same threshold is typically applied so that low-visibility points do not contribute.

So: the PLY still stores the **original opacity** (and all other standard 3DGS fields). The **visibility** is encoded in the visibility SH and must be evaluated per view in the renderer to obtain the **effective opacity** used for alpha-blending.

---

## 2. How Visibility Is Computed

Visibility is a **single scalar per Gaussian per view**, computed in two steps.

### 2.1 View direction (unit vector)

- **Definition**: Direction **from camera center to Gaussian center** (view direction toward the point).
- **Formula**:
  - `diff = camera_center - xyz`   (shape `(N, 3)` or broadcast)
  - `dist = max(||diff||, eps)`    (e.g. `eps = 1e-6`)
  - `D_view = -diff / dist`        → unit vector from camera to Gaussian  
  Equivalently: `D_view = (xyz - camera_center) / ||xyz - camera_center||`.
- **Convention**: `D_view` is used as the **direction argument** to the SH evaluation (same convention as in `utils/visibility_utils.py`: “camera -> Gaussian”).

### 2.2 Spherical harmonics evaluation (1 channel, degree 3)

- **Inputs**:
  - `dirs`: `D_view`, shape `(N, 3)`, unit vectors.
  - `sh`: visibility SH coefficients, shape `(N, 1, 16)` (1 channel, 16 coefficients for degree 3).
  - `deg = 3`.
- **Output**: One scalar per point: `vis_raw = eval_sh(deg=3, sh, dirs)`, shape `(N, 1)`.
- **SH formula**: The project uses the same hardcoded SH basis as in `utils/sh_utils.py` (C0, C1, C2, C3 constants). For degree 3, the first 16 coefficients are used. The evaluation is the standard SH sum (e.g. `result = C0*sh[...,0] + ...` up to degree 3). **Important**: This is the **same** `eval_sh()` used for color SH in 3DGS, but here with **1 channel** and **degree 3 only**.
- **Visibility scalar**:
  - `visibility = sigmoid(vis_raw)`
  - Clamped conceptually to [0, 1] (sigmoid already does this).

### 2.3 Effective opacity for rendering

- **Important**: In this project, the PLY file stores **opacity in logit (raw) form**, not as a value in [0, 1]. The training code uses `self._opacity` (logit) when writing PLY and applies `sigmoid` only when reading for rendering (see §7.2).
- `opacity_before` = **sigmoid**(value read from PLY `opacity`).
- `opacity_after = visibility * opacity_before`
- Optional (recommended for consistency with training): if `opacity_after < visibility_cull_threshold` (e.g. 0.005), set opacity to 0 so the point is not rasterized.

The rasterizer should use **opacity_after** (and the same threshold if desired) for alpha-blending; color and geometry (position, scaling, rotation) are unchanged.

---

## 3. PLY Format: What the Renderer Must Read

### 3.1 Standard 3DGS attributes (unchanged)

- `x`, `y`, `z`
- `nx`, `ny`, `nz` (often zeros)
- `f_dc_0`, `f_dc_1`, `f_dc_2`
- `f_rest_0` … `f_rest_44` (for max SH degree 3: 3×((3+1)² − 1) = 45)
- `opacity`
- `scale_0`, `scale_1`, `scale_2`
- `rot_0`, `rot_1`, `rot_2`, `rot_3`

### 3.2 Optional: 3D filter (mip-splatting)

- If present: one scalar per vertex, property name **`filter_3D`**.
- If your renderer supports mip-splatting, use it; otherwise you can ignore this field. Visibility does not depend on it.

### 3.3 Visibility SH (required for visibility-trained models)

- **Property names**: `v_sh_0`, `v_sh_1`, …, `v_sh_15` (16 floats per vertex).
- **Order**: Sort by the numeric suffix (0…15). So index `i` corresponds to the i-th SH coefficient for degree 3.
- **Shape in memory**: After loading, the visibility SH should be stored as `(N, 1, 16)` (N points, 1 channel, 16 coefficients) for use with the same `eval_sh(deg=3, sh, dirs)` API.
- **Detection**: If the PLY contains any property named `v_sh_*`, the model is visibility-trained and the renderer must compute visibility and multiply opacity as above. If no `v_sh_*` exists, render with the standard opacity only.

---

## 4. Rendering Pipeline (Summary)

1. **Load PLY**: Parse all standard attributes plus optional `filter_3D` and optional `v_sh_0`…`v_sh_15`.
2. **Per view**:
   - Compute camera center (world or the same space as Gaussian `xyz`).
   - Compute `D_view = (xyz - camera_center) / ||xyz - camera_center||`, shape `(N, 3)`.
   - If visibility SH is present:
     - `vis_raw = eval_sh(deg=3, visibility_sh, D_view)`  → `(N, 1)`.
     - `visibility = sigmoid(vis_raw)`.
     - `opacity_effective = visibility * opacity_before`.
     - Optionally: set `opacity_effective = 0` where `opacity_effective < visibility_cull_threshold` (e.g. 0.005).
   - Else: `opacity_effective = sigmoid(opacity_from_PLY)` (because PLY stores logit; see §7.2).
3. **Rasterization**: Use the same pipeline as standard 3DGS, but pass **opacity_effective** instead of the raw opacity. Color, position, scaling, rotation, and (if used) 3D filter are unchanged.

---

## 5. Reference Code Locations (This Repo)

- **View direction**: `utils/visibility_utils.py` — `compute_view_direction` / `compute_parallax_direction`.
- **Visibility SH evaluation**: `utils/visibility_utils.py` — `eval_sh_visibility` (calls `utils/sh_utils.eval_sh` with degree 3 and 1 channel).
- **SH basis constants**: `utils/sh_utils.py` — `C0`, `C1`, `C2`, `C3` (and `eval_sh` for deg 0..3).
- **Effective opacity in renderer**: `gaussian_renderer/__init__.py` — `render()` when `use_visibility=True` and `pc.has_visibility_params()`.
- **PLY save/load**: `scene/gaussian_model.py` — `construct_list_of_attributes`, `save_ply`, `load_ply` (visibility SH as `v_sh_0`…`v_sh_15`; optional `filter_3D`).

---

## 6. Checklist for Renderer Modifications

- [ ] PLY loader: detect `v_sh_0`…`v_sh_15`; load and store as (N, 1, 16).
- [ ] Implement or reuse `eval_sh(deg=3, sh, dirs)` with the same SH basis (C0–C3) for 1 channel.
- [ ] Per view: compute `D_view` from camera center and Gaussian positions; compute `visibility = sigmoid(eval_sh(3, visibility_sh, D_view))`.
- [ ] Apply sigmoid to PLY `opacity` to get `opacity_before`. Use `opacity_effective = visibility * opacity_before` when visibility SH is present; otherwise use `opacity_before`.
- [ ] Optional: apply `visibility_cull_threshold` (e.g. 0.005) and set opacity to 0 below that.
- [ ] Pass `opacity_effective` into the rasterizer; leave color and geometry logic unchanged.

Once these are in place, the renderer will correctly read and render visibility-trained Gaussian PLY files from this project.

---

## 7. FAQ / Implementation Q&A

### 7.1 SH basis: must it match the training side exactly?

**Yes.** The viewer must use the **same real SH basis (degree 0–3)** and the **same coefficient order** as the training side, otherwise visibility will be wrong.

- **Source**: This project uses the implementation in `utils/sh_utils.py`, which comes from **PlenOctree** (see file header). The basis is the standard real spherical harmonics used in 3DGS / NeRF-style SH (same convention as original 3DGS color SH).
- **What to implement**: Use the exact constants and polynomial formulas below. Appendix A gives numeric constants and the full `eval_sh(deg=3)` expression so you can port to TypeScript/GLSL without pulling Python code.

**Conclusion**: Implement `eval_sh(deg=3, sh, dirs)` in your viewer using the constants and formula in Appendix A; do not substitute a different SH basis.

---

### 7.2 How is opacity stored in the PLY?

**The PLY file stores opacity as the raw logit (unbounded real), not as a value in [0, 1].**

- **Evidence**: In `scene/gaussian_model.py`, `save_ply()` writes `opacities = self._opacity.detach().cpu().numpy()` — i.e. the internal parameter `_opacity` (logit). The property `get_opacity` used in rendering is `opacity_activation(self._opacity)` = `sigmoid(self._opacity)`.
- **Viewer behavior**:
  - **When reading**: After loading the `opacity` field from the PLY, apply **sigmoid** once to obtain `opacity_before` (in [0, 1]).
  - **When visibility is used**: `opacity_effective = visibility * opacity_before`. No second sigmoid on opacity.
  - **When visibility is not used**: `opacity_effective = sigmoid(opacity_from_PLY)`.

So: **sigmoid is applied at “use for rendering” time**, not at “write PLY” time. The training pipeline applies sigmoid when *reading* for render; the viewer must do the same.

---

### 7.3 Custom PLY attributes (e.g. PlayCanvas / gsplat)

The PLY format is standard: `v_sh_0` … `v_sh_15` are **additional vertex properties** (16 float properties per vertex). There is no special encoding.

- **If the engine automatically exposes all vertex properties**: You can read `v_sh_0` … `v_sh_15` like any other attribute and attach them to your splat data or texture.
- **If the engine does not expose arbitrary properties**: You must either:
  - Extend the loader to parse and expose `v_sh_*`, or
  - After the engine loads the PLY, run a **second pass** (custom parser or same file read) to read only the `v_sh_*` columns and then attach that array to your existing structure (e.g. `GSplatData` or a dedicated texture).

The training repo does not assume any specific viewer engine; it only guarantees that the exported PLY contains these property names and the same order as in §3.3.

---

### 7.4 Where to compute visibility: CPU vs GPU

Both are correct; the choice is a trade-off between implementation effort and performance.

- **CPU**: Each frame, for each splat, compute `D_view`, then `eval_sh(deg=3)`, then `sigmoid`; form `opacity_effective` and upload to the GPU (e.g. overwrite or update an opacity texture/buffer).  
  - **Pros**: Simpler to debug; no need to port SH to shaders.  
  - **Cons**: For large N, per-frame CPU work and upload can be a bottleneck.

- **GPU**: In a vertex (or compute) shader: compute `D_view` from camera position and splat center; implement the same `eval_sh(deg=3)` + `sigmoid`; multiply by the opacity (after sigmoid) and pass or write the effective opacity.  
  - **Pros**: No per-splat CPU work; no extra opacity upload per frame; generally better for real-time.  
  - **Cons**: You must port the SH basis and coefficient order exactly (use Appendix A).

**Recommendation**: Prefer **GPU** (e.g. vertex shader) for performance; use CPU only for prototyping or if the engine does not allow custom vertex logic. In both cases, the math (D_view, eval_sh, sigmoid, opacity_effective) must match this spec.

---

### 7.5 How to handle `filter_3D` if the viewer has no mip-splatting

**You can ignore `filter_3D` entirely.** If your viewer does not plan to support mip-splatting:

- **Load**: If the PLY contains a `filter_3D` property, you may read and discard it (or skip reading it).
- **Rendering**: Do not use it. Visibility and effective opacity do not depend on `filter_3D`. Rendering without mip-splatting will still be correct for visibility-trained models; only antialiasing behavior may differ from the training-side renderer.

---

## Appendix A: SH constants and eval_sh(deg=3) for porting

Use these so that your viewer’s visibility SH evaluation matches `utils/sh_utils.py` exactly. Direction `dirs` is (x, y, z) with **x = dirs[0], y = dirs[1], z = dirs[2]**; coefficients are **sh[0] … sh[15]** (same order as `v_sh_0` … `v_sh_15`).

**Constants:**

```text
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2[0..4] = [ 1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396 ]
C3[0..6] = [ -0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435 ]
```

**Formula for one channel, degree 3 (visibility):**  
`result = C0*sh[0]`  
`+ (-C1*y)*sh[1] + (C1*z)*sh[2] + (-C1*x)*sh[3]`  
`+ C2[0]*x*y*sh[4] + C2[1]*y*z*sh[5] + C2[2]*(2*z*z - x*x - y*y)*sh[6] + C2[3]*x*z*sh[7] + C2[4]*(x*x - y*y)*sh[8]`  
`+ C3[0]*y*(3*x*x - y*y)*sh[9] + C3[1]*x*y*z*sh[10] + C3[2]*y*(4*z*z - x*x - y*y)*sh[11] + C3[3]*z*(2*z*z - 3*x*x - 3*y*y)*sh[12]`  
`+ C3[4]*x*(4*z*z - x*x - y*y)*sh[13] + C3[5]*z*(x*x - y*y)*sh[14] + C3[6]*x*(x*x - 3*y*y)*sh[15]`

Then: **visibility = sigmoid(result)**.

**Pseudocode (GLSL-style):**

```glsl
float eval_sh_visibility_deg3(vec3 d, float sh[16]) {
    float x = d.x, y = d.y, z = d.z;
    float xx = x*x, yy = y*y, zz = z*z;
    float xy = x*y, yz = y*z, xz = x*z;
    float C0 = 0.28209479177387814;
    float C1 = 0.4886025119029199;
    float c2_0 = 1.0925484305920792, c2_1 = -1.0925484305920792, c2_2 = 0.31539156525252005, c2_3 = -1.0925484305920792, c2_4 = 0.5462742152960396;
    float c3_0 = -0.5900435899266435, c3_1 = 2.890611442640554, c3_2 = -0.4570457994644658, c3_3 = 0.3731763325901154;
    float c3_4 = -0.4570457994644658, c3_5 = 1.445305721320277, c3_6 = -0.5900435899266435;
    float r = C0*sh[0]
        - C1*y*sh[1] + C1*z*sh[2] - C1*x*sh[3]
        + c2_0*xy*sh[4] + c2_1*yz*sh[5] + c2_2*(2.0*zz - xx - yy)*sh[6] + c2_3*xz*sh[7] + c2_4*(xx - yy)*sh[8]
        + c3_0*y*(3.0*xx - yy)*sh[9] + c3_1*xy*z*sh[10] + c3_2*y*(4.0*zz - xx - yy)*sh[11] + c3_3*z*(2.0*zz - 3.0*xx - 3.0*yy)*sh[12]
        + c3_4*x*(4.0*zz - xx - yy)*sh[13] + c3_5*z*(xx - yy)*sh[14] + c3_6*x*(xx - 3.0*yy)*sh[15];
    return 1.0 / (1.0 + exp(-r));  // sigmoid
}
```

Use `d = normalize(splat_center - camera_center)` (or your engine’s equivalent) so the direction matches §2.1.
 the direction matches §2.1.
