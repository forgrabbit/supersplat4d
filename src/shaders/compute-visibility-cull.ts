/**
 * WGSL compute shader — VisibilityCull
 *
 * Per splat: evaluate effective_opacity = sigmoid(base_opacity) * sigmoid(SH) * trbf_kernel
 * and write 1 to activeMask if opacity > threshold AND the splat is in the active segment, else 0.
 *
 * Bindings:
 *   0: centers       (storage, read)       — f32×4 per splat updated world positions from CenterUpdateCompute
 *   1: opacity       (storage, read)       — f32×1 per splat (raw logit opacity, pre-sigmoid)
 *   2: trbf          (storage, read)       — f32×4 per splat (center, scale, _, _)
 *   3: visSH0..6     (storage, read)       — f32×4 per splat each, 16 visibility SH coefficients
 *   7: activeMask    (storage, read_write) — u32 per splat, output mask
 *   8: uniforms      (uniform buffer)      — currentTime, numSplats, cameraPos, cullThreshold, hasVisSH
 *   9: segmentMask   (storage, read)       — u32 per splat, 1=in active segment, 0=excluded
 *
 * Notes:
 *   - When hasVisSH == 0, only TRBF * opacity is used (no SH evaluation).
 *   - Deg-3 SH (16 coefficients) is evaluated in the view direction from camera to splat center.
 *   - Sigmoid(x) = 1 / (1 + exp(-x)).
 *   - A splat is active only when BOTH the segmentMask is 1 AND effOpacity > cullThreshold.
 */

export const computeVisibilityCullSource = /* wgsl */`

struct Uniforms {
    cameraPos     : vec3f,
    numSplats     : u32,
    currentTime   : f32,
    cullThreshold : f32,
    hasVisSH      : u32,
    skipTemporal  : u32,
};

@group(0) @binding(0) var<storage, read>       centers     : array<vec4f>;
@group(0) @binding(1) var<storage, read>       opacity     : array<f32>;
@group(0) @binding(2) var<storage, read>       trbf        : array<vec4f>;   // (center, scale, _, _)
@group(0) @binding(3) var<storage, read>       visSH0      : array<vec4f>;   // v_sh_0..3
@group(0) @binding(4) var<storage, read>       visSH1      : array<vec4f>;   // v_sh_4..7
@group(0) @binding(5) var<storage, read>       visSH2      : array<vec4f>;   // v_sh_8..11
@group(0) @binding(6) var<storage, read>       visSH3      : array<vec4f>;   // v_sh_12..15
@group(0) @binding(7) var<storage, read_write> activeMask  : array<u32>;
@group(0) @binding(8) var<uniform>             uniforms    : Uniforms;
@group(0) @binding(9) var<storage, read>       segmentMask : array<u32>;

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

// Evaluate degree-3 spherical harmonics in direction d.
// Coefficients: c0 (DC), c1-3 (l=1), c4-8 (l=2), c9-15 (l=3)
fn evalSHDeg3(d: vec3f,
              c0123: vec4f,   // sh0..3
              c4567: vec4f,   // sh4..7
              c89ab: vec4f,   // sh8..11
              cCDEF: vec4f    // sh12..15
             ) -> f32 {
    let x = d.x; let y = d.y; let z = d.z;

    // l=0
    var result = 0.28209479177387814 * c0123.x;

    // l=1
    result += -0.4886025119029199 * y * c0123.y;
    result +=  0.4886025119029199 * z * c0123.z;
    result += -0.4886025119029199 * x * c0123.w;

    // l=2
    let xy = x * y;
    let yz = y * z;
    let zz = z * z;
    let xz = x * z;
    let xx_yy = x * x - y * y;
    result += 1.0925484305920792 * xy              * c4567.x;
    result += -1.0925484305920792 * yz             * c4567.y;
    result += 0.31539156525252005 * (2.0*zz - x*x - y*y) * c4567.z;
    result += -1.0925484305920792 * xz             * c4567.w;
    result += 0.5462742152960396  * xx_yy          * c89ab.x;

    // l=3
    result += -0.5900435899266435 * y * (3.0*x*x - y*y)  * c89ab.y;
    result += 2.890611442640554   * xy * z                * c89ab.z;
    result += -0.4570457994644658 * y * (4.0*zz - x*x - y*y) * c89ab.w;
    result += 0.3731763325901154  * z * (2.0*zz - 3.0*x*x - 3.0*y*y) * cCDEF.x;
    result += -0.4570457994644658 * x * (4.0*zz - x*x - y*y) * cCDEF.y;
    result += 1.445305721320277   * z * xx_yy              * cCDEF.z;
    result += -0.5900435899266435 * x * (x*x - 3.0*y*y)   * cCDEF.w;

    return result;
}

@compute @workgroup_size(256, 1, 1)
fn visibilityCull(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i >= uniforms.numSplats) { return; }

    // Splats outside the active time segment are immediately excluded.
    if (segmentMask[i] == 0u) {
        activeMask[i] = 0u;
        return;
    }

    // Temporal opacity kernel: exp(-((t - center) / scale)^2); static splats use skipTemporal=1
    var temporalWeight = 1.0;
    if (uniforms.skipTemporal == 0u) {
        let trbfCenter = trbf[i].x;
        let trbfScale  = max(trbf[i].y, 1e-6);
        let dt = (uniforms.currentTime - trbfCenter) / trbfScale;
        temporalWeight = exp(-dt * dt);
    }

    // Base opacity
    var effOpacity = sigmoid(opacity[i]) * temporalWeight;

    // Visibility SH (when available)
    if (uniforms.hasVisSH != 0u) {
        let worldPos = centers[i].xyz;
        let dir = normalize(worldPos - uniforms.cameraPos);
        let shVal = evalSHDeg3(dir, visSH0[i], visSH1[i], visSH2[i], visSH3[i]);
        let vis = sigmoid(shVal);
        effOpacity *= vis;
    }

    activeMask[i] = select(0u, 1u, effOpacity > uniforms.cullThreshold);
}
`;

// ── Compact pass ─────────────────────────────────────────────────────────────
//
// Turns activeMask[n] → activeIndices[k], activeCount using atomicAdd.
// Non-deterministic ordering is fine since downstream radix-sort re-orders.

export const computeCompactSource = /* wgsl */`

struct Uniforms {
    numSplats : u32,
};

@group(0) @binding(0) var<storage, read>       activeMask    : array<u32>;
@group(0) @binding(1) var<storage, read_write> activeIndices : array<u32>;
@group(0) @binding(2) var<storage, read_write> activeCount   : atomic<u32>;
@group(0) @binding(3) var<uniform>             uniforms      : Uniforms;

@compute @workgroup_size(256, 1, 1)
fn compactActive(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i >= uniforms.numSplats) { return; }

    if (activeMask[i] == 1u) {
        let slot = atomicAdd(&activeCount, 1u);
        activeIndices[slot] = i;
    }
}
`;
