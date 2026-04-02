/**
 * WGSL compute shader — CenterUpdate
 *
 * Per splat: world_pos = base_pos + motion * (currentTime - trbf_center)
 * Reads from StorageBuffers and writes to centerBuffer (used by subsequent passes).
 *
 * Bindings:
 *   0: basePos     (storage, read)      — f32×4 per splat (x, y, z, _) base world positions
 *   1: motion      (storage, read)      — f32×4 per splat (motion_0, motion_1, motion_2, _)
 *   2: trbf        (storage, read)      — f32×4 per splat (center, scale, _, _)
 *   3: centerOut   (storage, read_write)— f32×4 per splat, output updated world centers
 *   4: uniforms    (uniform buffer)     — currentTime, numSplats
 */

export const computeCenterUpdateSource = /* wgsl */`

struct Uniforms {
    currentTime : f32,
    numSplats   : u32,
};

@group(0) @binding(0) var<storage, read>       basePos   : array<vec4f>;
@group(0) @binding(1) var<storage, read>       motion    : array<vec4f>;
@group(0) @binding(2) var<storage, read>       trbf      : array<vec4f>;
@group(0) @binding(3) var<storage, read_write> centerOut : array<vec4f>;
@group(0) @binding(4) var<uniform>             uniforms  : Uniforms;

@compute @workgroup_size(256, 1, 1)
fn updateCenters(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i >= uniforms.numSplats) { return; }

    let base  = basePos[i].xyz;
    let mot   = motion[i].xyz;
    let dt    = uniforms.currentTime - trbf[i].x;   // x = trbf_center

    centerOut[i] = vec4f(base + mot * dt, 1.0);
}
`;
