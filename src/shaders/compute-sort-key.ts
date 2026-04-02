/**
 * WGSL compute shader — SortKey (reads world centers from StorageBuffer)
 *
 * Variant of the engine's computeGsplatSortKeySource but reads from
 * centerBuffer (array<vec4f>) instead of a texture_2d<u32>, so it integrates
 * seamlessly with DynamicWorkBuffer without requiring a buffer-to-texture copy.
 *
 * Bindings:
 *   0: centerBuf   (storage, read)       — f32×4 per splat world centers (output of CenterUpdateCompute)
 *   1: indexBuf    (storage, read)       — u32 per splat active indices (output of compaction)
 *   2: sortKeys    (storage, read_write) — u32 output sort keys
 *   3: uniforms    (uniform buffer)
 */

export const computeSortKeyBufferSource = /* wgsl */`

struct Uniforms {
    cameraPosition  : vec3f,
    elementCount    : u32,
    cameraDirection : vec3f,
    numBits         : u32,
    minDist         : f32,
    invRange        : f32,
    numBins         : u32,
    _pad            : u32,
};

struct BinWeight {
    base    : f32,
    divider : f32,
};

@group(0) @binding(0) var<storage, read>       centerBuf  : array<vec4f>;
@group(0) @binding(1) var<storage, read>       indexBuf   : array<u32>;
@group(0) @binding(2) var<storage, read_write> sortKeys   : array<u32>;
@group(0) @binding(3) var<uniform>             uniforms   : Uniforms;
@group(0) @binding(4) var<storage, read>       binWeights : array<BinWeight>;

@compute @workgroup_size(256, 1, 1)
fn computeSortKey(@builtin(global_invocation_id) id: vec3u) {
    let k = id.x;
    if (k >= uniforms.elementCount) { return; }

    let splatIdx = indexBuf[k];
    let worldCenter = centerBuf[splatIdx].xyz;

    // Linear: dot(toSplat, cameraDirection)
    let toSplat = worldCenter - uniforms.cameraPosition;
    let dist = dot(toSplat, uniforms.cameraDirection) - uniforms.minDist;

    let numBins = uniforms.numBins;
    let d = dist * uniforms.invRange * f32(numBins);
    let binFloat = clamp(d, 0.0, f32(numBins) - 0.001);
    let bin = u32(binFloat);
    let binFrac = binFloat - f32(bin);

    // Invert key so farther objects get SMALLER keys.
    // ComputeRadixSort sorts ascending (small key first = drawn first),
    // giving back-to-front order required for correct alpha blending.
    let maxKey = (1u << uniforms.numBits) - 1u;
    sortKeys[k] = maxKey - u32(binWeights[bin].base + binWeights[bin].divider * binFrac);
}
`;
