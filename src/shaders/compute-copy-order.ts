/**
 * WGSL compute shader — CopyOrder
 *
 * Resolves the two-level indirection from radix sort:
 *   sortedPositions[k] = index into activeIndices array
 *   activeIndices[sortedPositions[k]] = actual splat global index
 *
 * Writes the final render order into orderTexture (texture_storage_2d<r32uint, write>)
 * so the legacy GSplatInstance shader can read it.
 *
 * Bindings:
 *   0: sortedPositions (storage, read)          — u32 per active splat, output of ComputeRadixSort
 *   1: activeIndices   (storage, read)          — u32 per active splat, compact splat indices
 *   2: orderTexture    (storage texture, write)  — r32uint 2D texture for GSplatInstance
 *   3: uniforms        (uniform buffer)
 */

export const computeCopyOrderSource = /* wgsl */`

struct Uniforms {
    activeCount : u32,
    textureSize : u32,
};

@group(0) @binding(0) var<storage, read> sortedPositions : array<u32>;
@group(0) @binding(1) var<storage, read> activeIndices   : array<u32>;
@group(0) @binding(2) var orderTexture                   : texture_storage_2d<r32uint, write>;
@group(0) @binding(3) var<uniform> uniforms              : Uniforms;

@compute @workgroup_size(256, 1, 1)
fn copyOrder(@builtin(global_invocation_id) id: vec3u) {
    let k = id.x;
    if (k >= uniforms.activeCount) { return; }

    // Resolve two-level indirection
    let pos       = sortedPositions[k];
    let splatIdx  = activeIndices[pos];

    // Write to 2D texture: linearIndex → (u, v)
    let ts  = uniforms.textureSize;
    let uv  = vec2u(k % ts, k / ts);
    textureStore(orderTexture, uv, vec4u(splatIdx, 0u, 0u, 0u));
}
`;
