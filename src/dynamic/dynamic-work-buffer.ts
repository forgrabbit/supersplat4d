/**
 * DynamicWorkBuffer
 *
 * Owns the GPU StorageBuffers used by the 4DGS compute pipeline:
 *
 *   centerBuffer         f32×4 per splat  — world-space centers written by CenterUpdateCompute
 *   activeMaskBuffer     u32  per splat   — 0/1 mask written by VisibilitySHCullCompute
 *   activeIndicesBuffer  u32  per splat   — compact active splat indices (written by compaction)
 *   sortKeysBuffer       u32  per splat   — sort keys written by SortKeyCompute
 *   activeCountBuffer    u32  ×1          — number of active splats after culling
 *   segmentMaskBuffer    u32  per splat   — per-segment activation mask (1=in segment, 0=not)
 *
 * All buffers are sized for the maximum number of splats (numSplats); the actual active
 * count is stored in activeCountBuffer so downstream passes can use indirect dispatch.
 */

import {
    BUFFERUSAGE_COPY_DST,
    BUFFERUSAGE_COPY_SRC,
    GraphicsDevice,
    StorageBuffer
} from 'playcanvas';

// StorageBuffer constructor auto-adds BUFFERUSAGE_STORAGE — we only need the transfer flags.
// COPY_DST: allows CPU write via StorageBuffer.write() / queue.writeBuffer()
// COPY_SRC: allows CPU readback via StorageBuffer.read() (staging copy pattern)
// NEVER use BUFFERUSAGE_READ (MAP_READ) or BUFFERUSAGE_WRITE (MAP_WRITE) on storage buffers —
// WebGPU forbids combining MapRead/MapWrite with Storage usage.
function makeBuffer(device: GraphicsDevice, byteSize: number): StorageBuffer {
    return new StorageBuffer(device, byteSize, BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC);
}

class DynamicWorkBuffer {
    readonly numSplats: number;

    /** f32×4 per splat — updated world-space centers (x, y, z, 1) */
    readonly centerBuffer: StorageBuffer;

    /** u32 per splat — 1 if active (not culled), 0 otherwise */
    readonly activeMaskBuffer: StorageBuffer;

    /** u32 per splat — compact list of active splat global indices */
    readonly activeIndicesBuffer: StorageBuffer;

    /** u32 per splat — sort keys (distance bins) for radix sort */
    readonly sortKeysBuffer: StorageBuffer;

    /** u32 ×1 — total number of active splats (written by compaction compute) */
    readonly activeCountBuffer: StorageBuffer;

    /**
     * u32 per splat — per-segment activation mask.
     * Initialised to all-1 (all splats active). Updated by DynamicGSplatPipeline.updateSegmentMask()
     * whenever the active time segment changes. The visibility-cull pass ANDs this mask with
     * the TRBF+opacity result so that only splats belonging to the current segment are rendered.
     */
    readonly segmentMaskBuffer: StorageBuffer;

    constructor(device: GraphicsDevice, numSplats: number) {
        this.numSplats = numSplats;
        const n = numSplats;

        this.centerBuffer        = makeBuffer(device, n * 4 * 4);   // vec4f per splat
        this.activeMaskBuffer    = makeBuffer(device, n * 4);        // u32 per splat
        this.activeIndicesBuffer = makeBuffer(device, n * 4);        // u32 per splat
        this.sortKeysBuffer      = makeBuffer(device, n * 4);        // u32 per splat
        this.activeCountBuffer   = makeBuffer(device, 4);            // single u32

        this.segmentMaskBuffer   = makeBuffer(device, n * 4);        // u32 per splat
        // Default: all splats belong to the active segment.
        const initMask = new Uint32Array(n).fill(1);
        this.segmentMaskBuffer.write(0, initMask, 0, n);
    }

    destroy() {
        this.centerBuffer.destroy();
        this.activeMaskBuffer.destroy();
        this.activeIndicesBuffer.destroy();
        this.sortKeysBuffer.destroy();
        this.activeCountBuffer.destroy();
        this.segmentMaskBuffer.destroy();
    }
}

export { DynamicWorkBuffer };
