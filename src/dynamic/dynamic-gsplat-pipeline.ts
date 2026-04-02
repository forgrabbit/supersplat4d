/**
 * GsplatComputePipeline
 *
 * GPU path for splat cull + sort + orderTexture (WebGPU compute):
 *
 * Dynamic mode:
 *   1. CenterUpdateCompute — p(t) = p0 + motion * (t - trbf_center)
 *   2. VisibilitySHCullCompute (+ temporal TRBF)
 *   3–5. SortKey, radix sort, CopyOrder
 *
 * Static mode: skips per-frame center update (bootstrap once from base positions);
 * cull uses skipTemporal (opacity × visibility SH only).
 *
 * After update(), render uses GPU-sorted order with culled splats omitted.
 */

import {
    ADDRESS_CLAMP_TO_EDGE,
    BindGroupFormat,
    BindStorageBufferFormat,
    BindStorageTextureFormat,
    BindUniformBufferFormat,
    BUFFERUSAGE_COPY_DST,
    BUFFERUSAGE_STORAGE,
    ComputeRadixSort,
    Entity,
    FILTER_NEAREST,
    GraphicsDevice,
    Mat4,
    PIXELFORMAT_R32U,
    SHADERLANGUAGE_WGSL,
    SHADERSTAGE_COMPUTE,
    Shader,
    Compute,
    StorageBuffer,
    Texture,
    UNIFORMTYPE_FLOAT,
    UNIFORMTYPE_UINT,
    UNIFORMTYPE_VEC3,
    UniformBufferFormat,
    UniformFormat,
    Vec3
} from 'playcanvas';
import { DynamicWorkBuffer } from './dynamic-work-buffer';
import { CenterUpdateCompute } from './center-update-compute';
import { VisibilitySHCullCompute } from './visibility-sh-cull-compute';
import { computeSortKeyBufferSource } from '../shaders/compute-sort-key';
import { computeCopyOrderSource } from '../shaders/compute-copy-order';

const WORKGROUP = 256;
const NUM_BITS  = 16;                         // radix sort key bits
const MAX_KEY   = (1 << NUM_BITS) - 1;        // 65535
const NUM_BINS  = 64;                         // camera-relative precision bins (uniform spacing)

/** Generate NUM_BINS × 2 float32 values (base, divider) for uniform bin spacing. */
function buildUniformBinWeights(numBins: number, numBits: number): Float32Array {
    const maxKey = (1 << numBits) - 1;
    const slotSize = maxKey / numBins;
    const out = new Float32Array(numBins * 2);
    for (let b = 0; b < numBins; b++) {
        out[b * 2 + 0] = b * slotSize;          // base
        out[b * 2 + 1] = slotSize;              // divider
    }
    return out;
}

/** Resource contract for cull/sort (DynamicGSplatResource or GSplatResource + compute buffers). */
export interface GsplatComputePipelineResource {
    streams: { textureDimensions: { x: number; y: number } };
    /** Present on GSplatResource; used only for scene AABB in _initSceneBounds. */
    gsplatData?: unknown;
    basePosBuffer: StorageBuffer;
    motionBuffer: StorageBuffer;
    trbfBuffer: StorageBuffer;
    opacityBuffer: StorageBuffer;
    visSHStorageBuffers: StorageBuffer[];
    hasVisibilitySH: boolean;
}

export type GsplatComputePipelineMode = 'dynamic' | 'static';

class GsplatComputePipeline {
    private device: GraphicsDevice;
    private numSplats: number;

    readonly workBuf: DynamicWorkBuffer;

    private centerUpdate: CenterUpdateCompute;
    private readonly _mode: GsplatComputePipelineMode;
    private visCull: VisibilitySHCullCompute;
    private radixSort: ComputeRadixSort;

    // Sort key compute
    private sortKeyCompute: Compute;
    private sortKeyBindFormat: BindGroupFormat;
    private sortKeyUniformFormat: UniformBufferFormat;
    private binWeightsBuffer: StorageBuffer;

    // Copy-order pass
    private copyOrderCompute: Compute;
    private copyOrderBindFormat: BindGroupFormat;
    private copyOrderUniformFormat: UniformBufferFormat;

    /**
     * TRBF StorageBuffer — owned by DynamicGSplatResource, referenced here for convenience
     * to pass to VisibilitySHCullCompute.dispatch() without going through resource each time.
     */
    private trbfBuffer: StorageBuffer;

    /**
     * Storage-writeable orderTexture that the copy-order compute shader writes to.
     * This REPLACES the GSplatInstance's default orderTexture (which is not storage-writeable).
     * The caller (Splat.onUpdate) must redirect the material's 'splatOrder' parameter to this.
     */
    readonly orderTexture: Texture;

    /** Inverse model matrix — used to transform camera from world to model space for sort/cull. */
    private _invModelMat = new Mat4();

    /** Active splat count after the most recent completed `update()` (same-frame readback). */
    lastActiveCount: number;

    /** Reusable Uint32Array(1) for GPU→CPU readback of activeCount. */
    private _readbackData = new Uint32Array(1);

    /** Reusable buffer for updateSegmentMask — avoids per-call allocation. */
    private _segmentMaskData: Uint32Array;

    // Scene-space AABB (model space) computed once from gsplatData, used for depth range.
    private _sceneMin = new Vec3();
    private _sceneMax = new Vec3();

    static isSupported(device: GraphicsDevice): boolean {
        return !!(device as any).supportsCompute;
    }

    constructor(
        device: GraphicsDevice,
        resource: GsplatComputePipelineResource,
        numSplats: number,
        options?: { mode?: GsplatComputePipelineMode }
    ) {
        this.device    = device;
        this.numSplats = numSplats;
        this._mode     = options?.mode ?? 'dynamic';
        this._segmentMaskData = new Uint32Array(numSplats);

        this.workBuf = new DynamicWorkBuffer(device, numSplats);

        // ── Pre-fill activeIndicesBuffer with [0, 1, 2, ...] ─────────────────
        // Ensures the first-frame sort-key pass reads valid (though unculled) indices
        // even before the compact pass has had a chance to run.
        {
            const initIndices = new Uint32Array(numSplats);
            for (let i = 0; i < numSplats; i++) initIndices[i] = i;
            this.workBuf.activeIndicesBuffer.write(0, initIndices, 0, initIndices.length);
        }

        // ── Start with zero active count ──────────────────────────────────────
        // Using numSplats as the initial value would cause ALL splats (from all
        // time frames) to be drawn on the first few frames before the async GPU
        // readback resolves — causing over-brightness artefacts.  Starting at 0
        // keeps the mesh hidden (instancingCount = 0, visible = false in
        // Splat.onUpdate) until the first real readback arrives.
        this.lastActiveCount = 0;

        // ── Storage-writeable orderTexture ────────────────────────────────────
        const dims = resource.streams.textureDimensions;
        this.orderTexture = new Texture(device, {
            name: this._mode === 'static' ? 'staticSplatOrder' : 'dynSplatOrder',
            width:  dims.x,
            height: dims.y,
            format: PIXELFORMAT_R32U,
            mipmaps: false,
            minFilter: FILTER_NEAREST,
            magFilter: FILTER_NEAREST,
            addressU: ADDRESS_CLAMP_TO_EDGE,
            addressV: ADDRESS_CLAMP_TO_EDGE,
            storage: true    // PlayCanvas 2.16: maps to GPUTextureUsage.STORAGE_BINDING
        } as any);

        // ── Compute passes ─────────────────────────────────────────────────────
        // Data buffers are owned by DynamicGSplatResource and shared here to avoid
        // duplicate GPU uploads of the same static data.
        this.centerUpdate = new CenterUpdateCompute(
            device,
            resource.basePosBuffer!,
            resource.motionBuffer!,
            resource.trbfBuffer!,
            this.workBuf
        );
        this.trbfBuffer = resource.trbfBuffer!;

        this.visCull = new VisibilitySHCullCompute(
            device,
            resource.opacityBuffer!,
            resource.visSHStorageBuffers.length === 4
                ? resource.visSHStorageBuffers
                : Array(4).fill(resource.opacityBuffer!),   // fallback (no visSH)
            resource.hasVisibilitySH,
            this.workBuf
        );
        this.radixSort = new ComputeRadixSort(device);

        // ── Sort key compute ──────────────────────────────────────────────────
        this._buildSortKeyPass(device);

        // ── Copy-order pass ───────────────────────────────────────────────────
        this._buildCopyOrderPass(device);

        // ── Pre-compute scene AABB for depth-range estimation ─────────────────
        this._initSceneBounds(resource);

        // ── Initialize segment mask to all-active ─────────────────────────────
        // The GPU buffer is zero-initialized by WebGPU, which would cull ALL splats
        // on every frame until a segment file loads. Starting with all-1s lets the
        // temporal opacity (TRBF) filter handle visibility on early frames so the
        // scene is visible before the first segment ACT file arrives.
        this.resetSegmentMask();

        // Static: fill centerBuffer once (motion=0, t=0 → centers = base).
        if (this._mode === 'static') {
            this.centerUpdate.dispatch(0, this.workBuf);
        }
    }

    private _buildSortKeyPass(device: GraphicsDevice) {
        this.sortKeyUniformFormat = new UniformBufferFormat(device, [
            new UniformFormat('cameraPosition',  UNIFORMTYPE_VEC3),
            new UniformFormat('elementCount',    UNIFORMTYPE_UINT),
            new UniformFormat('cameraDirection', UNIFORMTYPE_VEC3),
            new UniformFormat('numBits',         UNIFORMTYPE_UINT),
            new UniformFormat('minDist',         UNIFORMTYPE_FLOAT),
            new UniformFormat('invRange',        UNIFORMTYPE_FLOAT),
            new UniformFormat('numBins',         UNIFORMTYPE_UINT),
            new UniformFormat('_pad',            UNIFORMTYPE_UINT)
        ]);

        this.sortKeyBindFormat = new BindGroupFormat(device, [
            new BindStorageBufferFormat('centerBuf',  SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('indexBuf',   SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('sortKeys',   SHADERSTAGE_COMPUTE, false),
            new BindUniformBufferFormat('uniforms',   SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('binWeights', SHADERSTAGE_COMPUTE, true)
        ]);

        const shader = new Shader(device, {
            name: 'DynSortKeyCompute',
            shaderLanguage: SHADERLANGUAGE_WGSL,
            cshader: computeSortKeyBufferSource,
            computeEntryPoint: 'computeSortKey',
            computeBindGroupFormat: this.sortKeyBindFormat,
            computeUniformBufferFormats: { uniforms: this.sortKeyUniformFormat }
        } as any);
        this.sortKeyCompute = new Compute(device, shader, 'DynSortKeyCompute');

        const weights = buildUniformBinWeights(NUM_BINS, NUM_BITS);
        this.binWeightsBuffer = new StorageBuffer(
            device,
            weights.byteLength,
            BUFFERUSAGE_COPY_DST
        );
        this.binWeightsBuffer.write(0, weights, 0, weights.length);
    }

    private _buildCopyOrderPass(device: GraphicsDevice) {
        this.copyOrderUniformFormat = new UniformBufferFormat(device, [
            new UniformFormat('activeCount', UNIFORMTYPE_UINT),
            new UniformFormat('textureSize', UNIFORMTYPE_UINT)
        ]);

        this.copyOrderBindFormat = new BindGroupFormat(device, [
            new BindStorageBufferFormat('sortedPositions', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('activeIndices',   SHADERSTAGE_COMPUTE, true),
            new BindStorageTextureFormat('orderTexture', PIXELFORMAT_R32U),
            new BindUniformBufferFormat('uniforms',        SHADERSTAGE_COMPUTE)
        ]);

        const shader = new Shader(device, {
            name: 'DynCopyOrderCompute',
            shaderLanguage: SHADERLANGUAGE_WGSL,
            cshader: computeCopyOrderSource,
            computeEntryPoint: 'copyOrder',
            computeBindGroupFormat: this.copyOrderBindFormat,
            computeUniformBufferFormats: { uniforms: this.copyOrderUniformFormat }
        } as any);
        this.copyOrderCompute = new Compute(device, shader, 'DynCopyOrderCompute');
    }

    /**
     * Compute the scene's model-space AABB once from the raw position arrays.
     * Used every frame to derive a tight depth range for the sort key shader,
     * dramatically improving sort precision for small/medium scenes.
     */
    private _initSceneBounds(resource: GsplatComputePipelineResource) {
        const gsplatData = (resource as any).gsplatData ?? (resource as any)._gsplatData;
        const x = gsplatData?.getProp('x') as Float32Array | null;
        const y = gsplatData?.getProp('y') as Float32Array | null;
        const z = gsplatData?.getProp('z') as Float32Array | null;
        if (!x || !y || !z || x.length === 0) {
            this._sceneMin.set(-50, -50, -50);
            this._sceneMax.set( 50,  50,  50);
            return;
        }
        let minX = x[0], maxX = x[0];
        let minY = y[0], maxY = y[0];
        let minZ = z[0], maxZ = z[0];
        for (let i = 1; i < x.length; i++) {
            if (x[i] < minX) minX = x[i]; else if (x[i] > maxX) maxX = x[i];
            if (y[i] < minY) minY = y[i]; else if (y[i] > maxY) maxY = y[i];
            if (z[i] < minZ) minZ = z[i]; else if (z[i] > maxZ) maxZ = z[i];
        }
        this._sceneMin.set(minX, minY, minZ);
        this._sceneMax.set(maxX, maxY, maxZ);
    }

    /**
     * Project the 8 corners of the model-space scene AABB onto the current camera
     * forward direction and return a padded [minDist, maxDist] range.
     * This is fast (8 dot products per frame, no allocation) and avoids the
     * hardcoded 0..1000 range that wastes 16-bit precision on small scenes.
     */
    private _computeDepthRange(): { minDist: number; maxDist: number } {
        const min = this._sceneMin;
        const max = this._sceneMax;
        const cx = this._camPosVec.x, cy = this._camPosVec.y, cz = this._camPosVec.z;
        const dx = this._camDirVec.x, dy = this._camDirVec.y, dz = this._camDirVec.z;

        let dMin = Infinity, dMax = -Infinity;
        for (let i = 0; i < 8; i++) {
            const px = (i & 1) ? max.x : min.x;
            const py = (i & 2) ? max.y : min.y;
            const pz = (i & 4) ? max.z : min.z;
            const depth = (px - cx) * dx + (py - cy) * dy + (pz - cz) * dz;
            if (depth < dMin) dMin = depth;
            if (depth > dMax) dMax = depth;
        }

        const pad = Math.max((dMax - dMin) * 0.1, 0.5);
        return {
            minDist: Math.max(0, dMin - pad),
            maxDist: dMax + pad
        };
    }

    private _camPos    = new Float32Array(3);
    private _camDir    = new Float32Array(3);
    private _camPosVec = new Vec3();
    private _camDirVec = new Vec3();

    /**
     * Run the full GPU pipeline for a given frame.
     *
     * @param currentTime    Absolute animation time in seconds.
     * @param cameraEntity   Camera entity for position/direction.
     * @param splatEntity    Entity that owns the GSplatComponent (for model→world transform).
     * @param cullThreshold  Opacity threshold (default 0.01).
     */
    async update(currentTime: number, cameraEntity: Entity, splatEntity: Entity, cullThreshold = 0.01): Promise<void> {
        const device = this.device;

        // ── Camera in model space ──────────────────────────────────────────────
        // The GPU center data is in model space, so we must transform the camera
        // into model space before computing depths/directions.
        const camWorld = cameraEntity.getWorldTransform();
        const modelMat = splatEntity.getWorldTransform();
        this._invModelMat.invert(modelMat);

        // World-space camera position → model space
        camWorld.getTranslation(this._camPosVec);
        this._invModelMat.transformPoint(this._camPosVec, this._camPosVec);
        this._camPos[0] = this._camPosVec.x;
        this._camPos[1] = this._camPosVec.y;
        this._camPos[2] = this._camPosVec.z;

        // Camera forward direction (toward scene = -Z local axis) → model space.
        // Mat4.getZ() returns the camera's +Z axis (the backward/away-from-scene direction),
        // so we negate it to obtain the look direction, then normalize to remove scale effects.
        camWorld.getZ(this._camDirVec);
        // getZ() returns +Z (backward/away-from-scene); negate to get the forward look direction.
        this._camDirVec.x = -this._camDirVec.x;
        this._camDirVec.y = -this._camDirVec.y;
        this._camDirVec.z = -this._camDirVec.z;
        this._invModelMat.transformVector(this._camDirVec, this._camDirVec);
        this._camDirVec.normalize();
        this._camDir[0] = this._camDirVec.x;
        this._camDir[1] = this._camDirVec.y;
        this._camDir[2] = this._camDirVec.z;

        // ── 1. Center update (dynamic only) ───────────────────────────────────
        if (this._mode === 'dynamic') {
            this.centerUpdate.dispatch(currentTime, this.workBuf);
        }

        // ── 2. Visibility cull + compact ──────────────────────────────────────
        this.visCull.dispatch(
            this.workBuf, this.trbfBuffer, currentTime,
            { x: this._camPos[0], y: this._camPos[1], z: this._camPos[2] },
            cullThreshold,
            this._mode === 'static'
        );

        // ── 2b. Same-frame active count (must match this frame's compact pass) ─
        // Previously we used lastActiveCount from an async readback scheduled at the
        // end of update — one frame stale.  When TRBF/segment culling changes the
        // active set every frame, staleCount > trueCount makes the sort pass read
        // past the compacted prefix into stale activeIndices[] → wrong order / chaos.
        // read(..., true) submits the encoder then maps — see WebgpuGraphicsDevice.readBuffer.
        let activeCount: number;
        try {
            const readView = await this.workBuf.activeCountBuffer.read(0, 4, this._readbackData, true);
            const rd = (readView instanceof Uint32Array)
                ? readView
                : new Uint32Array(readView.buffer, readView.byteOffset, 1);
            activeCount = rd[0];
        } catch {
            return;
        }
        if (activeCount > this.numSplats) {
            activeCount = this.numSplats;
        }
        this.lastActiveCount = activeCount;

        if (activeCount === 0) {
            return;
        }

        // ── 3. Sort keys ──────────────────────────────────────────────────────
        // Compute a tight depth range from the scene AABB projected onto the camera
        // forward direction, giving full 16-bit precision across the visible depth.
        const { minDist, maxDist } = this._computeDepthRange();
        const invRange = 1.0 / Math.max(maxDist - minDist, 1e-6);

        const sk = this.sortKeyCompute;
        sk.setParameter('centerBuf',       this.workBuf.centerBuffer);
        sk.setParameter('indexBuf',        this.workBuf.activeIndicesBuffer);
        sk.setParameter('sortKeys',        this.workBuf.sortKeysBuffer);
        sk.setParameter('binWeights',      this.binWeightsBuffer);
        sk.setParameter('cameraPosition',  this._camPos);
        sk.setParameter('cameraDirection', this._camDir);
        sk.setParameter('elementCount',    activeCount);
        sk.setParameter('numBits',         NUM_BITS);
        sk.setParameter('minDist',         minDist);
        sk.setParameter('invRange',        invRange);
        sk.setParameter('numBins',         NUM_BINS);
        sk.setParameter('_pad',            0);

        const skGroups = Math.ceil(activeCount / WORKGROUP);
        sk.setupDispatch(skGroups, 1, 1);
        device.computeDispatch([sk], 'DynSortKeyCompute');

        // ── 4. Radix sort ─────────────────────────────────────────────────────
        const sortedPositions = this.radixSort.sort(
            this.workBuf.sortKeysBuffer,
            activeCount,
            NUM_BITS
        );

        // ── 5. Copy sorted order → this.orderTexture ─────────────────────────
        const texSize = this.orderTexture.width;

        const co = this.copyOrderCompute;
        co.setParameter('sortedPositions', sortedPositions);
        co.setParameter('activeIndices',   this.workBuf.activeIndicesBuffer);
        co.setParameter('orderTexture',    this.orderTexture);
        co.setParameter('activeCount',     activeCount);
        co.setParameter('textureSize',     texSize);

        const coGroups = Math.ceil(activeCount / WORKGROUP);
        co.setupDispatch(coGroups, 1, 1);
        device.computeDispatch([co], 'DynCopyOrderCompute');
    }

    /**
     * Update the per-segment activation mask.
     * Call this whenever the active time segment changes (e.g. after loading a .act file).
     *
     * @param activeIndices  Global splat indices that belong to the new active segment.
     *                       All other splats will be excluded from rendering.
     */
    updateSegmentMask(activeIndices: Uint32Array): void {
        const mask = this._segmentMaskData;
        mask.fill(0);
        for (let i = 0; i < activeIndices.length; i++) {
            const idx = activeIndices[i];
            if (idx < this.numSplats) mask[idx] = 1;
        }
        this.workBuf.segmentMaskBuffer.write(0, mask, 0, this.numSplats);
    }

    /** Reset the segment mask to all-active (all splats participate). */
    resetSegmentMask(): void {
        this._segmentMaskData.fill(1);
        this.workBuf.segmentMaskBuffer.write(0, this._segmentMaskData, 0, this.numSplats);
    }

    destroy() {
        this.centerUpdate.destroy();
        this.visCull.destroy();
        this.radixSort.destroy();
        this.workBuf.destroy();
        this.binWeightsBuffer.destroy();
        this.sortKeyCompute.shader?.destroy();
        this.sortKeyBindFormat.destroy();
        this.copyOrderCompute.shader?.destroy();
        this.copyOrderBindFormat.destroy();
        this.orderTexture.destroy();
    }
}

export { GsplatComputePipeline, GsplatComputePipeline as DynamicGSplatPipeline };
