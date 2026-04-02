/**
 * VisibilitySHCullCompute
 *
 * Orchestrates two GPU compute passes:
 *   1. Visibility cull — evaluates effective opacity per splat, applies segment mask,
 *                        and writes activeMask
 *   2. Compact          — streams active splat indices into activeIndicesBuffer
 *
 * Both passes are implemented as exported WGSL strings in compute-visibility-cull.ts.
 * The activeCount (number of active splats after culling) is written to activeCountBuffer;
 * DynamicGSplatPipeline reads it on the CPU (same frame, after submit) before radix sort.
 *
 * Static data buffers (opacity, visSH) are supplied by DynamicGSplatResource to
 * avoid re-uploading data that is already on the GPU.
 */

import {
    BindGroupFormat,
    BindStorageBufferFormat,
    BindUniformBufferFormat,
    Compute,
    GraphicsDevice,
    SHADERLANGUAGE_WGSL,
    SHADERSTAGE_COMPUTE,
    Shader,
    StorageBuffer,
    UNIFORMTYPE_FLOAT,
    UNIFORMTYPE_UINT,
    UNIFORMTYPE_VEC3,
    UniformBufferFormat,
    UniformFormat
} from 'playcanvas';
import { DynamicWorkBuffer } from './dynamic-work-buffer';
import {
    computeVisibilityCullSource,
    computeCompactSource
} from '../shaders/compute-visibility-cull';

const WORKGROUP = 256;

class VisibilitySHCullCompute {
    private device: GraphicsDevice;
    private numSplats: number;
    private hasVisSH: boolean;

    // Externally-owned read-only buffers (provided by DynamicGSplatResource)
    private opacityBuffer: StorageBuffer;
    private visSHBuffers: StorageBuffer[];   // always length 4

    // Cull pass
    private cullCompute: Compute;
    private cullUniformFormat: UniformBufferFormat;
    private cullBindFormat: BindGroupFormat;

    // Compact pass
    private compactCompute: Compute;
    private compactUniformFormat: UniformBufferFormat;
    private compactBindFormat: BindGroupFormat;

    /** Reusable Float32Array for camera position uniform */
    private camPosData = new Float32Array(3);

    constructor(
        device: GraphicsDevice,
        opacityBuffer: StorageBuffer,
        visSHBuffers: StorageBuffer[],
        hasVisSH: boolean,
        workBuf: DynamicWorkBuffer
    ) {
        this.device = device;
        this.numSplats = workBuf.numSplats;
        this.hasVisSH = hasVisSH;
        this.opacityBuffer = opacityBuffer;
        this.visSHBuffers  = visSHBuffers;

        // ── Cull pass ─────────────────────────────────────────────────────────
        this.cullUniformFormat = new UniformBufferFormat(device, [
            new UniformFormat('cameraPos',     UNIFORMTYPE_VEC3),
            new UniformFormat('numSplats',     UNIFORMTYPE_UINT),
            new UniformFormat('currentTime',   UNIFORMTYPE_FLOAT),
            new UniformFormat('cullThreshold', UNIFORMTYPE_FLOAT),
            new UniformFormat('hasVisSH',      UNIFORMTYPE_UINT),
            new UniformFormat('skipTemporal',  UNIFORMTYPE_UINT)
        ]);

        // Binding layout must match @binding indices in compute-visibility-cull.ts:
        // 0:centers 1:opacity 2:trbf 3:visSH0 4:visSH1 5:visSH2 6:visSH3 7:activeMask
        // 8:uniforms  9:segmentMask
        this.cullBindFormat = new BindGroupFormat(device, [
            new BindStorageBufferFormat('centers',     SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('opacity',     SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('trbf',        SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('visSH0',      SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('visSH1',      SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('visSH2',      SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('visSH3',      SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('activeMask',  SHADERSTAGE_COMPUTE, false),
            new BindUniformBufferFormat('uniforms',    SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('segmentMask', SHADERSTAGE_COMPUTE, true)
        ]);

        const cullShader = new Shader(device, {
            name: 'VisibilityCullCompute',
            shaderLanguage: SHADERLANGUAGE_WGSL,
            cshader: computeVisibilityCullSource,
            computeEntryPoint: 'visibilityCull',
            computeBindGroupFormat: this.cullBindFormat,
            computeUniformBufferFormats: { uniforms: this.cullUniformFormat }
        } as any);
        this.cullCompute = new Compute(device, cullShader, 'VisibilityCullCompute');

        // ── Compact pass ──────────────────────────────────────────────────────
        this.compactUniformFormat = new UniformBufferFormat(device, [
            new UniformFormat('numSplats', UNIFORMTYPE_UINT)
        ]);

        this.compactBindFormat = new BindGroupFormat(device, [
            new BindStorageBufferFormat('activeMask',    SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('activeIndices', SHADERSTAGE_COMPUTE, false),
            new BindStorageBufferFormat('activeCount',   SHADERSTAGE_COMPUTE, false),
            new BindUniformBufferFormat('uniforms',      SHADERSTAGE_COMPUTE)
        ]);

        const compactShader = new Shader(device, {
            name: 'CompactActiveCompute',
            shaderLanguage: SHADERLANGUAGE_WGSL,
            cshader: computeCompactSource,
            computeEntryPoint: 'compactActive',
            computeBindGroupFormat: this.compactBindFormat,
            computeUniformBufferFormats: { uniforms: this.compactUniformFormat }
        } as any);
        this.compactCompute = new Compute(device, compactShader, 'CompactActiveCompute');
    }

    /**
     * Dispatch the cull + compact passes.
     * @param workBuf      Provides centerBuffer (input) and activeMask/activeIndices/activeCount (output)
     * @param trbfBuf      TRBF storage buffer (center, scale, _, _) × numSplats
     * @param currentTime  Absolute animation time
     * @param cameraPos    Camera position in model space
     * @param cullThreshold Opacity threshold below which a splat is culled (default 0.01)
     */
    dispatch(
        workBuf: DynamicWorkBuffer,
        trbfBuf: StorageBuffer,
        currentTime: number,
        cameraPos: { x: number; y: number; z: number },
        cullThreshold = 0.01,
        skipTemporal = false
    ) {
        const groups = Math.ceil(this.numSplats / WORKGROUP);

        // ── Cull pass ─────────────────────────────────────────────────────────
        const cull = this.cullCompute;
        cull.setParameter('centers',     workBuf.centerBuffer);
        cull.setParameter('opacity',     this.opacityBuffer);
        cull.setParameter('trbf',        trbfBuf);
        cull.setParameter('visSH0',      this.visSHBuffers[0]);
        cull.setParameter('visSH1',      this.visSHBuffers[1]);
        cull.setParameter('visSH2',      this.visSHBuffers[2]);
        cull.setParameter('visSH3',      this.visSHBuffers[3]);
        cull.setParameter('activeMask',  workBuf.activeMaskBuffer);
        cull.setParameter('segmentMask', workBuf.segmentMaskBuffer);

        this.camPosData[0] = cameraPos.x;
        this.camPosData[1] = cameraPos.y;
        this.camPosData[2] = cameraPos.z;
        cull.setParameter('cameraPos',     this.camPosData);
        cull.setParameter('numSplats',     this.numSplats);
        cull.setParameter('currentTime',   currentTime);
        cull.setParameter('cullThreshold', cullThreshold);
        cull.setParameter('hasVisSH',      this.hasVisSH ? 1 : 0);
        cull.setParameter('skipTemporal',  skipTemporal ? 1 : 0);

        cull.setupDispatch(groups, 1, 1);

        // ── Compact pass ──────────────────────────────────────────────────────
        // Reset active count to 0 before dispatch (CPU write, occurs before GPU dispatch)
        const zero = new Uint32Array([0]);
        workBuf.activeCountBuffer.write(0, zero, 0, 1);

        const compact = this.compactCompute;
        compact.setParameter('activeMask',    workBuf.activeMaskBuffer);
        compact.setParameter('activeIndices', workBuf.activeIndicesBuffer);
        compact.setParameter('activeCount',   workBuf.activeCountBuffer);
        compact.setParameter('numSplats',     this.numSplats);

        compact.setupDispatch(groups, 1, 1);

        // Dispatch both passes in a single command buffer; within one compute pass
        // writes from cull (activeMask) are visible to compact.
        this.device.computeDispatch([cull, compact], 'VisibilityCullAndCompact');
    }

    destroy() {
        // Data buffers are owned by DynamicGSplatResource — do NOT destroy them here.
        this.cullCompute.shader?.destroy();
        this.compactCompute.shader?.destroy();
        this.cullBindFormat.destroy();
        this.compactBindFormat.destroy();
    }
}

export { VisibilitySHCullCompute };
