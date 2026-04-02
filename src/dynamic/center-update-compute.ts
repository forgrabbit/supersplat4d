/**
 * CenterUpdateCompute
 *
 * TypeScript wrapper around the compute-center-update WGSL shader.
 * On every call to dispatch() it updates the per-splat world-space centers
 * in DynamicWorkBuffer.centerBuffer for the given absolute time.
 *
 * Static data (base positions, motion, TRBF) is supplied via StorageBuffers created
 * and owned by DynamicGSplatResource, so the same GPU upload is shared with any
 * other compute class that needs the same data (no double upload).
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
    UniformBufferFormat,
    UniformFormat
} from 'playcanvas';
import { DynamicWorkBuffer } from './dynamic-work-buffer';
import { computeCenterUpdateSource } from '../shaders/compute-center-update';

const WORKGROUP = 256;

class CenterUpdateCompute {
    private device: GraphicsDevice;
    private numSplats: number;

    // Externally-owned read-only buffers (provided by DynamicGSplatResource)
    private basePosBuffer: StorageBuffer;
    private motionBuffer: StorageBuffer;
    /** TRBF buffer — also exposed so DynamicGSplatPipeline can pass it to VisibilitySHCullCompute. */
    readonly trbfBuffer: StorageBuffer;

    private compute: Compute;
    private uniformBufFormat: UniformBufferFormat;
    private bindGroupFormat: BindGroupFormat;

    constructor(
        device: GraphicsDevice,
        basePosBuffer: StorageBuffer,
        motionBuffer: StorageBuffer,
        trbfBuffer: StorageBuffer,
        workBuf: DynamicWorkBuffer
    ) {
        this.device = device;
        this.numSplats = workBuf.numSplats;
        this.basePosBuffer = basePosBuffer;
        this.motionBuffer  = motionBuffer;
        this.trbfBuffer    = trbfBuffer;

        this.uniformBufFormat = new UniformBufferFormat(device, [
            new UniformFormat('currentTime', UNIFORMTYPE_FLOAT),
            new UniformFormat('numSplats',   UNIFORMTYPE_UINT)
        ]);

        this.bindGroupFormat = new BindGroupFormat(device, [
            new BindStorageBufferFormat('basePos',   SHADERSTAGE_COMPUTE, true),   // read-only
            new BindStorageBufferFormat('motion',    SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('trbf',      SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('centerOut', SHADERSTAGE_COMPUTE, false),  // read_write
            new BindUniformBufferFormat('uniforms',  SHADERSTAGE_COMPUTE)
        ]);

        const shader = new Shader(device, {
            name: 'CenterUpdateCompute',
            shaderLanguage: SHADERLANGUAGE_WGSL,
            cshader: computeCenterUpdateSource,
            computeEntryPoint: 'updateCenters',
            computeBindGroupFormat: this.bindGroupFormat,
            computeUniformBufferFormats: { uniforms: this.uniformBufFormat }
        } as any);

        this.compute = new Compute(device, shader, 'CenterUpdateCompute');
    }

    /**
     * Dispatches the center-update compute shader.
     * @param currentTime  Absolute animation time in seconds.
     * @param workBuf      DynamicWorkBuffer — output written to workBuf.centerBuffer.
     */
    dispatch(currentTime: number, workBuf: DynamicWorkBuffer) {
        const c = this.compute;
        c.setParameter('basePos',    this.basePosBuffer);
        c.setParameter('motion',     this.motionBuffer);
        c.setParameter('trbf',       this.trbfBuffer);
        c.setParameter('centerOut',  workBuf.centerBuffer);
        c.setParameter('currentTime', currentTime);
        c.setParameter('numSplats',   this.numSplats);

        const groups = Math.ceil(this.numSplats / WORKGROUP);
        c.setupDispatch(groups, 1, 1);
        this.device.computeDispatch([c], 'CenterUpdateCompute');
    }

    destroy() {
        // Data buffers are owned by DynamicGSplatResource — do NOT destroy them here.
        this.compute.shader?.destroy();
        this.bindGroupFormat.destroy();
    }
}

export { CenterUpdateCompute };
