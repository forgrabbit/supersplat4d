/**
 * Shared GPU StorageBuffers for GSplat compute (cull + sort).
 * Used by DynamicGSplatResource and by static GSplatResource when WebGPU compute is enabled.
 */

import { BUFFERUSAGE_COPY_DST, GSplatData, GraphicsDevice, StorageBuffer } from 'playcanvas';

const VIS_SH_COEFFS = 16;

/** True when all deg-3 visibility SH coefficients exist on gsplatData. */
export function gsplatDataHasVisibilitySH(gsplatData: GSplatData): boolean {
    for (let k = 0; k < VIS_SH_COEFFS; k++) {
        const a = gsplatData.getProp(`v_sh_${k}`) as Float32Array | null | undefined;
        if (a == null) return false;
    }
    return true;
}

/**
 * Creates compute buffers and attaches them to `target` (typically a GSplatResource).
 * Idempotent: no-op if `target.basePosBuffer` already exists.
 * Static path: motion = 0, trbf = (0, 1, 0, 0); temporal cull uses skipTemporal uniform.
 */
export function attachGsplatComputeBuffers(
    device: GraphicsDevice,
    gsplatData: GSplatData,
    target: any
): void {
    if (!(device as any).supportsCompute) return;
    if (target.basePosBuffer) return;

    const n = gsplatData.numSplats;
    const uploadUsage = BUFFERUSAGE_COPY_DST;

    const x = gsplatData.getProp('x') as Float32Array | null;
    const y = gsplatData.getProp('y') as Float32Array | null;
    const z = gsplatData.getProp('z') as Float32Array | null;
    const motion0 = gsplatData.getProp('motion_0') as Float32Array | null;
    const motion1 = gsplatData.getProp('motion_1') as Float32Array | null;
    const motion2 = gsplatData.getProp('motion_2') as Float32Array | null;
    const trbfCenter = gsplatData.getProp('trbf_center') as Float32Array | null;
    const trbfScale = gsplatData.getProp('trbf_scale') as Float32Array | null;
    const rawOpacity = gsplatData.getProp('opacity') as Float32Array | null;

    const hasVisFromData = gsplatDataHasVisibilitySH(gsplatData);
    target.hasVisibilitySH = !!(target as { hasVisibilitySH?: boolean }).hasVisibilitySH || hasVisFromData;

    target.basePosBuffer = new StorageBuffer(device, n * 4 * 4, uploadUsage);
    {
        const data = new Float32Array(n * 4);
        if (x && y && z) {
            for (let i = 0; i < n; i++) {
                data[i * 4 + 0] = x[i];
                data[i * 4 + 1] = y[i];
                data[i * 4 + 2] = z[i];
                data[i * 4 + 3] = 1;
            }
        }
        // Always upload — WebGPU storage is otherwise undefined; zeros are safe fallback.
        target.basePosBuffer.write(0, data, 0, data.length);
    }

    target.motionBuffer = new StorageBuffer(device, n * 4 * 4, uploadUsage);
    {
        const data = new Float32Array(n * 4);
        if (motion0 && motion1 && motion2) {
            for (let i = 0; i < n; i++) {
                data[i * 4 + 0] = motion0[i];
                data[i * 4 + 1] = motion1[i];
                data[i * 4 + 2] = motion2[i];
                data[i * 4 + 3] = 0;
            }
        }
        target.motionBuffer.write(0, data, 0, data.length);
    }

    target.trbfBuffer = new StorageBuffer(device, n * 4 * 4, uploadUsage);
    if (trbfCenter && trbfScale) {
        const data = new Float32Array(n * 4);
        for (let i = 0; i < n; i++) {
            data[i * 4 + 0] = trbfCenter[i];
            data[i * 4 + 1] = trbfScale[i];
            data[i * 4 + 2] = 0;
            data[i * 4 + 3] = 0;
        }
        target.trbfBuffer.write(0, data, 0, data.length);
    } else {
        const data = new Float32Array(n * 4);
        for (let i = 0; i < n; i++) {
            data[i * 4 + 0] = 0;
            data[i * 4 + 1] = 1;
            data[i * 4 + 2] = 0;
            data[i * 4 + 3] = 0;
        }
        target.trbfBuffer.write(0, data, 0, data.length);
    }

    target.opacityBuffer = new StorageBuffer(device, n * 4, uploadUsage);
    if (rawOpacity) {
        target.opacityBuffer.write(0, rawOpacity, 0, rawOpacity.length);
    } else {
        const fallback = new Float32Array(n);
        fallback.fill(0);
        target.opacityBuffer.write(0, fallback, 0, fallback.length);
    }

    if (!Array.isArray(target.visSHStorageBuffers)) {
        target.visSHStorageBuffers = [];
    }
    const visArr: StorageBuffer[] = target.visSHStorageBuffers;
    for (const b of visArr) {
        b.destroy();
    }
    visArr.length = 0;
    for (let t = 0; t < 4; t++) {
        visArr.push(new StorageBuffer(device, n * 4 * 4, uploadUsage));
    }
    if (hasVisFromData) {
        for (let t = 0; t < 4; t++) {
            const data = new Float32Array(n * 4);
            for (let c = 0; c < 4; c++) {
                const coeff = gsplatData.getProp(`v_sh_${t * 4 + c}`) as Float32Array | null;
                if (coeff) {
                    for (let i = 0; i < n; i++) {
                        data[i * 4 + c] = coeff[i];
                    }
                }
            }
            visArr[t].write(0, data, 0, data.length);
        }
    }
}

/** Destroy buffers attached by attachGsplatComputeBuffers when Splat owns them (static path). */
export function destroyAttachedGsplatComputeBuffers(target: any): void {
    target.basePosBuffer?.destroy();
    target.motionBuffer?.destroy();
    target.trbfBuffer?.destroy();
    target.opacityBuffer?.destroy();
    for (const buf of target.visSHStorageBuffers ?? []) buf.destroy();
    target.basePosBuffer = null;
    target.motionBuffer = null;
    target.trbfBuffer = null;
    target.opacityBuffer = null;
    target.visSHStorageBuffers = [];
}
