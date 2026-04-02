/**
 * DynamicGSplatResource
 *
 * Extends GSplatResource with extra GPU streams needed by the 4DGS pipeline:
 *   - splatMotion  (RGBA32F): [motion_0, motion_1, motion_2, 0]
 *   - splatTrbf    (RGBA32F): [trbf_center, trbf_scale, 0, 0]
 *   - splatVisibilitySH0..3 (RGBA32F): v_sh_* (names match GLSL loadSplatVisibilitySH* / static loader)
 *
 * All extra textures are owned and destroyed by this class.
 * They are uploaded once at construction time and do not change during playback;
 * dynamic position offsets are applied each frame by the GPU compute pipeline.
 *
 * When the device supports WebGPU compute (device.supportsCompute), this class also
 * creates StorageBuffers for the compute pipeline, eliminating the need for each compute
 * class to re-upload the same data from CPU. The StorageBuffers are co-owned here and
 * shared across CenterUpdateCompute and VisibilitySHCullCompute.
 */

import {
    ADDRESS_CLAMP_TO_EDGE,
    FILTER_NEAREST,
    PIXELFORMAT_RGBA32F,
    GSplatData,
    GSplatResource,
    GraphicsDevice,
    StorageBuffer,
    Texture
} from 'playcanvas';
import { attachGsplatComputeBuffers } from './gsplat-compute-buffers';

/** Number of visibility SH coefficients (deg-3 → 16 = 1+3+5+7) */
const VIS_SH_COEFFS = 16;

class DynamicGSplatResource extends GSplatResource {
    /** Extra dynamic-only textures (not in the base GSplatFormat) */
    motionTexture: Texture;
    trbfTexture: Texture;
    visSHTextures: Texture[] = [];   // 4 RGBA32F textures, 4 coefficients each

    /** Whether visibility SH data was present in gsplatData */
    hasVisibilitySH: boolean;

    // ── Compute StorageBuffers (WebGPU only, null on WebGL) ───────────────────
    // Shared with CenterUpdateCompute and VisibilitySHCullCompute to avoid
    // uploading the same data twice (once for raster textures, once for compute).

    /** f32×4 per splat — (x, y, z, 1) base world positions */
    readonly basePosBuffer: StorageBuffer | null = null;
    /** f32×4 per splat — (motion_0, motion_1, motion_2, 0) */
    readonly motionBuffer: StorageBuffer | null = null;
    /** f32×4 per splat — (trbf_center, trbf_scale, 0, 0) */
    readonly trbfBuffer: StorageBuffer | null = null;
    /** f32×1 per splat — raw logit opacity (pre-sigmoid) */
    readonly opacityBuffer: StorageBuffer | null = null;
    /** 4 × f32×4 per splat — deg-3 visibility SH coefficients (4 coeffs each) */
    readonly visSHStorageBuffers: StorageBuffer[] = [];

    constructor(device: GraphicsDevice, gsplatData: GSplatData) {
        super(device, gsplatData);
        this.hasVisibilitySH = false;
        this._uploadTextureData(device, gsplatData);

        if ((device as any).supportsCompute) {
            this._createComputeBuffers(device, gsplatData);
        }
    }

    private _createTex(device: GraphicsDevice, name: string, width: number, height: number): Texture {
        return new Texture(device, {
            name,
            width,
            height,
            format: PIXELFORMAT_RGBA32F,
            mipmaps: false,
            minFilter: FILTER_NEAREST,
            magFilter: FILTER_NEAREST,
            addressU: ADDRESS_CLAMP_TO_EDGE,
            addressV: ADDRESS_CLAMP_TO_EDGE
        });
    }

    private _uploadTextureData(device: GraphicsDevice, gsplatData: GSplatData) {
        const colorTex = this.streams.getTexture('splatColor');
        if (!colorTex) {
            console.error('[DynamicGSplatResource] Cannot find splatColor texture for dimension reference');
            return;
        }
        const { width, height } = colorTex;
        const numSplats = gsplatData.numSplats;

        // ── Motion texture (motion_0, motion_1, motion_2) ──────────────────────
        const motion0 = gsplatData.getProp('motion_0') as Float32Array | null;
        const motion1 = gsplatData.getProp('motion_1') as Float32Array | null;
        const motion2 = gsplatData.getProp('motion_2') as Float32Array | null;

        this.motionTexture = this._createTex(device, 'splatMotionDyn', width, height);
        if (motion0 && motion1 && motion2) {
            const data = this.motionTexture.lock() as Float32Array;
            for (let i = 0; i < numSplats; i++) {
                data[i * 4 + 0] = motion0[i];
                data[i * 4 + 1] = motion1[i];
                data[i * 4 + 2] = motion2[i];
                data[i * 4 + 3] = 0;
            }
            this.motionTexture.unlock();
        }

        // ── TRBF texture (trbf_center, trbf_scale) ────────────────────────────
        const trbfCenter = gsplatData.getProp('trbf_center') as Float32Array | null;
        const trbfScale  = gsplatData.getProp('trbf_scale')  as Float32Array | null;

        this.trbfTexture = this._createTex(device, 'splatTrbfDyn', width, height);
        if (trbfCenter && trbfScale) {
            const data = this.trbfTexture.lock() as Float32Array;
            for (let i = 0; i < numSplats; i++) {
                data[i * 4 + 0] = trbfCenter[i];
                data[i * 4 + 1] = trbfScale[i];
                data[i * 4 + 2] = 0;
                data[i * 4 + 3] = 0;
            }
            this.trbfTexture.unlock();
        }

        // ── Visibility SH textures (v_sh_0 .. v_sh_15) ───────────────────────
        const shArrays: (Float32Array | null)[] = [];
        for (let k = 0; k < VIS_SH_COEFFS; k++) {
            shArrays.push(gsplatData.getProp(`v_sh_${k}`) as Float32Array | null);
        }

        // Use loose inequality (!=) to catch BOTH null and undefined,
        // since getProp() returns undefined (not null) for missing properties.
        this.hasVisibilitySH = shArrays.every(a => a != null);

        if (this.hasVisibilitySH) {
            for (let t = 0; t < 4; t++) {
                const tex = this._createTex(device, `splatVisibilitySH${t}`, width, height);
                const data = tex.lock() as Float32Array;
                for (let i = 0; i < numSplats; i++) {
                    data[i * 4 + 0] = shArrays[t * 4 + 0]![i];
                    data[i * 4 + 1] = shArrays[t * 4 + 1]![i];
                    data[i * 4 + 2] = shArrays[t * 4 + 2]![i];
                    data[i * 4 + 3] = shArrays[t * 4 + 3]![i];
                }
                tex.unlock();
                this.visSHTextures.push(tex);
            }
        }
    }

    /** Create StorageBuffers for compute shaders (WebGPU only). */
    private _createComputeBuffers(device: GraphicsDevice, gsplatData: GSplatData) {
        attachGsplatComputeBuffers(device, gsplatData, this);
    }

    override destroy() {
        this.motionTexture?.destroy();
        this.trbfTexture?.destroy();
        for (const tex of this.visSHTextures) tex.destroy();
        this.visSHTextures.length = 0;

        this.basePosBuffer?.destroy();
        this.motionBuffer?.destroy();
        this.trbfBuffer?.destroy();
        this.opacityBuffer?.destroy();
        for (const buf of this.visSHStorageBuffers) buf.destroy();
        this.visSHStorageBuffers.length = 0;

        super.destroy();
    }
}

export { DynamicGSplatResource };
