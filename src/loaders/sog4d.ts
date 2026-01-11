/**
 * SOG4D Loader - Load compressed dynamic gaussian splatting files
 *
 * File format: ZIP archive containing:
 *   - meta.json: Metadata, codebooks, and dynamic parameters
 *   - means_l.webp, means_u.webp: Position (16-bit quantized)
 *   - quats.webp: Rotation quaternions
 *   - scales.webp: Scale (k-means labels)
 *   - sh0.webp: Color (k-means labels) + opacity
 *   - motion_l.webp, motion_u.webp: Motion vectors (16-bit quantized)
 *   - trbf_l.webp, trbf_u.webp: TRBF parameters (16-bit quantized)
 *   - segments/seg_XXX.act: Time segment active indices
 */

import { Asset, AssetRegistry, GSplatData, GSplatResource } from 'playcanvas';
// JSZip is loaded globally via script tag in index.html
declare const JSZip: any;

import { AssetSource, createReadSource } from './asset-source';
import type { DynManifest } from './dyn';

// =============================================================================
// Types
// =============================================================================

interface Sog4dMeta {
    version: number;
    type: 'sog4d';
    generator: string;
    count: number;
    width: number;
    height: number;
    sh_degree: number;

    // Dynamic parameters
    start: number;
    duration: number;
    fps: number;

    // Static gaussian attributes
    means: {
        mins: number[];
        maxs: number[];
        files: string[];
    };
    quats: {
        files: string[];
    };
    scales: {
        codebook: number[];
        files: string[];
    };
    sh0: {
        codebook: number[];
        files: string[];
    };

    // Dynamic gaussian attributes
    motion: {
        mins: number[];
        maxs: number[];
        files: string[];
    };
    trbf: {
        encoding: 'kmeans' | 'quantize16';
        // For kmeans encoding
        center_codebook?: number[];
        scale_codebook?: number[];  // Values are log(trbf_scale)
        // For quantize16 encoding
        center_min?: number;
        center_max?: number;
        scale_min?: number;  // log(trbf_scale)
        scale_max?: number;
        files: string[];
    };

    // Segments
    segments: Array<{
        t0: number;
        t1: number;
        url: string;
        count: number;
    }>;
}

// =============================================================================
// Decode utilities
// =============================================================================

/**
 * Decode WebP image to RGBA Uint8Array
 * Uses options to prevent color space conversion and premultiplied alpha
 */
const decodeWebP = async (data: ArrayBuffer): Promise<{ rgba: Uint8Array, width: number, height: number }> => {
    const blob = new Blob([data], { type: 'image/webp' });

    // Disable color space conversion and premultiplied alpha to preserve raw data
    const bitmap = await createImageBitmap(blob, {
        premultiplyAlpha: 'none',
        colorSpaceConversion: 'none'
    });

    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    // Use willReadFrequently for better performance when reading pixel data
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    ctx.drawImage(bitmap, 0, 0);

    const imageData = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
    return {
        rgba: new Uint8Array(imageData.data.buffer),
        width: bitmap.width,
        height: bitmap.height
    };
};

/**
 * Inverse log transform: sign(y) * (exp(|y|) - 1)
 */
const invLogTransform = (v: number): number => {
    const a = Math.abs(v);
    const e = Math.exp(a) - 1;
    return v < 0 ? -e : e;
};

/**
 * Dequantize 16-bit value to float
 */
const dequantize16bit = (lo: number, hi: number, min: number, max: number): number => {
    const val16 = lo | (hi << 8);
    const scale = (max - min) || 1;
    return min + (val16 / 65535) * scale;
};

/**
 * Unpack quaternion from smallest-component compression
 */
const unpackQuat = (px: number, py: number, pz: number, tag: number): [number, number, number, number] => {
    const maxComp = tag - 252;
    const sqrt2 = Math.sqrt(2);

    const a = (px / 255) * 2 - 1;
    const b = (py / 255) * 2 - 1;
    const c = (pz / 255) * 2 - 1;

    const comps = [0, 0, 0, 0];
    const idx = [
        [1, 2, 3],
        [0, 2, 3],
        [0, 1, 3],
        [0, 1, 2]
    ][maxComp];

    comps[idx[0]] = a / sqrt2;
    comps[idx[1]] = b / sqrt2;
    comps[idx[2]] = c / sqrt2;

    // Reconstruct max component
    const t = 1 - (comps[0] * comps[0] + comps[1] * comps[1] + comps[2] * comps[2] + comps[3] * comps[3]);
    comps[maxComp] = Math.sqrt(Math.max(0, t));

    return comps as [number, number, number, number];
};

/**
 * Inverse sigmoid (logit)
 */
const sigmoidInv = (y: number): number => {
    const e = Math.min(1 - 1e-6, Math.max(1e-6, y));
    return Math.log(e / (1 - e));
};

// =============================================================================
// Main decoder
// =============================================================================

let assetId = 0;

/**
 * Parse SOG4D ZIP file and create GSplatData
 */
const parseSog4d = async (zipData: ArrayBuffer): Promise<{ gsplatData: GSplatData, meta: Sog4dMeta, zipEntries: Map<string, ArrayBuffer> }> => {
    console.log('📦 Parsing SOG4D file...');

    // Load ZIP using global JSZip loaded via script tag
    const zip = await JSZip.loadAsync(zipData);

    // Helper to load file from ZIP
    const loadFile = async (name: string): Promise<ArrayBuffer> => {
        const file = zip.file(name);
        if (!file) {
            throw new Error(`Missing file in SOG4D: ${name}`);
        }
        return await file.async('arraybuffer');
    };

    // Parse meta.json
    const metaJson = await loadFile('meta.json');
    const meta: Sog4dMeta = JSON.parse(new TextDecoder().decode(metaJson));

    if (meta.type !== 'sog4d') {
        throw new Error(`Expected type 'sog4d', got '${meta.type}'`);
    }

    console.log(`📊 SOG4D: ${meta.count} splats, ${meta.width}x${meta.height} texture`);
    console.log(`📊 Duration: ${meta.duration}s @ ${meta.fps} fps`);

    const count = meta.count;

    // Load and decode WebP images (common files)
    const [
        meansL, meansU,
        quatsData,
        scalesData,
        sh0Data,
        motionL, motionU
    ] = await Promise.all([
        loadFile('means_l.webp').then(decodeWebP),
        loadFile('means_u.webp').then(decodeWebP),
        loadFile('quats.webp').then(decodeWebP),
        loadFile('scales.webp').then(decodeWebP),
        loadFile('sh0.webp').then(decodeWebP),
        loadFile('motion_l.webp').then(decodeWebP),
        loadFile('motion_u.webp').then(decodeWebP)
    ]);

    // Load TRBF data (different files depending on encoding mode)
    const trbfIsKmeans = meta.trbf.encoding === 'kmeans';
    let trbfData: { rgba: Uint8Array, width: number, height: number } | null = null;
    let trbfL: { rgba: Uint8Array, width: number, height: number } | null = null;
    let trbfU: { rgba: Uint8Array, width: number, height: number } | null = null;

    if (trbfIsKmeans) {
        trbfData = await loadFile('trbf.webp').then(decodeWebP);
    } else {
        [trbfL, trbfU] = await Promise.all([
            loadFile('trbf_l.webp').then(decodeWebP),
            loadFile('trbf_u.webp').then(decodeWebP)
        ]);
    }

    // Allocate output arrays
    const x = new Float32Array(count);
    const y = new Float32Array(count);
    const z = new Float32Array(count);
    const scale_0 = new Float32Array(count);
    const scale_1 = new Float32Array(count);
    const scale_2 = new Float32Array(count);
    const rot_0 = new Float32Array(count);
    const rot_1 = new Float32Array(count);
    const rot_2 = new Float32Array(count);
    const rot_3 = new Float32Array(count);
    const f_dc_0 = new Float32Array(count);
    const f_dc_1 = new Float32Array(count);
    const f_dc_2 = new Float32Array(count);
    const opacity = new Float32Array(count);
    const motion_0 = new Float32Array(count);
    const motion_1 = new Float32Array(count);
    const motion_2 = new Float32Array(count);
    const trbf_center = new Float32Array(count);
    const trbf_scale = new Float32Array(count);

    // Decode means (position)
    console.log('  Decoding means...');
    const meansLRgba = meansL.rgba;
    const meansURgba = meansU.rgba;
    const meansMins = meta.means.mins;
    const meansMaxs = meta.means.maxs;

    for (let i = 0; i < count; i++) {
        const o = i * 4;
        const xLog = dequantize16bit(meansLRgba[o], meansURgba[o], meansMins[0], meansMaxs[0]);
        const yLog = dequantize16bit(meansLRgba[o + 1], meansURgba[o + 1], meansMins[1], meansMaxs[1]);
        const zLog = dequantize16bit(meansLRgba[o + 2], meansURgba[o + 2], meansMins[2], meansMaxs[2]);
        x[i] = invLogTransform(xLog);
        y[i] = invLogTransform(yLog);
        z[i] = invLogTransform(zLog);
    }

    // Decode quaternions
    console.log('  Decoding quaternions...');
    const quatsRgba = quatsData.rgba;

    for (let i = 0; i < count; i++) {
        const o = i * 4;
        const tag = quatsRgba[o + 3];
        if (tag < 252 || tag > 255) {
            // Invalid tag, use identity quaternion
            rot_0[i] = 0;
            rot_1[i] = 0;
            rot_2[i] = 0;
            rot_3[i] = 1;
            continue;
        }
        const [qx, qy, qz, qw] = unpackQuat(quatsRgba[o], quatsRgba[o + 1], quatsRgba[o + 2], tag);
        rot_0[i] = qx;
        rot_1[i] = qy;
        rot_2[i] = qz;
        rot_3[i] = qw;
    }

    // Decode scales (codebook lookup)
    console.log('  Decoding scales...');
    const scalesRgba = scalesData.rgba;
    const scalesCodebook = new Float32Array(meta.scales.codebook);

    for (let i = 0; i < count; i++) {
        const o = i * 4;
        scale_0[i] = scalesCodebook[scalesRgba[o]];
        scale_1[i] = scalesCodebook[scalesRgba[o + 1]];
        scale_2[i] = scalesCodebook[scalesRgba[o + 2]];
    }

    // Decode colors and opacity (codebook lookup)
    console.log('  Decoding colors and opacity...');
    const sh0Rgba = sh0Data.rgba;
    const colorsCodebook = new Float32Array(meta.sh0.codebook);

    for (let i = 0; i < count; i++) {
        const o = i * 4;
        f_dc_0[i] = colorsCodebook[sh0Rgba[o]];
        f_dc_1[i] = colorsCodebook[sh0Rgba[o + 1]];
        f_dc_2[i] = colorsCodebook[sh0Rgba[o + 2]];
        // Opacity: stored as sigmoid(opacity), convert back to logit
        opacity[i] = sigmoidInv(sh0Rgba[o + 3] / 255);
    }

    // Decode motion vectors
    console.log('  Decoding motion vectors...');
    const motionLRgba = motionL.rgba;
    const motionURgba = motionU.rgba;
    const motionMins = meta.motion.mins;
    const motionMaxs = meta.motion.maxs;

    for (let i = 0; i < count; i++) {
        const o = i * 4;
        const m0Log = dequantize16bit(motionLRgba[o], motionURgba[o], motionMins[0], motionMaxs[0]);
        const m1Log = dequantize16bit(motionLRgba[o + 1], motionURgba[o + 1], motionMins[1], motionMaxs[1]);
        const m2Log = dequantize16bit(motionLRgba[o + 2], motionURgba[o + 2], motionMins[2], motionMaxs[2]);
        motion_0[i] = invLogTransform(m0Log);
        motion_1[i] = invLogTransform(m1Log);
        motion_2[i] = invLogTransform(m2Log);
    }

    // Decode TRBF parameters
    console.log('  Decoding TRBF parameters...');
    
    if (trbfIsKmeans && trbfData) {
        // K-means encoded: lookup from codebooks
        const trbfRgba = trbfData.rgba;
        const centerCodebook = new Float32Array(meta.trbf.center_codebook!);
        const scaleCodebook = new Float32Array(meta.trbf.scale_codebook!);  // Values are log(trbf_scale)

        for (let i = 0; i < count; i++) {
            const o = i * 4;
            trbf_center[i] = centerCodebook[trbfRgba[o]];
            // scale_codebook contains log(trbf_scale), need to exp
            trbf_scale[i] = Math.exp(scaleCodebook[trbfRgba[o + 1]]);
        }
    } else if (trbfL && trbfU) {
        // 16-bit quantized
        const trbfLRgba = trbfL.rgba;
        const trbfURgba = trbfU.rgba;

        for (let i = 0; i < count; i++) {
            const o = i * 4;
            // trbf_center: direct 16-bit quantization
            trbf_center[i] = dequantize16bit(trbfLRgba[o], trbfURgba[o], meta.trbf.center_min!, meta.trbf.center_max!);
            // trbf_scale: stored as log, need to exp
            const scaleLog = dequantize16bit(trbfLRgba[o + 1], trbfURgba[o + 1], meta.trbf.scale_min!, meta.trbf.scale_max!);
            trbf_scale[i] = Math.exp(scaleLog);
        }
    }

    // Build GSplatData
    const properties: any[] = [
        { type: 'float', name: 'x', storage: x, byteSize: 4 },
        { type: 'float', name: 'y', storage: y, byteSize: 4 },
        { type: 'float', name: 'z', storage: z, byteSize: 4 },
        { type: 'float', name: 'scale_0', storage: scale_0, byteSize: 4 },
        { type: 'float', name: 'scale_1', storage: scale_1, byteSize: 4 },
        { type: 'float', name: 'scale_2', storage: scale_2, byteSize: 4 },
        { type: 'float', name: 'rot_0', storage: rot_0, byteSize: 4 },
        { type: 'float', name: 'rot_1', storage: rot_1, byteSize: 4 },
        { type: 'float', name: 'rot_2', storage: rot_2, byteSize: 4 },
        { type: 'float', name: 'rot_3', storage: rot_3, byteSize: 4 },
        { type: 'float', name: 'f_dc_0', storage: f_dc_0, byteSize: 4 },
        { type: 'float', name: 'f_dc_1', storage: f_dc_1, byteSize: 4 },
        { type: 'float', name: 'f_dc_2', storage: f_dc_2, byteSize: 4 },
        { type: 'float', name: 'opacity', storage: opacity, byteSize: 4 },
        { type: 'float', name: 'motion_0', storage: motion_0, byteSize: 4 },
        { type: 'float', name: 'motion_1', storage: motion_1, byteSize: 4 },
        { type: 'float', name: 'motion_2', storage: motion_2, byteSize: 4 },
        { type: 'float', name: 'trbf_center', storage: trbf_center, byteSize: 4 },
        { type: 'float', name: 'trbf_scale', storage: trbf_scale, byteSize: 4 }
    ];

    const gsplatData = new GSplatData([{
        name: 'vertex',
        count: count,
        properties
    }]);

    // Preload segment files into a map for later use
    const zipEntries = new Map<string, ArrayBuffer>();
    for (const segment of meta.segments) {
        const segmentData = await loadFile(segment.url);
        zipEntries.set(segment.url, segmentData);
    }

    console.log('✅ SOG4D parsing complete');

    return { gsplatData, meta, zipEntries };
};

/**
 * Load SOG4D file and create Asset
 */
const loadSog4d = async (assets: AssetRegistry, assetSource: AssetSource, device: any): Promise<Asset> => {
    console.log('🔄 Loading SOG4D file...');

    // Load file data
    const source = await createReadSource(assetSource);
    const zipData = await source.arrayBuffer();

    // Parse SOG4D
    const { gsplatData, meta, zipEntries } = await parseSog4d(zipData);

    // Create DynManifest compatible structure
    const dynManifest: DynManifest = {
        version: meta.version,
        type: 'dyn',  // Use 'dyn' type for compatibility with existing code
        start: meta.start,
        duration: meta.duration,
        fps: meta.fps,
        sh_degree: meta.sh_degree,
        global: {
            url: '',  // Not used for SOG4D
            numSplats: meta.count
        },
        segments: meta.segments
    };

    // Diagnostic output
    if (meta.trbf.encoding === 'kmeans') {
        const centerMin = Math.min(...meta.trbf.center_codebook!);
        const centerMax = Math.max(...meta.trbf.center_codebook!);
        console.log(`📊 TRBF center range (k-means): [${centerMin.toFixed(3)}, ${centerMax.toFixed(3)}]`);
    } else {
        console.log(`📊 TRBF center range: [${meta.trbf.center_min!.toFixed(3)}, ${meta.trbf.center_max!.toFixed(3)}]`);
    }
    console.log(`📊 Manifest: start=${meta.start.toFixed(3)}, duration=${meta.duration.toFixed(3)}, fps=${meta.fps}`);

    // Create asset
    const filename = assetSource.filename || assetSource.url || 'dynamic-splat.sog4d';
    const file = {
        url: assetSource.contents ? `local-asset-${assetId++}` : (assetSource.url ?? filename),
        filename: filename,
        contents: assetSource.contents
    };

    return new Promise<Asset>((resolve, reject) => {
        const asset = new Asset(
            filename,
            'gsplat',
            // @ts-ignore
            file
        );

        // Validate required properties
        const required = [
            'x', 'y', 'z',
            'scale_0', 'scale_1', 'scale_2',
            'rot_0', 'rot_1', 'rot_2', 'rot_3',
            'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
            'motion_0', 'motion_1', 'motion_2',
            'trbf_center', 'trbf_scale'
        ];
        const missing = required.filter(prop => !gsplatData.getProp(prop));
        if (missing.length > 0) {
            reject(new Error(`SOG4D file is missing required properties: ${missing.join(', ')}`));
            return;
        }

        // Create resource and store dynamic metadata
        const resource = new GSplatResource(device, gsplatData);
        (resource as any).dynManifest = dynManifest;
        (resource as any).dynBaseUrl = '';  // Not used for SOG4D
        (resource as any).sog4dSegments = zipEntries;  // Store preloaded segments

        asset.resource = resource;

        // Add asset to registry
        assets.add(asset);

        // Trigger load:data event
        asset.fire('load:data', gsplatData);

        // Mark asset as loaded
        (asset as any)._loaded = true;
        (asset as any)._loading = false;

        // Use setTimeout to ensure event handlers are registered first
        setTimeout(() => {
            asset.fire('load', asset);
            resolve(asset);
        }, 0);
    });
};

export { loadSog4d };
export type { Sog4dMeta };
