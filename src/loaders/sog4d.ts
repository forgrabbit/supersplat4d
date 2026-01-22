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

import { Asset, AssetRegistry, GSplatData, GSplatResource, Vec3 } from 'playcanvas';
// JSZip is loaded globally via script tag in index.html
declare const JSZip: any;

import { AssetSource, createReadSource } from './asset-source';
import type { DynManifest } from './dyn';
import { getNextAssetId } from './asset-id-counter';

const defaultOrientation = new Vec3(0, 0, 180);

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
    
    // Optional cubemap background
    cubemap?: {
        file: string;
        format: string;  // e.g., 'horizontal_cross'
    };
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

/**
 * Parse SOG4D ZIP file and create GSplatData
 * Returns either single splat data or multi-splat indicator
 */
const parseSog4d = async (zipData: ArrayBuffer): Promise<{ gsplatData?: GSplatData, meta?: Sog4dMeta, zipEntries?: Map<string, ArrayBuffer>, cubemapData?: ArrayBuffer, isMulti?: boolean, mainMeta?: any, zip?: any }> => {
    const parseStartTime = performance.now();
    console.log('📦 Parsing SOG4D file...');

    // Load ZIP using global JSZip loaded via script tag
    const zipStartTime = performance.now();
    const zip = await JSZip.loadAsync(zipData);
    const zipTime = performance.now() - zipStartTime;
    console.log(`⏱️  ZIP decompression: ${zipTime.toFixed(2)}ms`);

    // Helper to load file from ZIP
    const loadFile = async (name: string): Promise<ArrayBuffer> => {
        const file = zip.file(name);
        if (!file) {
            throw new Error(`Missing file in SOG4D: ${name}`);
        }
        const data = await file.async('arraybuffer');
        // Log file size for debugging
        const sizeKB = (data.byteLength / 1024).toFixed(2);
        console.log(`  📦 ${name}: ${sizeKB} KB`);
        return data;
    };

    // Parse meta.json
    const metaStartTime = performance.now();
    const metaJson = await loadFile('meta.json');
    const mainMeta: any = JSON.parse(new TextDecoder().decode(metaJson));
    const metaTime = performance.now() - metaStartTime;
    console.log(`⏱️  Meta.json parsing: ${metaTime.toFixed(2)}ms`);

    // Check if this is a multi-splat format
    if (mainMeta.type === 'sog4d_multi') {
        // This is a multi-splat file, return indicator for special handling
        return { isMulti: true, mainMeta, zip };
    }

    if (mainMeta.type !== 'sog4d') {
        throw new Error(`Expected type 'sog4d' or 'sog4d_multi', got '${mainMeta.type}'`);
    }

    const meta: Sog4dMeta = mainMeta;

    console.log(`📊 SOG4D: ${meta.count} splats, ${meta.width}x${meta.height} texture`);
    console.log(`📊 Duration: ${meta.duration}s @ ${meta.fps} fps`);

    const count = meta.count;

    // Determine TRBF encoding mode
    const trbfIsKmeans = meta.trbf.encoding === 'kmeans';

    // Load and decode ALL WebP images in parallel (common files + TRBF files)
    const totalWebpFiles = trbfIsKmeans ? 8 : 9; // 7 common + 1 or 2 TRBF files
    console.log(`  Loading and decoding ${totalWebpFiles} WebP files (all parallel)...`);
    const webpStartTime = performance.now();
    
    // Build the parallel decode array
    const webpPromises: Promise<{ rgba: Uint8Array, width: number, height: number }>[] = [
        loadFile('means_l.webp').then(decodeWebP),
        loadFile('means_u.webp').then(decodeWebP),
        loadFile('quats.webp').then(decodeWebP),
        loadFile('scales.webp').then(decodeWebP),
        loadFile('sh0.webp').then(decodeWebP),
        loadFile('motion_l.webp').then(decodeWebP),
        loadFile('motion_u.webp').then(decodeWebP)
    ];
    
    // Add TRBF files based on encoding mode
    if (trbfIsKmeans) {
        webpPromises.push(loadFile('trbf.webp').then(decodeWebP));
    } else {
        webpPromises.push(
            loadFile('trbf_l.webp').then(decodeWebP),
            loadFile('trbf_u.webp').then(decodeWebP)
        );
    }
    
    // Execute all WebP decodes in parallel
    const webpResults = await Promise.all(webpPromises);
    const webpTime = performance.now() - webpStartTime;
    console.log(`⏱️  WebP decoding (${totalWebpFiles} files, all parallel): ${webpTime.toFixed(2)}ms`);
    
    // Extract results
    const [
        meansL, meansU,
        quatsData,
        scalesData,
        sh0Data,
        motionL, motionU
    ] = webpResults.slice(0, 7);
    
    // Extract TRBF results
    let trbfData: { rgba: Uint8Array, width: number, height: number } | null = null;
    let trbfL: { rgba: Uint8Array, width: number, height: number } | null = null;
    let trbfU: { rgba: Uint8Array, width: number, height: number } | null = null;
    
    if (trbfIsKmeans) {
        trbfData = webpResults[7];
    } else {
        trbfL = webpResults[7];
        trbfU = webpResults[8];
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
    const meansDecodeStartTime = performance.now();
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
    const meansDecodeTime = performance.now() - meansDecodeStartTime;
    console.log(`⏱️  Means decoding: ${meansDecodeTime.toFixed(2)}ms`);

    // Decode quaternions
    const quatsDecodeStartTime = performance.now();
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
    const quatsDecodeTime = performance.now() - quatsDecodeStartTime;
    console.log(`⏱️  Quaternions decoding: ${quatsDecodeTime.toFixed(2)}ms`);

    // Decode scales (codebook lookup)
    const scalesDecodeStartTime = performance.now();
    console.log('  Decoding scales...');
    const scalesRgba = scalesData.rgba;
    const scalesCodebook = new Float32Array(meta.scales.codebook);

    for (let i = 0; i < count; i++) {
        const o = i * 4;
        scale_0[i] = scalesCodebook[scalesRgba[o]];
        scale_1[i] = scalesCodebook[scalesRgba[o + 1]];
        scale_2[i] = scalesCodebook[scalesRgba[o + 2]];
    }
    const scalesDecodeTime = performance.now() - scalesDecodeStartTime;
    console.log(`⏱️  Scales decoding: ${scalesDecodeTime.toFixed(2)}ms`);

    // Decode colors and opacity (codebook lookup)
    const colorsDecodeStartTime = performance.now();
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
    const colorsDecodeTime = performance.now() - colorsDecodeStartTime;
    console.log(`⏱️  Colors/opacity decoding: ${colorsDecodeTime.toFixed(2)}ms`);

    // Decode motion vectors
    const motionDecodeStartTime = performance.now();
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
    const motionDecodeTime = performance.now() - motionDecodeStartTime;
    console.log(`⏱️  Motion vectors decoding: ${motionDecodeTime.toFixed(2)}ms`);

    // Decode TRBF parameters
    const trbfDecodeStartTime = performance.now();
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
    const trbfDecodeTime = performance.now() - trbfDecodeStartTime;
    console.log(`⏱️  TRBF decoding: ${trbfDecodeTime.toFixed(2)}ms`);

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
    const segmentsStartTime = performance.now();
    const zipEntries = new Map<string, ArrayBuffer>();
    for (const segment of meta.segments) {
        const segmentData = await loadFile(segment.url);
        zipEntries.set(segment.url, segmentData);
    }
    const segmentsTime = performance.now() - segmentsStartTime;
    console.log(`⏱️  Segments loading (${meta.segments.length} segments): ${segmentsTime.toFixed(2)}ms`);

    const totalParseTime = performance.now() - parseStartTime;
    console.log('✅ SOG4D parsing complete');
    console.log(`⏱️  Total parsing time: ${totalParseTime.toFixed(2)}ms`);

    // Extract cubemap if present
    let cubemapData: ArrayBuffer | undefined = undefined;
    if (meta.cubemap) {
        const cubemapFile = zip.file(meta.cubemap.file);
        if (cubemapFile) {
            cubemapData = await cubemapFile.async('arraybuffer');
            console.log(`📦 Found cubemap: ${meta.cubemap.file} (${(cubemapData.byteLength / 1024).toFixed(2)} KB)`);
        } else {
            console.warn(`⚠️  Cubemap file '${meta.cubemap.file}' not found in ZIP`);
        }
    }

    return { gsplatData, meta, zipEntries, cubemapData };
};

/**
 * Load SOG4D file and create Asset
 */
const loadSog4d = async (assets: AssetRegistry, assetSource: AssetSource, device: any, events?: any): Promise<Asset> => {
    const totalStartTime = performance.now();
    console.log('🔄 Loading SOG4D file...');

    // Load file data
    const loadStartTime = performance.now();
    const source = await createReadSource(assetSource);
    const zipData = await source.arrayBuffer();
    const loadTime = performance.now() - loadStartTime;
    console.log(`⏱️  SOG4D file download/read: ${loadTime.toFixed(2)}ms`);

    // Parse SOG4D
    const parseStartTime = performance.now();
    const parseResult = await parseSog4d(zipData);
    const parseTime = performance.now() - parseStartTime;
    console.log(`⏱️  SOG4D parsing total: ${parseTime.toFixed(2)}ms`);

    // Check if this is a multi-splat file
    if (parseResult.isMulti) {
        // Handle multi-splat format
        // parseSog4dMulti now loads static splats first, then dynamic splat
        // So loadedAssets = [static1, static2, ..., dynamic]
        const multiAssets = await parseSog4dMulti(parseResult.zip!, parseResult.mainMeta!, assetSource, assets, device, events);
        
        // Return the last asset (dynamic) so it will be added to scene first
        // Then static splats will be added via _pendingMultiSplatAssets
        // This ensures dynamic splat is added last and gets selected automatically
        const dynamicAsset = multiAssets[multiAssets.length - 1];
        const staticAssets = multiAssets.slice(0, -1);
        
        // Store static assets for later loading (they will be added before dynamic)
        if (events && staticAssets.length > 0) {
            (events as any)._pendingMultiSplatAssets = staticAssets;
        }
        
        // Return dynamic asset (will be added to scene last, so it gets selected)
        return dynamicAsset;
    }

    // Single splat format (backward compatible)
    const { gsplatData, meta, zipEntries, cubemapData } = parseResult;

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
        url: assetSource.contents ? `local-asset-${getNextAssetId()}` : (assetSource.url ?? filename),
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
        setTimeout(async () => {
            const totalTime = performance.now() - totalStartTime;
            console.log(`⏱️  SOG4D loading total time: ${totalTime.toFixed(2)}ms`);
            asset.fire('load', asset);
            
            // Auto-load cubemap if present
            if (cubemapData && meta.cubemap && events) {
                try {
                    console.log('🌌 Auto-loading cubemap from SOG4D...');
                    // Create a File object from the cubemap data
                    // Determine MIME type from filename extension
                    // Cubemap is converted to WebP in Python script, so it should be .webp
                    const ext = meta.cubemap.file.toLowerCase().split('.').pop();
                    const mimeType = ext === 'webp' ? 'image/webp' : ext === 'png' ? 'image/png' : ext === 'jpg' || ext === 'jpeg' ? 'image/jpeg' : 'image/webp';
                    const blob = new Blob([cubemapData], { type: mimeType });
                    const file = new File([blob], meta.cubemap.file, { type: mimeType });
                    
                    // Import cubemap using background handler
                    await events.invoke('background.importFromFile', file);
                    // Auto-show the cubemap
                    await events.invoke('background.autoShow', meta.cubemap.file);
                    console.log('✅ Cubemap auto-loaded and displayed');
                } catch (error) {
                    console.warn('⚠️  Failed to auto-load cubemap:', error);
                }
            }
            
            resolve(asset);
        }, 0);
    });
};

/**
 * Parse multi-splat SOG4D file and load all splats
 * This function extracts each splat as a separate virtual SOG4D/SOG file and loads them
 */
const parseSog4dMulti = async (zip: any, mainMeta: any, assetSource: AssetSource, assets: AssetRegistry, device: any, events?: any): Promise<Asset[]> => {
    console.log('📦 Parsing multi-splat SOG4D file...');
    console.log(`  Dynamic: ${mainMeta.dynamic ? 'Yes' : 'No'}`);
    console.log(`  Static splats: ${mainMeta.static ? Object.keys(mainMeta.static).length : 0}`);
    
    const loadedAssets: Asset[] = [];

    // Load static splats FIRST (so dynamic splat will be selected last)
    if (mainMeta.static) {
        const staticNames = Object.keys(mainMeta.static);
        for (const staticName of staticNames) {
            console.log(`🔄 Loading ${staticName}...`);
            
            // Extract static folder as a virtual SOG file
            // Check if static folder exists
            const hasStaticFiles = Object.keys(zip.files).some((path: string) => path.startsWith(`${staticName}/`));
            if (!hasStaticFiles) {
                throw new Error(`Missing ${staticName}/ folder in multi-splat SOG4D`);
            }
            
            const staticZip = new JSZip();
            const staticFilePromises: Promise<void>[] = [];
            
            // Iterate over all files in the main zip and filter by path starting with staticName/
            Object.keys(zip.files).forEach((path: string) => {
                if (path.startsWith(`${staticName}/`) && !path.endsWith('/')) {
                    const file = zip.file(path);
                    if (file) {
                        const cleanPath = path.replace(new RegExp(`^${staticName}/`), '');
                        const promise = file.async('arraybuffer').then((data: ArrayBuffer) => {
                            staticZip.file(cleanPath, data);
                        });
                        staticFilePromises.push(promise);
                    }
                }
            });
            
            await Promise.all(staticFilePromises);
            const staticZipBlob = await staticZip.generateAsync({ type: 'blob', compression: 'DEFLATE' });
            const staticZipArrayBuffer = await staticZipBlob.arrayBuffer();
            
            // Create virtual asset source
            const virtualStaticSource: AssetSource = {
                filename: `${staticName}.sog`,
                url: '',
                contents: staticZipArrayBuffer
            };
            
            // Load static SOG using GSplat loader (which handles SOG format)
            const { loadGsplat } = await import('./gsplat');
            const staticAsset = await loadGsplat(assets, virtualStaticSource);
            loadedAssets.push(staticAsset);
        }
    }

    // Load dynamic splat LAST (so it will be selected automatically)
    if (mainMeta.dynamic) {
        console.log('🔄 Loading dynamic splat...');
        
        // Extract dynamic/ folder as a virtual SOG4D file
        // Check if dynamic folder exists by checking for any file starting with 'dynamic/'
        const hasDynamicFiles = Object.keys(zip.files).some((path: string) => path.startsWith('dynamic/'));
        if (!hasDynamicFiles) {
            throw new Error('Missing dynamic/ folder in multi-splat SOG4D');
        }
        
        // Create a virtual ZIP from dynamic folder
        const dynamicZip = new JSZip();
        const dynamicFilePromises: Promise<void>[] = [];
        
        // Iterate over all files in the main zip and filter by path starting with 'dynamic/'
        Object.keys(zip.files).forEach((path: string) => {
            if (path.startsWith('dynamic/') && !path.endsWith('/')) {
                const file = zip.file(path);
                if (file) {
                    const cleanPath = path.replace(/^dynamic\//, '');
                    const promise = file.async('arraybuffer').then((data: ArrayBuffer) => {
                        dynamicZip.file(cleanPath, data);
                    });
                    dynamicFilePromises.push(promise);
                }
            }
        });
        
        await Promise.all(dynamicFilePromises);
        const dynamicZipBlob = await dynamicZip.generateAsync({ type: 'blob', compression: 'DEFLATE' });
        const dynamicZipArrayBuffer = await dynamicZipBlob.arrayBuffer();
        
        // Create virtual asset source with the extracted ZIP
        const virtualDynamicSource: AssetSource = {
            filename: 'dynamic.sog4d',
            url: '',
            contents: dynamicZipArrayBuffer
        };
        
        // Load dynamic splat using existing loader
        const dynamicAsset = await loadSog4d(assets, virtualDynamicSource, device, events);
        loadedAssets.push(dynamicAsset);
    }

    // Handle cubemap from main meta
    if (mainMeta.cubemap && events) {
        try {
            const cubemapFile = zip.file(mainMeta.cubemap.file);
            if (cubemapFile) {
                const cubemapData = await cubemapFile.async('arraybuffer');
                console.log('🌌 Auto-loading cubemap from multi-splat SOG4D...');
                const ext = mainMeta.cubemap.file.toLowerCase().split('.').pop();
                const mimeType = ext === 'webp' ? 'image/webp' : ext === 'png' ? 'image/png' : ext === 'jpg' || ext === 'jpeg' ? 'image/jpeg' : 'image/webp';
                const blob = new Blob([cubemapData], { type: mimeType });
                const file = new File([blob], mainMeta.cubemap.file, { type: mimeType });
                
                await events.invoke('background.importFromFile', file);
                await events.invoke('background.autoShow', mainMeta.cubemap.file);
                console.log('✅ Cubemap auto-loaded and displayed');
            }
        } catch (error) {
            console.warn('⚠️  Failed to auto-load cubemap:', error);
        }
    }

    return loadedAssets;
};

export { loadSog4d, parseSog4dMulti };
export type { Sog4dMeta };
