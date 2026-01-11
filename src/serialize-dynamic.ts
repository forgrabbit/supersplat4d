/**
 * Serialize dynamic gaussian splatting data
 * Supports PLY and SOG4D formats with time range selection
 */

import { GSplatData } from 'playcanvas';
import { Splat } from './splat';
import { State } from './splat-state';
import type { DynamicExportOptions } from './ui/dynamic-export-dialog';
import type { DynManifest } from './loaders/dyn';

// JSZip is loaded globally via script tag
declare const JSZip: any;

// =============================================================================
// Types
// =============================================================================

interface Writer {
    write(data: Uint8Array): void | Promise<void>;
    close(): any | Promise<any>;
}

interface DynamicSplatInfo {
    splat: Splat;
    manifest: DynManifest;
    splatData: GSplatData;
}

// =============================================================================
// Morton order sorting (Z-order curve for better compression)
// =============================================================================

/**
 * Spread bits for Morton encoding (21-bit input)
 * Uses BigInt for 64-bit operations
 */
const spreadBits = (v: number): bigint => {
    let x = BigInt(v);
    x = (x | (x << 32n)) & 0x1f00000000ffffn;
    x = (x | (x << 16n)) & 0x1f0000ff0000ffn;
    x = (x | (x << 8n)) & 0x100f00f00f00f00fn;
    x = (x | (x << 4n)) & 0x10c30c30c30c30c3n;
    x = (x | (x << 2n)) & 0x1249249249249249n;
    return x;
};

/**
 * Encode 3D coordinates to Morton code (Z-order curve)
 */
const mortonEncode3D = (x: number, y: number, z: number): bigint => {
    return spreadBits(x) | (spreadBits(y) << 1n) | (spreadBits(z) << 2n);
};

/**
 * Sort indices by Morton order for better spatial locality
 */
const sortMortonOrder = (
    x: Float32Array,
    y: Float32Array,
    z: Float32Array,
    indices: Uint32Array
): Uint32Array => {
    const count = indices.length;
    
    // Find min/max for normalization
    let xMin = Infinity, xMax = -Infinity;
    let yMin = Infinity, yMax = -Infinity;
    let zMin = Infinity, zMax = -Infinity;
    
    for (let i = 0; i < count; i++) {
        const idx = indices[i];
        if (x[idx] < xMin) xMin = x[idx]; if (x[idx] > xMax) xMax = x[idx];
        if (y[idx] < yMin) yMin = y[idx]; if (y[idx] > yMax) yMax = y[idx];
        if (z[idx] < zMin) zMin = z[idx]; if (z[idx] > zMax) zMax = z[idx];
    }
    
    const xScale = xMax - xMin || 1;
    const yScale = yMax - yMin || 1;
    const zScale = zMax - zMin || 1;
    const maxVal = (1 << 21) - 1;  // 21-bit precision
    
    // Compute Morton codes
    const mortonCodes: Array<{ index: number; morton: bigint }> = [];
    
    for (let i = 0; i < count; i++) {
        const idx = indices[i];
        const xNorm = Math.floor(((x[idx] - xMin) / xScale) * maxVal);
        const yNorm = Math.floor(((y[idx] - yMin) / yScale) * maxVal);
        const zNorm = Math.floor(((z[idx] - zMin) / zScale) * maxVal);
        
        mortonCodes.push({
            index: idx,
            morton: mortonEncode3D(xNorm, yNorm, zNorm)
        });
    }
    
    // Sort by Morton code
    mortonCodes.sort((a, b) => {
        if (a.morton < b.morton) return -1;
        if (a.morton > b.morton) return 1;
        return 0;
    });
    
    // Return sorted indices
    const sortedIndices = new Uint32Array(count);
    for (let i = 0; i < count; i++) {
        sortedIndices[i] = mortonCodes[i].index;
    }
    
    return sortedIndices;
};

// =============================================================================
// Utility functions
// =============================================================================

/**
 * Compute which splat indices are visible within the given time range
 * Also filters out deleted splats (respects user edits)
 */
const computeVisibleIndices = (
    splatData: GSplatData,
    manifest: DynManifest,
    exportStart: number,
    exportDuration: number,
    opacityThreshold: number = 0.005
): Uint32Array => {
    const trbfCenter = splatData.getProp('trbf_center') as Float32Array;
    const trbfScale = splatData.getProp('trbf_scale') as Float32Array;
    const opacity = splatData.getProp('opacity') as Float32Array;
    const state = splatData.getProp('state') as Uint8Array | null;
    
    if (!trbfCenter || !trbfScale || !opacity) {
        throw new Error('Missing dynamic properties');
    }
    
    const numSplats = splatData.numSplats;
    const fps = manifest.fps;
    const exportEnd = exportStart + exportDuration;
    
    // Sample times within the export range
    const numSamples = Math.ceil(exportDuration * fps);
    const sampleTimes: number[] = [];
    for (let s = 0; s <= numSamples; s++) {
        sampleTimes.push(exportStart + s / fps);
    }
    
    const visibleIndices: number[] = [];
    
    for (let i = 0; i < numSplats; i++) {
        // Skip deleted splats (respect user edits)
        if (state && (state[i] & State.deleted) !== 0) {
            continue;
        }
        
        const tc = trbfCenter[i];
        const ts = trbfScale[i];  // Already exp-converted
        const opLogit = opacity[i];
        
        // Compute sigmoid of opacity
        const baseOp = 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, opLogit))));
        
        // Check if visible at any sample time
        let isVisible = false;
        for (const t of sampleTimes) {
            const dt = (t - tc) / Math.max(ts, 1e-6);
            const trbfGauss = Math.exp(-dt * dt);
            const dynOpacity = baseOp * trbfGauss;
            
            if (dynOpacity > opacityThreshold) {
                isVisible = true;
                break;
            }
        }
        
        if (isVisible) {
            visibleIndices.push(i);
        }
    }
    
    console.log(`📊 Visible splats in time range [${exportStart.toFixed(2)}, ${exportEnd.toFixed(2)}]: ${visibleIndices.length} / ${numSplats}`);
    
    return new Uint32Array(visibleIndices);
};

// =============================================================================
// Dynamic PLY Export
// =============================================================================

/**
 * Export dynamic gaussian as PLY file with time range selection
 */
const serializeDynamicPly = async (
    info: DynamicSplatInfo,
    options: DynamicExportOptions,
    writer: Writer,
    progress?: (p: number) => void
): Promise<void> => {
    const { splat, manifest, splatData } = info;
    const { start: exportStart, duration: exportDuration } = options;
    
    console.log(`📤 Exporting dynamic PLY: start=${exportStart}, duration=${exportDuration}`);
    
    // Find visible splats
    const visibleIndices = computeVisibleIndices(splatData, manifest, exportStart, exportDuration);
    const numVisible = visibleIndices.length;
    
    if (numVisible === 0) {
        throw new Error('No visible splats in the selected time range');
    }
    
    // Get all property names
    const vertexElement = splatData.getElement('vertex');
    const props = vertexElement.properties.filter((p: any) => 
        p.name !== 'state' && p.name !== 'transform'
    );
    
    // Build header with cfg_args
    const cfgArgs = `comment cfg_args: start=${exportStart} duration=${exportDuration} fps=${manifest.fps} sh_degree=${manifest.sh_degree || 0}`;
    
    const headerLines = [
        'ply',
        'format binary_little_endian 1.0',
        cfgArgs,
        `element vertex ${numVisible}`,
        ...props.map((p: any) => `property ${p.type === 'uchar' ? 'uchar' : 'float'} ${p.name}`),
        'end_header',
        ''
    ];
    
    const headerText = headerLines.join('\n');
    const headerBytes = new TextEncoder().encode(headerText);
    
    // Calculate bytes per splat
    const bytesPerSplat = props.reduce((sum: number, p: any) => 
        sum + (p.type === 'uchar' ? 1 : 4), 0);
    
    const totalBytes = headerBytes.length + numVisible * bytesPerSplat;
    let bytesWritten = 0;
    
    // Write header
    await writer.write(headerBytes);
    bytesWritten += headerBytes.length;
    
    // Write splat data
    const bufferSize = 1024 * bytesPerSplat;
    const buf = new Uint8Array(bufferSize);
    const dataView = new DataView(buf.buffer);
    let offset = 0;
    
    for (let idx = 0; idx < numVisible; idx++) {
        const i = visibleIndices[idx];
        
        for (const prop of props) {
            const storage = prop.storage as Float32Array | Uint8Array;
            let value = storage[i];
            
            // trbf_scale in splatData is linear (exp-converted), but PLY stores log(trbf_scale)
            if (prop.name === 'trbf_scale') {
                value = Math.log(Math.max(value, 1e-8));
            }
            
            if (prop.type === 'uchar') {
                dataView.setUint8(offset, value);
                offset += 1;
            } else {
                dataView.setFloat32(offset, value, true);
                offset += 4;
            }
        }
        
        // Flush buffer if full
        if (offset >= bufferSize) {
            await writer.write(buf);
            bytesWritten += bufferSize;
            offset = 0;
            
            if (progress) {
                progress(bytesWritten / totalBytes);
            }
        }
    }
    
    // Write remaining data
    if (offset > 0) {
        await writer.write(new Uint8Array(buf.buffer, 0, offset));
    }
    
    console.log(`✅ Dynamic PLY exported: ${numVisible} splats`);
};

// =============================================================================
// SOG4D Export (Browser-side compression)
// =============================================================================

/**
 * Simple k-means clustering (browser implementation, async for UI responsiveness)
 */
const kMeansAsync = async (data: Float32Array, k: number, maxIter: number = 20): Promise<{ centroids: Float32Array, labels: Uint8Array }> => {
    const n = data.length;
    
    // Initialize centroids using k-means++
    const centroids = new Float32Array(k);
    const labels = new Uint8Array(n);
    
    // First centroid: random
    centroids[0] = data[Math.floor(Math.random() * n)];
    
    // Remaining centroids: probability proportional to distance squared
    for (let c = 1; c < k; c++) {
        const distances = new Float32Array(n);
        let totalDist = 0;
        
        for (let i = 0; i < n; i++) {
            let minDist = Infinity;
            for (let j = 0; j < c; j++) {
                const d = Math.abs(data[i] - centroids[j]);
                if (d < minDist) minDist = d;
            }
            distances[i] = minDist * minDist;
            totalDist += distances[i];
        }
        
        let r = Math.random() * totalDist;
        for (let i = 0; i < n; i++) {
            r -= distances[i];
            if (r <= 0) {
                centroids[c] = data[i];
                break;
            }
        }
        
        // Yield every few centroids for large k
        if (c % 50 === 0) await yieldToMain();
    }
    
    // Iterate
    for (let iter = 0; iter < maxIter; iter++) {
        // Assign labels
        for (let i = 0; i < n; i++) {
            let minDist = Infinity;
            let minLabel = 0;
            for (let c = 0; c < k; c++) {
                const d = Math.abs(data[i] - centroids[c]);
                if (d < minDist) {
                    minDist = d;
                    minLabel = c;
                }
            }
            labels[i] = minLabel;
        }
        
        // Update centroids
        const sums = new Float32Array(k);
        const counts = new Uint32Array(k);
        
        for (let i = 0; i < n; i++) {
            sums[labels[i]] += data[i];
            counts[labels[i]]++;
        }
        
        for (let c = 0; c < k; c++) {
            if (counts[c] > 0) {
                centroids[c] = sums[c] / counts[c];
            }
        }
        
        // Yield every iteration to keep UI responsive
        await yieldToMain();
    }
    
    // Sort centroids and remap labels
    const sortedIndices = Array.from({ length: k }, (_, i) => i)
        .sort((a, b) => centroids[a] - centroids[b]);
    
    const sortedCentroids = new Float32Array(k);
    const remapTable = new Uint8Array(k);
    
    for (let i = 0; i < k; i++) {
        sortedCentroids[i] = centroids[sortedIndices[i]];
        remapTable[sortedIndices[i]] = i;
    }
    
    for (let i = 0; i < n; i++) {
        labels[i] = remapTable[labels[i]];
    }
    
    return { centroids: sortedCentroids, labels };
};

/**
 * Quantize float to 16-bit
 */
const quantize16bit = (value: number, min: number, max: number): [number, number] => {
    // Handle edge case where min == max
    const range = max - min;
    const normalized = range > 1e-10 ? (value - min) / range : 0.5;
    const quantized = Math.round(normalized * 65535);
    const clamped = Math.max(0, Math.min(65535, quantized));
    return [clamped & 0xff, (clamped >> 8) & 0xff];
};

/**
 * Log transform for better distribution: sign(x) * log(1 + |x|)
 */
const logTransform = (x: number): number => {
    const a = Math.abs(x);
    const l = Math.log(1 + a);
    return x < 0 ? -l : l;
};

/**
 * Yield to main thread to allow UI updates
 */
const yieldToMain = (): Promise<void> => {
    return new Promise(resolve => setTimeout(resolve, 0));
};

/**
 * Create WebP image from RGBA data (using canvas)
 */
const createWebP = async (rgba: Uint8Array, width: number, height: number): Promise<Blob> => {
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d')!;
    const imageData = new ImageData(new Uint8ClampedArray(rgba), width, height);
    ctx.putImageData(imageData, 0, 0);
    return await canvas.convertToBlob({ type: 'image/webp', quality: 1.0 });
};

/**
 * Export dynamic gaussian as SOG4D file (browser-side compression)
 */
const serializeSog4d = async (
    info: DynamicSplatInfo,
    options: DynamicExportOptions,
    writer: Writer,
    progress?: (p: number, stage: string) => void
): Promise<void> => {
    const { manifest, splatData } = info;
    const { start: exportStart, duration: exportDuration, filename } = options;
    
    console.log(`📤 Exporting SOG4D: start=${exportStart}, duration=${exportDuration}`);
    
    if (progress) progress(0, 'Finding visible splats...');
    await yieldToMain();
    
    // Find visible splats
    const visibleIndicesUnsorted = computeVisibleIndices(splatData, manifest, exportStart, exportDuration);
    const count = visibleIndicesUnsorted.length;
    
    if (count === 0) {
        throw new Error('No visible splats in the selected time range');
    }
    
    // Get property arrays (needed for Morton sorting and encoding)
    const x = splatData.getProp('x') as Float32Array;
    const y = splatData.getProp('y') as Float32Array;
    const z = splatData.getProp('z') as Float32Array;
    
    // Sort by Morton order for better compression
    if (progress) progress(0.02, 'Sorting by Morton order...');
    await yieldToMain();
    
    const visibleIndices = sortMortonOrder(x, y, z, visibleIndicesUnsorted);
    
    console.log(`📊 Sorted ${count} splats by Morton order`);
    
    // Calculate texture dimensions (aligned to 4 for better compression)
    const width = Math.ceil(Math.ceil(Math.sqrt(count)) / 4) * 4;
    const height = Math.ceil(count / width);
    const textureSize = width * height;
    
    console.log(`📊 SOG4D: ${count} splats, texture ${width}x${height}`);
    const scale0 = splatData.getProp('scale_0') as Float32Array;
    const scale1 = splatData.getProp('scale_1') as Float32Array;
    const scale2 = splatData.getProp('scale_2') as Float32Array;
    const rot0 = splatData.getProp('rot_0') as Float32Array;
    const rot1 = splatData.getProp('rot_1') as Float32Array;
    const rot2 = splatData.getProp('rot_2') as Float32Array;
    const rot3 = splatData.getProp('rot_3') as Float32Array;
    const fdc0 = splatData.getProp('f_dc_0') as Float32Array;
    const fdc1 = splatData.getProp('f_dc_1') as Float32Array;
    const fdc2 = splatData.getProp('f_dc_2') as Float32Array;
    const opacity = splatData.getProp('opacity') as Float32Array;
    const motion0 = splatData.getProp('motion_0') as Float32Array;
    const motion1 = splatData.getProp('motion_1') as Float32Array;
    const motion2 = splatData.getProp('motion_2') as Float32Array;
    const trbfCenter = splatData.getProp('trbf_center') as Float32Array;
    const trbfScale = splatData.getProp('trbf_scale') as Float32Array;
    
    // =========================================================================
    // Encode position (16-bit quantized with log transform)
    // =========================================================================
    if (progress) progress(0.05, 'Encoding positions...');
    await yieldToMain();
    
    // Compute min/max for positions
    let xMin = Infinity, xMax = -Infinity;
    let yMin = Infinity, yMax = -Infinity;
    let zMin = Infinity, zMax = -Infinity;
    
    for (let idx = 0; idx < count; idx++) {
        const i = visibleIndices[idx];
        const xl = logTransform(x[i]), yl = logTransform(y[i]), zl = logTransform(z[i]);
        if (xl < xMin) xMin = xl; if (xl > xMax) xMax = xl;
        if (yl < yMin) yMin = yl; if (yl > yMax) yMax = yl;
        if (zl < zMin) zMin = zl; if (zl > zMax) zMax = zl;
    }
    
    const meansL = new Uint8Array(textureSize * 4);
    const meansU = new Uint8Array(textureSize * 4);
    
    for (let idx = 0; idx < count; idx++) {
        const i = visibleIndices[idx];
        const o = idx * 4;
        
        const [xl, xh] = quantize16bit(logTransform(x[i]), xMin, xMax);
        const [yl, yh] = quantize16bit(logTransform(y[i]), yMin, yMax);
        const [zl, zh] = quantize16bit(logTransform(z[i]), zMin, zMax);
        
        meansL[o] = xl; meansL[o + 1] = yl; meansL[o + 2] = zl; meansL[o + 3] = 255;
        meansU[o] = xh; meansU[o + 1] = yh; meansU[o + 2] = zh; meansU[o + 3] = 255;
    }
    
    // =========================================================================
    // Encode rotation (largest component compression - matching Python/decoder)
    // =========================================================================
    if (progress) progress(0.15, 'Encoding rotations...');
    await yieldToMain();
    
    const quats = new Uint8Array(textureSize * 4);
    const sqrt2 = Math.sqrt(2.0);
    
    // Index mapping: for each max component, which other 3 components to store
    const idxMap = [
        [1, 2, 3],  // max is 0: store 1,2,3
        [0, 2, 3],  // max is 1: store 0,2,3
        [0, 1, 3],  // max is 2: store 0,1,3
        [0, 1, 2]   // max is 3: store 0,1,2
    ];
    
    for (let idx = 0; idx < count; idx++) {
        const i = visibleIndices[idx];
        const o = idx * 4;
        
        let q = [rot0[i], rot1[i], rot2[i], rot3[i]];
        
        // Normalize
        const norm = Math.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
        if (norm > 0) {
            q = q.map(v => v / norm);
        }
        
        // Find max component (by absolute value)
        let maxComp = 0;
        let maxVal = Math.abs(q[0]);
        for (let j = 1; j < 4; j++) {
            if (Math.abs(q[j]) > maxVal) {
                maxVal = Math.abs(q[j]);
                maxComp = j;
            }
        }
        
        // Make max component positive (flip entire quaternion if needed)
        if (q[maxComp] < 0) {
            q = q.map(v => -v);
        }
        
        // Scale by sqrt(2) to fit remaining components in [-1, 1]
        q = q.map(v => v * sqrt2);
        
        // Get the 3 non-max components
        const indices = idxMap[maxComp];
        
        // Quantize to 8-bit: (q * 0.5 + 0.5) * 255
        quats[o] = Math.round(Math.max(0, Math.min(255, (q[indices[0]] * 0.5 + 0.5) * 255)));
        quats[o + 1] = Math.round(Math.max(0, Math.min(255, (q[indices[1]] * 0.5 + 0.5) * 255)));
        quats[o + 2] = Math.round(Math.max(0, Math.min(255, (q[indices[2]] * 0.5 + 0.5) * 255)));
        quats[o + 3] = 252 + maxComp;  // Tag: 252-255 indicates which component is max
    }
    
    // =========================================================================
    // Encode scale (k-means with joint clustering of all components)
    // =========================================================================
    if (progress) progress(0.25, 'Encoding scales (k-means)...');
    await yieldToMain();
    
    // Concatenate all scale values for joint clustering (like Python script)
    const allScales = new Float32Array(count * 3);
    for (let idx = 0; idx < count; idx++) {
        const i = visibleIndices[idx];
        allScales[idx] = scale0[i];
        allScales[count + idx] = scale1[i];
        allScales[count * 2 + idx] = scale2[i];
    }
    
    // K-means on all scale values
    const { centroids: scaleCentroids, labels: scaleLabels } = await kMeansAsync(allScales, 256);
    
    // Split labels back to components
    const scales = new Uint8Array(textureSize * 4);
    for (let idx = 0; idx < count; idx++) {
        const o = idx * 4;
        scales[o] = scaleLabels[idx];              // scale_0 label
        scales[o + 1] = scaleLabels[count + idx];  // scale_1 label
        scales[o + 2] = scaleLabels[count * 2 + idx];  // scale_2 label
        scales[o + 3] = 255;
    }
    
    // =========================================================================
    // Encode color + opacity (k-means with joint clustering of all components)
    // =========================================================================
    if (progress) progress(0.35, 'Encoding colors (k-means)...');
    await yieldToMain();
    
    // Concatenate all color values for joint clustering (like Python script)
    const allColors = new Float32Array(count * 3);
    for (let idx = 0; idx < count; idx++) {
        const i = visibleIndices[idx];
        allColors[idx] = fdc0[i];
        allColors[count + idx] = fdc1[i];
        allColors[count * 2 + idx] = fdc2[i];
    }
    
    // K-means on all color values
    const { centroids: colorCentroids, labels: colorLabels } = await kMeansAsync(allColors, 256);
    
    // Split labels back to components
    const sh0 = new Uint8Array(textureSize * 4);
    for (let idx = 0; idx < count; idx++) {
        const i = visibleIndices[idx];
        const o = idx * 4;
        
        sh0[o] = colorLabels[idx];               // f_dc_0 label
        sh0[o + 1] = colorLabels[count + idx];   // f_dc_1 label
        sh0[o + 2] = colorLabels[count * 2 + idx];  // f_dc_2 label
        
        // Quantize opacity (logit -> sigmoid -> 0-255)
        const sigmoidOp = 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, opacity[i]))));
        sh0[o + 3] = Math.round(sigmoidOp * 255);
    }
    
    // =========================================================================
    // Encode motion (16-bit quantized with log transform)
    // =========================================================================
    if (progress) progress(0.45, 'Encoding motion...');
    await yieldToMain();
    
    let m0Min = Infinity, m0Max = -Infinity;
    let m1Min = Infinity, m1Max = -Infinity;
    let m2Min = Infinity, m2Max = -Infinity;
    
    for (let idx = 0; idx < count; idx++) {
        const i = visibleIndices[idx];
        const m0l = logTransform(motion0[i]);
        const m1l = logTransform(motion1[i]);
        const m2l = logTransform(motion2[i]);
        if (m0l < m0Min) m0Min = m0l; if (m0l > m0Max) m0Max = m0l;
        if (m1l < m1Min) m1Min = m1l; if (m1l > m1Max) m1Max = m1l;
        if (m2l < m2Min) m2Min = m2l; if (m2l > m2Max) m2Max = m2l;
    }
    
    const motionL = new Uint8Array(textureSize * 4);
    const motionU = new Uint8Array(textureSize * 4);
    
    for (let idx = 0; idx < count; idx++) {
        const i = visibleIndices[idx];
        const o = idx * 4;
        
        const [m0l, m0h] = quantize16bit(logTransform(motion0[i]), m0Min, m0Max);
        const [m1l, m1h] = quantize16bit(logTransform(motion1[i]), m1Min, m1Max);
        const [m2l, m2h] = quantize16bit(logTransform(motion2[i]), m2Min, m2Max);
        
        motionL[o] = m0l; motionL[o + 1] = m1l; motionL[o + 2] = m2l; motionL[o + 3] = 255;
        motionU[o] = m0h; motionU[o + 1] = m1h; motionU[o + 2] = m2h; motionU[o + 3] = 255;
    }
    
    // =========================================================================
    // Encode TRBF (k-means)
    // =========================================================================
    if (progress) progress(0.55, 'Encoding TRBF (k-means)...');
    await yieldToMain();
    
    const tcData = new Float32Array(count);
    const tsData = new Float32Array(count);
    
    for (let idx = 0; idx < count; idx++) {
        const i = visibleIndices[idx];
        tcData[idx] = trbfCenter[i];
        tsData[idx] = Math.log(Math.max(trbfScale[i], 1e-8));  // Store as log
    }
    
    const { centroids: tcCentroids, labels: tcLabels } = await kMeansAsync(tcData, 256);
    const { centroids: tsCentroids, labels: tsLabels } = await kMeansAsync(tsData, 256);
    
    const trbf = new Uint8Array(textureSize * 4);
    for (let idx = 0; idx < count; idx++) {
        const o = idx * 4;
        trbf[o] = tcLabels[idx];
        trbf[o + 1] = tsLabels[idx];
        trbf[o + 2] = 0;
        trbf[o + 3] = 255;
    }
    
    // =========================================================================
    // Compute segments
    // =========================================================================
    if (progress) progress(0.65, 'Computing segments...');
    await yieldToMain();
    
    const segmentDuration = 0.5;
    const fps = manifest.fps;
    const numSegments = Math.ceil(exportDuration / segmentDuration);
    const segments: Array<{ t0: number; t1: number; url: string; count: number; data: Uint8Array }> = [];
    
    for (let segIdx = 0; segIdx < numSegments; segIdx++) {
        const t0 = segIdx * segmentDuration;
        const t1 = Math.min((segIdx + 1) * segmentDuration, exportDuration);
        
        // Sample times
        const sampleTimes: number[] = [];
        for (let s = 0; s <= Math.ceil((t1 - t0) * fps); s++) {
            sampleTimes.push(exportStart + t0 + s / fps);
        }
        
        // Find visible splats in this segment
        const segmentIndices: number[] = [];
        
        for (let idx = 0; idx < count; idx++) {
            const i = visibleIndices[idx];
            const tc = trbfCenter[i];
            const ts = trbfScale[i];
            const opLogit = opacity[i];
            const baseOp = 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, opLogit))));
            
            let isVisible = false;
            for (const t of sampleTimes) {
                const dt = (t - tc) / Math.max(ts, 1e-6);
                const trbfGauss = Math.exp(-dt * dt);
                if (baseOp * trbfGauss > 0.005) {
                    isVisible = true;
                    break;
                }
            }
            
            if (isVisible) {
                segmentIndices.push(idx);  // Index in compressed data
            }
        }
        
        const segData = new Uint32Array(segmentIndices);
        segments.push({
            t0,
            t1,
            url: `segments/seg_${String(segIdx).padStart(3, '0')}.act`,
            count: segmentIndices.length,
            data: new Uint8Array(segData.buffer)
        });
    }
    
    // =========================================================================
    // Create WebP images
    // =========================================================================
    if (progress) progress(0.75, 'Creating WebP images...');
    await yieldToMain();
    
    const [
        meansLWebp, meansUWebp,
        quatsWebp,
        scalesWebp,
        sh0Webp,
        motionLWebp, motionUWebp,
        trbfWebp
    ] = await Promise.all([
        createWebP(meansL, width, height),
        createWebP(meansU, width, height),
        createWebP(quats, width, height),
        createWebP(scales, width, height),
        createWebP(sh0, width, height),
        createWebP(motionL, width, height),
        createWebP(motionU, width, height),
        createWebP(trbf, width, height)
    ]);
    
    // =========================================================================
    // Build metadata
    // =========================================================================
    const meta = {
        version: 1,
        type: 'sog4d',
        generator: 'SuperSplat4D Browser Export',
        count,
        width,
        height,
        sh_degree: manifest.sh_degree || 0,
        start: exportStart,
        duration: exportDuration,
        fps: manifest.fps,
        means: {
            mins: [xMin, yMin, zMin],
            maxs: [xMax, yMax, zMax],
            files: ['means_l.webp', 'means_u.webp']
        },
        quats: {
            files: ['quats.webp']
        },
        scales: {
            codebook: Array.from(scaleCentroids),
            files: ['scales.webp']
        },
        sh0: {
            codebook: Array.from(colorCentroids),
            files: ['sh0.webp']
        },
        motion: {
            mins: [m0Min, m1Min, m2Min],
            maxs: [m0Max, m1Max, m2Max],
            files: ['motion_l.webp', 'motion_u.webp']
        },
        trbf: {
            encoding: 'kmeans',
            center_codebook: Array.from(tcCentroids),
            scale_codebook: Array.from(tsCentroids),
            files: ['trbf.webp']
        },
        segments: segments.map(s => ({ t0: s.t0, t1: s.t1, url: s.url, count: s.count }))
    };
    
    // =========================================================================
    // Create ZIP file
    // =========================================================================
    if (progress) progress(0.9, 'Creating ZIP archive...');
    await yieldToMain();
    
    const zip = new JSZip();
    
    zip.file('meta.json', JSON.stringify(meta, null, 2));
    zip.file('means_l.webp', meansLWebp);
    zip.file('means_u.webp', meansUWebp);
    zip.file('quats.webp', quatsWebp);
    zip.file('scales.webp', scalesWebp);
    zip.file('sh0.webp', sh0Webp);
    zip.file('motion_l.webp', motionLWebp);
    zip.file('motion_u.webp', motionUWebp);
    zip.file('trbf.webp', trbfWebp);
    
    for (const seg of segments) {
        zip.file(seg.url, seg.data);
    }
    
    const zipBlob = await zip.generateAsync({ type: 'uint8array' });
    await writer.write(zipBlob);
    
    if (progress) progress(1.0, 'Done!');
    console.log(`✅ SOG4D exported: ${count} splats, ${segments.length} segments`);
};

export { serializeDynamicPly, serializeSog4d, computeVisibleIndices };
export type { DynamicSplatInfo };
