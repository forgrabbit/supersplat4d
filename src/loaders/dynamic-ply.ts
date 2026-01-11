/**
 * Dynamic PLY Loader
 * 
 * Loads PLY files with dynamic gaussian attributes (trbf_center, trbf_scale, motion_*)
 * and computes segments in the browser.
 */

import { Asset, AssetRegistry, GSplatData, GSplatResource } from 'playcanvas';

import { AssetSource, createReadSource } from './asset-source';
import type { DynManifest } from './dyn';

// =============================================================================
// Types
// =============================================================================

interface DynamicPlyParams {
    start: number;
    duration: number;
    fps: number;
    sh_degree: number;
}

interface SegmentInfo {
    t0: number;
    t1: number;
    url: string;  // Will be virtual URL like 'segment_0'
    count: number;
    indices: Uint32Array;
}

// =============================================================================
// PLY Header Parsing
// =============================================================================

/**
 * Parse PLY header to check for dynamic properties and cfg_args
 */
const parsePlyHeader = async (data: ArrayBuffer): Promise<{
    isDynamic: boolean;
    cfgArgs: DynamicPlyParams | null;
    headerEndOffset: number;
}> => {
    // Read first 64KB to find header
    const headerBytes = new Uint8Array(data, 0, Math.min(65536, data.byteLength));
    const headerText = new TextDecoder('ascii').decode(headerBytes);
    
    // Find end_header
    const endHeaderIndex = headerText.indexOf('end_header');
    if (endHeaderIndex === -1) {
        throw new Error('Invalid PLY file: missing end_header');
    }
    
    const headerEndOffset = endHeaderIndex + 'end_header'.length + 1;  // +1 for newline
    const header = headerText.substring(0, endHeaderIndex);
    
    // Check for dynamic properties
    const hasTrbfCenter = header.includes('property float trbf_center');
    const hasTrbfScale = header.includes('property float trbf_scale');
    const hasMotion = header.includes('property float motion_0');
    
    const isDynamic = hasTrbfCenter && hasTrbfScale && hasMotion;
    
    // Parse cfg_args comment
    let cfgArgs: DynamicPlyParams | null = null;
    const cfgArgsMatch = header.match(/comment\s+cfg_args:\s*(.+)/i);
    
    if (cfgArgsMatch) {
        const argsText = cfgArgsMatch[1];
        const params: any = {};
        
        // Parse key=value or key value pairs
        const patterns = [
            /duration[=\s]+([0-9.]+)/i,
            /start[=\s]+([0-9.]+)/i,
            /fps[=\s]+([0-9.]+)/i,
            /sh_degree[=\s]+([0-9]+)/i
        ];
        
        const keys = ['duration', 'start', 'fps', 'sh_degree'];
        
        for (let i = 0; i < patterns.length; i++) {
            const match = argsText.match(patterns[i]);
            if (match) {
                params[keys[i]] = parseFloat(match[1]);
            }
        }
        
        if (params.duration !== undefined && params.fps !== undefined) {
            cfgArgs = {
                start: params.start ?? 0,
                duration: params.duration,
                fps: params.fps,
                sh_degree: params.sh_degree ?? 0
            };
        }
    }
    
    return { isDynamic, cfgArgs, headerEndOffset };
};

/**
 * Check if a PLY file is dynamic (has trbf_center, trbf_scale, motion_*)
 * This is a quick check that only reads the header (first 64KB).
 */
const checkPlyIsDynamic = async (assetSource: AssetSource): Promise<{
    isDynamic: boolean;
    cfgArgs: DynamicPlyParams | null;
}> => {
    // Only read first 64KB to check header
    const source = await createReadSource(assetSource, 0, 65536);
    const data = await source.arrayBuffer();
    const result = await parsePlyHeader(data);
    return { isDynamic: result.isDynamic, cfgArgs: result.cfgArgs };
};

// =============================================================================
// Segment Computation (Browser-side)
// =============================================================================

/**
 * Compute segments for dynamic gaussian splatting
 * Uses optimized vectorized operations
 */
const computeSegments = (
    trbfCenter: Float32Array,
    trbfScale: Float32Array,
    opacity: Float32Array,
    params: DynamicPlyParams,
    segmentDuration: number = 0.5,
    opacityThreshold: number = 0.005
): SegmentInfo[] => {
    console.log('🔄 Computing segments...');
    const startTime = performance.now();
    
    const { start, duration, fps } = params;
    const numSplats = trbfCenter.length;
    const numSegments = Math.ceil(duration / segmentDuration);
    const segments: SegmentInfo[] = [];
    
    // Convert trbf_scale from log space to linear space (PLY stores log(trbf_scale))
    const trbfScaleExp = new Float32Array(numSplats);
    for (let i = 0; i < numSplats; i++) {
        trbfScaleExp[i] = Math.exp(trbfScale[i]);
    }
    
    // Debug: Check trbfScale range (after exp)
    let minScale = Infinity, maxScale = -Infinity;
    for (let i = 0; i < numSplats; i++) {
        if (trbfScaleExp[i] < minScale) minScale = trbfScaleExp[i];
        if (trbfScaleExp[i] > maxScale) maxScale = trbfScaleExp[i];
    }
    console.log(`📊 TRBF scale range (after exp): [${minScale.toFixed(6)}, ${maxScale.toFixed(6)}]`);
    
    // Precompute sigmoid of opacity (PLY stores opacity as logit)
    const baseOpacity = new Float32Array(numSplats);
    for (let i = 0; i < numSplats; i++) {
        baseOpacity[i] = 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, opacity[i]))));
    }
    
    // Debug: Check sigmoid opacity range
    let minSigOp = Infinity, maxSigOp = -Infinity;
    for (let i = 0; i < numSplats; i++) {
        if (baseOpacity[i] < minSigOp) minSigOp = baseOpacity[i];
        if (baseOpacity[i] > maxSigOp) maxSigOp = baseOpacity[i];
    }
    console.log(`📊 Sigmoid opacity range: [${minSigOp.toFixed(4)}, ${maxSigOp.toFixed(4)}]`);
    
    for (let segIdx = 0; segIdx < numSegments; segIdx++) {
        const t0 = segIdx * segmentDuration;
        const t1 = Math.min((segIdx + 1) * segmentDuration, duration);
        
        // Sample times within this segment
        const numSamples = Math.ceil((t1 - t0) * fps);
        const sampleTimes: number[] = [];
        for (let s = 0; s < numSamples; s++) {
            sampleTimes.push(start + t0 + s / fps);
        }
        
        // Find visible splats
        const visibleIndices: number[] = [];
        
        for (let i = 0; i < numSplats; i++) {
            const tc = trbfCenter[i];
            const ts = Math.max(trbfScaleExp[i], 1e-6);  // Use exp-transformed scale
            const baseOp = baseOpacity[i];
            
            // Check if visible at any sample time
            let isVisible = false;
            for (const t of sampleTimes) {
                const dt = (t - tc) / ts;
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
        
        segments.push({
            t0,
            t1,
            url: `segment_${segIdx}`,
            count: visibleIndices.length,
            indices: new Uint32Array(visibleIndices)
        });
        
        console.log(`  Segment ${segIdx}: [${t0.toFixed(2)}s - ${t1.toFixed(2)}s], ${visibleIndices.length} active splats`);
    }
    
    const elapsed = performance.now() - startTime;
    console.log(`✅ Segments computed in ${elapsed.toFixed(0)}ms`);
    
    return segments;
};

// =============================================================================
// Main Loader
// =============================================================================

let assetId = 0;

/**
 * Load dynamic PLY file
 */
const loadDynamicPly = async (
    assets: AssetRegistry,
    assetSource: AssetSource,
    params: DynamicPlyParams
): Promise<Asset> => {
    console.log('🔄 Loading dynamic PLY file...');
    
    // First, load the PLY using PlayCanvas's standard loader
    const contents = assetSource.contents && (assetSource.contents instanceof Response ? assetSource.contents : new Response(assetSource.contents));
    
    const file = {
        url: contents ? `local-asset-${assetId++}` : assetSource.url ?? assetSource.filename,
        filename: assetSource.filename,
        contents
    };
    
    const data = {
        decompress: true,
        reorder: false  // Don't reorder for dynamic PLY
    };
    
    return new Promise<Asset>((resolve, reject) => {
        const asset = new Asset(
            assetSource.filename || assetSource.url,
            'gsplat',
            // @ts-ignore
            file,
            data
        );
        
        asset.on('load', () => {
            try {
                const splatData = (asset.resource as GSplatResource).gsplatData as GSplatData;
                
                // Validate required properties
                const required = [
                    'x', 'y', 'z',
                    'scale_0', 'scale_1', 'scale_2',
                    'rot_0', 'rot_1', 'rot_2', 'rot_3',
                    'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
                    'motion_0', 'motion_1', 'motion_2',
                    'trbf_center', 'trbf_scale'
                ];
                
                const missing = required.filter(x => !splatData.getProp(x));
                if (missing.length > 0) {
                    reject(new Error(`Dynamic PLY is missing required properties: ${missing.join(', ')}`));
                    return;
                }
                
                // Get dynamic properties
                const trbfCenter = splatData.getProp('trbf_center') as Float32Array;
                const trbfScaleLog = splatData.getProp('trbf_scale') as Float32Array;
                const opacity = splatData.getProp('opacity') as Float32Array;
                
                console.log(`📊 Dynamic PLY: ${splatData.numSplats} splats`);
                console.log(`📊 Params: start=${params.start}, duration=${params.duration}, fps=${params.fps}`);
                
                // Convert trbf_scale from log to linear (PLY stores log(trbf_scale))
                // This is critical for both segment computation AND rendering
                const trbfScaleLinear = new Float32Array(trbfScaleLog.length);
                for (let i = 0; i < trbfScaleLog.length; i++) {
                    trbfScaleLinear[i] = Math.exp(trbfScaleLog[i]);
                }
                
                // Replace trbf_scale in splatData with exp-converted values
                // This ensures updateDynamicTextures() in splat.ts gets correct values
                const vertexElement = splatData.getElement('vertex');
                const trbfScaleProp = vertexElement.properties.find((p: any) => p.name === 'trbf_scale');
                if (trbfScaleProp) {
                    trbfScaleProp.storage = trbfScaleLinear;
                }
                
                // Compute segments using the log values (computeSegments does its own exp)
                const segments = computeSegments(trbfCenter, trbfScaleLog, opacity, params);
                
                // Create DynManifest
                const dynManifest: DynManifest = {
                    version: 1,
                    type: 'dyn',
                    start: params.start,
                    duration: params.duration,
                    fps: params.fps,
                    sh_degree: params.sh_degree,
                    global: {
                        url: '',
                        numSplats: splatData.numSplats
                    },
                    segments: segments.map(s => ({
                        t0: s.t0,
                        t1: s.t1,
                        url: s.url,
                        count: s.count
                    }))
                };
                
                // Store precomputed segment indices
                const segmentIndicesMap = new Map<string, ArrayBuffer>();
                for (const seg of segments) {
                    segmentIndicesMap.set(seg.url, seg.indices.buffer as ArrayBuffer);
                }
                
                // Attach dynamic metadata to resource
                const resource = asset.resource as GSplatResource;
                (resource as any).dynManifest = dynManifest;
                (resource as any).dynBaseUrl = '';
                (resource as any).sog4dSegments = segmentIndicesMap;
                
                // Diagnostic output (use loop to avoid stack overflow with large arrays)
                let minTrbf = Infinity, maxTrbf = -Infinity;
                for (let i = 0; i < trbfCenter.length; i++) {
                    if (trbfCenter[i] < minTrbf) minTrbf = trbfCenter[i];
                    if (trbfCenter[i] > maxTrbf) maxTrbf = trbfCenter[i];
                }
                console.log(`📊 TRBF center range: [${minTrbf.toFixed(3)}, ${maxTrbf.toFixed(3)}]`);
                console.log(`📊 Manifest: start=${params.start.toFixed(3)}, duration=${params.duration.toFixed(3)}, fps=${params.fps}`);
                
                resolve(asset);
            } catch (error) {
                reject(error);
            }
        });
        
        asset.on('error', (err: string) => {
            reject(err);
        });
        
        assets.add(asset);
        assets.load(asset);
    });
};

export { checkPlyIsDynamic, loadDynamicPly };
export type { DynamicPlyParams };
