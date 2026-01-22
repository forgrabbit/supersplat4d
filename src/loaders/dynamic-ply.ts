/**
 * Dynamic PLY Loader
 * 
 * Loads PLY files with dynamic gaussian attributes (trbf_center, trbf_scale, motion_*)
 * and computes segments in the browser.
 */

import { Asset, AssetRegistry, GSplatData, GSplatResource } from 'playcanvas';

import { AssetSource, createReadSource } from './asset-source';
import type { DynManifest } from './dyn';
import { getNextAssetId } from './asset-id-counter';

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
// PLY Header Parsing and Custom Property Loading
// =============================================================================

/**
 * Manually parse and add dynamic properties to GSplatData
 * PlayCanvas only loads standard properties, so we need to parse custom ones ourselves
 */
const parseAndAddDynamicProperties = async (splatData: GSplatData, rawData: ArrayBuffer): Promise<void> => {
    const headerBytes = new Uint8Array(rawData, 0, Math.min(65536, rawData.byteLength));
    const headerText = new TextDecoder('ascii').decode(headerBytes);
    
    const endHeaderIndex = headerText.indexOf('end_header');
    if (endHeaderIndex === -1) {
        throw new Error('Invalid PLY: missing end_header');
    }
    
    const header = headerText.substring(0, endHeaderIndex);
    const headerEndOffset = endHeaderIndex + 'end_header'.length + 1;
    
    // Parse property declarations
    const propertyLines = header.split('\n').filter(line => line.trim().startsWith('property'));
    const properties: Array<{ type: string; name: string; byteSize: number }> = [];
    
    for (const line of propertyLines) {
        const match = line.match(/property\s+(float|uchar|double|int)\s+(\w+)/);
        if (match) {
            const type = match[1];
            const name = match[2];
            const byteSize = type === 'uchar' ? 1 : 4;
            properties.push({ type, name, byteSize });
        }
    }
    
    // Get vertex count
    const vertexMatch = header.match(/element\s+vertex\s+(\d+)/);
    if (!vertexMatch) {
        throw new Error('Invalid PLY: missing vertex element');
    }
    const vertexCount = parseInt(vertexMatch[1], 10);
    
    // Calculate bytes per vertex
    const bytesPerVertex = properties.reduce((sum, p) => sum + p.byteSize, 0);
    
    // Read binary data
    const dataView = new DataView(rawData, headerEndOffset);
    
    // Find dynamic property indices
    const dynamicProps = ['motion_0', 'motion_1', 'motion_2', 'trbf_center', 'trbf_scale'];
    const dynamicPropIndices = new Map<string, number>();
    
    for (const propName of dynamicProps) {
        const idx = properties.findIndex(p => p.name === propName);
        if (idx >= 0) {
            dynamicPropIndices.set(propName, idx);
        }
    }
    
    if (dynamicPropIndices.size === 0) {
        console.log('No dynamic properties found in PLY');
        return;
    }
    
    console.log(`📝 Parsing ${dynamicPropIndices.size} dynamic properties from ${vertexCount} vertices...`);
    
    // Create storage arrays for dynamic properties
    const propData = new Map<string, Float32Array>();
    for (const propName of dynamicProps) {
        if (dynamicPropIndices.has(propName)) {
            propData.set(propName, new Float32Array(vertexCount));
        }
    }
    
    // Parse data
    for (let v = 0; v < vertexCount; v++) {
        let offset = v * bytesPerVertex;
        
        for (let p = 0; p < properties.length; p++) {
            const prop = properties[p];
            
            if (dynamicPropIndices.has(prop.name)) {
                const value = dataView.getFloat32(offset, true); // little endian
                propData.get(prop.name)![v] = value;
            }
            
            offset += prop.byteSize;
        }
    }
    
    // Add dynamic properties to splatData
    const vertexElement = splatData.getElement('vertex');
    for (const [propName, storage] of propData.entries()) {
        // Check if property already exists
        const existing = splatData.getProp(propName);
        if (!existing) {
            vertexElement.properties.push({
                type: 'float',
                name: propName,
                storage,
                byteSize: 4
            });
            console.log(`✅ Added dynamic property: ${propName}`);
        } else {
            console.log(`⚠️  Property ${propName} already exists, skipping`);
        }
    }
    
    console.log('✅ Dynamic properties parsed and added successfully');
};

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
    const computeStartTime = performance.now();
    
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
    
    const elapsed = performance.now() - computeStartTime;
    console.log(`✅ Segments computed in ${elapsed.toFixed(0)}ms`);
    
    return segments;
};

// =============================================================================
// Main Loader
// =============================================================================

/**
 * Load dynamic PLY file
 */
const loadDynamicPly = async (
    assets: AssetRegistry,
    assetSource: AssetSource,
    params: DynamicPlyParams
): Promise<Asset> => {
    const totalStartTime = performance.now();
    console.log('🔄 Loading dynamic PLY file...');
    
    // CRITICAL FIX: Cache the raw data to avoid Response body consumption issues
    // Response body can only be read once, so we read it upfront and reuse
    let cachedRawData: ArrayBuffer | null = null;
    
    if (assetSource.contents) {
        // assetSource.contents is File (from file picker) or could be Response
        // Read it upfront to avoid body consumption issues
        cachedRawData = await assetSource.contents.arrayBuffer();
    }
    
    // Create contents for PlayCanvas (use cached data if available)
    const contents = cachedRawData ? new Response(cachedRawData) : 
                     assetSource.contents ? new Response(assetSource.contents) : 
                     null;
    
    const file = {
        url: contents ? `local-asset-${getNextAssetId()}` : assetSource.url ?? assetSource.filename,
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
        
        // Store segments outside to avoid recomputation
        let computedSegments: SegmentInfo[] | null = null;
        
        // CRITICAL: Use 'load:data' event instead of 'load' event
        // The 'load:data' event fires when GSplatData is fully parsed
        // The 'load' event fires later and may have timing issues with property access
        asset.on('load:data', async (splatData: GSplatData) => {
            try {
                const plyParseStartTime = performance.now();
                
                // CRITICAL FIX: PlayCanvas only loads standard properties
                // We need to manually parse custom dynamic properties from the cached raw PLY data
                if (cachedRawData) {
                    console.log('🔧 Manually parsing dynamic properties from PLY...');
                    await parseAndAddDynamicProperties(splatData, cachedRawData);
                } else {
                    console.warn('⚠️  No cached raw data available for parsing dynamic properties');
                }
                
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
                
                const plyParseTime = performance.now() - plyParseStartTime;
                console.log(`⏱️  PLY parsing: ${plyParseTime.toFixed(2)}ms`);
                console.log(`📊 Dynamic PLY: ${splatData.numSplats} splats`);
                console.log(`📊 Params: start=${params.start}, duration=${params.duration}, fps=${params.fps}`);
                
                // Convert trbf_scale from log to linear (PLY stores log(trbf_scale))
                // This is critical for both segment computation AND rendering
                const trbfConvertStartTime = performance.now();
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
                const trbfConvertTime = performance.now() - trbfConvertStartTime;
                console.log(`⏱️  TRBF scale conversion: ${trbfConvertTime.toFixed(2)}ms`);
                
                // Compute segments using the log values (computeSegments does its own exp)
                const segmentsStartTime = performance.now();
                computedSegments = computeSegments(trbfCenter, trbfScaleLog, opacity, params);
                const segmentsTime = performance.now() - segmentsStartTime;
                console.log(`⏱️  Segments computation: ${segmentsTime.toFixed(2)}ms`);
                
                // Diagnostic output (use loop to avoid stack overflow with large arrays)
                let minTrbf = Infinity, maxTrbf = -Infinity;
                for (let i = 0; i < trbfCenter.length; i++) {
                    if (trbfCenter[i] < minTrbf) minTrbf = trbfCenter[i];
                    if (trbfCenter[i] > maxTrbf) maxTrbf = trbfCenter[i];
                }
                console.log(`📊 TRBF center range: [${minTrbf.toFixed(3)}, ${maxTrbf.toFixed(3)}]`);
                console.log(`📊 Manifest: start=${params.start.toFixed(3)}, duration=${params.duration.toFixed(3)}, fps=${params.fps}`);
            } catch (error) {
                reject(error);
                return;
            }
        });
        
        asset.on('load', () => {
            // CRITICAL FIX: Use setTimeout to ensure PLY properties are fully parsed
            // On first load, PlayCanvas PLY parser may not have finished parsing all properties
            // when the 'load' event fires. A small delay allows parsing to complete.
            setTimeout(() => {
                try {
                    const splatData = (asset.resource as GSplatResource).gsplatData as GSplatData;
                    
                    let segments: SegmentInfo[];
                    
                    // FALLBACK: If load:data didn't process (first load issue), process here
                    if (!computedSegments) {
                        console.warn('⚠️  load:data did not process segments, handling in load event (first load issue)');
                        
                        // DEBUG: Log all available properties
                        const vertexElem = splatData.getElement('vertex');
                        const allPropertyNames = vertexElem.properties.map((p: any) => p.name);
                        console.log('🔍 DEBUG: All available properties:', allPropertyNames.join(', '));
                        console.log('🔍 DEBUG: Total property count:', allPropertyNames.length);
                        console.log('🔍 DEBUG: numSplats:', splatData.numSplats);
                        
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
                        console.log('🔍 DEBUG: Missing properties:', missing.join(', ') || 'none');
                        console.log('🔍 DEBUG: Present dynamic properties:', 
                            ['motion_0', 'motion_1', 'motion_2', 'trbf_center', 'trbf_scale']
                                .filter(x => splatData.getProp(x))
                                .join(', '));
                        
                        if (missing.length > 0) {
                            // Print first few bytes of each property storage to verify data exists
                            required.forEach(propName => {
                                const prop = splatData.getProp(propName);
                                if (prop) {
                                    const sample = prop instanceof Float32Array ? 
                                        `[${prop[0]?.toFixed(3)}, ${prop[1]?.toFixed(3)}, ...]` : 
                                        `[${prop[0]}, ${prop[1]}, ...]`;
                                    console.log(`  ✅ ${propName}: ${prop.length} values, sample: ${sample}`);
                                } else {
                                    console.log(`  ❌ ${propName}: NOT FOUND`);
                                }
                            });
                            reject(new Error(`Dynamic PLY is missing required properties: ${missing.join(', ')}`));
                            return;
                        }
                    
                    // Get dynamic properties
                    const trbfCenter = splatData.getProp('trbf_center') as Float32Array;
                    const trbfScaleLog = splatData.getProp('trbf_scale') as Float32Array;
                    const opacity = splatData.getProp('opacity') as Float32Array;
                    
                    // Convert trbf_scale from log to linear
                    const trbfScaleLinear = new Float32Array(trbfScaleLog.length);
                    for (let i = 0; i < trbfScaleLog.length; i++) {
                        trbfScaleLinear[i] = Math.exp(trbfScaleLog[i]);
                    }
                    
                    // Replace trbf_scale in splatData with exp-converted values
                    const vertexElement = splatData.getElement('vertex');
                    const trbfScaleProp = vertexElement.properties.find((p: any) => p.name === 'trbf_scale');
                    if (trbfScaleProp) {
                        trbfScaleProp.storage = trbfScaleLinear;
                    }
                    
                    // Compute segments
                    segments = computeSegments(trbfCenter, trbfScaleLog, opacity, params);
                    console.log('✅ Segments computed in load event (fallback)');
                } else {
                    // Use pre-computed segments from load:data
                    segments = computedSegments;
                }
                
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
                
                    const totalTime = performance.now() - totalStartTime;
                    console.log(`⏱️  Dynamic PLY loading total time: ${totalTime.toFixed(2)}ms`);
                    
                    resolve(asset);
                } catch (error) {
                    reject(error);
                }
            }, 50); // 50ms delay to allow PLY parser to finish
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
