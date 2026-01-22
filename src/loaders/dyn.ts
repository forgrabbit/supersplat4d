import { Asset, AssetRegistry, GSplatData, GSplatResource } from 'playcanvas';

import { AssetSource, createReadSource } from './asset-source';
import { getNextAssetId } from './asset-id-counter';

interface DynManifest {
    version: number;
    type: string;
    start: number;
    duration: number;
    fps: number;
    sh_degree: number;
    global: {
        url: string;
        numSplats: number;
    };
    segments: Array<{
        t0: number;
        t1: number;
        url: string;
        count: number;
    }>;
}

// Parse global.bin binary format
// Header: MAGIC (4 bytes) + VERSION (4) + num (4) + sh_degree (4) + reserved (4)
// Then fields in order: x, y, z, scale_0-2, rot_0-3, f_dc_0-2, opacity_logit, motion_0-2, trbf_center, trbf_scale, sh_rest (if sh_degree > 0)
const parseGlobalBin = async (data: ArrayBuffer, manifest: DynManifest): Promise<GSplatData> => {
    const view = new DataView(data);
    let offset = 0;

    // Read header
    const magic = String.fromCharCode(view.getUint8(offset), view.getUint8(offset + 1), view.getUint8(offset + 2), view.getUint8(offset + 3));
    if (magic !== 'DYGS') {
        throw new Error(`Invalid global.bin magic: ${magic}`);
    }
    offset += 4;

    const version = view.getUint32(offset, true);
    offset += 4;
    const numSplats = view.getUint32(offset, true);
    offset += 4;
    const shDegree = view.getUint32(offset, true);
    offset += 4;
    const reserved = view.getUint32(offset, true);
    offset += 4;

    if (version !== 1) {
        throw new Error(`Unsupported global.bin version: ${version}`);
    }

    if (numSplats !== manifest.global.numSplats) {
        throw new Error(`Mismatch in numSplats: header=${numSplats}, manifest=${manifest.global.numSplats}`);
    }

    // Field order as defined in ply_to_dyn_assets.py
    const fieldOrder = ['x', 'y', 'z', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity_logit', 'motion_0', 'motion_1', 'motion_2', 'trbf_center', 'trbf_scale'];
    const fields: Record<string, Float32Array> = {};

    // Read each field (each is numSplats * 4 bytes)
    for (const fieldName of fieldOrder) {
        const fieldData = new Float32Array(data, offset, numSplats);
        fields[fieldName] = fieldData;
        offset += numSplats * 4;
    }

    // Read sh_rest if sh_degree > 0
    if (shDegree > 0) {
        const shCoeffs = (shDegree + 1) ** 2;
        const rest = shCoeffs - 1;
        const shRestSize = numSplats * rest * 3 * 4;
        const shRestData = new Float32Array(data, offset, numSplats * rest * 3);
        fields['sh_rest'] = shRestData;
        offset += shRestSize;
    }

    // Convert opacity_logit to opacity (sigmoid inverse: logit)
    // In the shader we'll use sigmoid, but GSplatData expects opacity in [0,1] range
    // Actually, we should keep it as logit for consistency with the format
    // But GSplatData might expect opacity, let's check what format it uses
    // For now, we'll keep opacity_logit and add a computed opacity if needed

    // Build GSplatData
    const properties: any[] = [];

    // Standard properties
    properties.push({ type: 'float', name: 'x', storage: fields['x'], byteSize: 4 });
    properties.push({ type: 'float', name: 'y', storage: fields['y'], byteSize: 4 });
    properties.push({ type: 'float', name: 'z', storage: fields['z'], byteSize: 4 });
    properties.push({ type: 'float', name: 'scale_0', storage: fields['scale_0'], byteSize: 4 });
    properties.push({ type: 'float', name: 'scale_1', storage: fields['scale_1'], byteSize: 4 });
    properties.push({ type: 'float', name: 'scale_2', storage: fields['scale_2'], byteSize: 4 });
    properties.push({ type: 'float', name: 'rot_0', storage: fields['rot_0'], byteSize: 4 });
    properties.push({ type: 'float', name: 'rot_1', storage: fields['rot_1'], byteSize: 4 });
    properties.push({ type: 'float', name: 'rot_2', storage: fields['rot_2'], byteSize: 4 });
    properties.push({ type: 'float', name: 'rot_3', storage: fields['rot_3'], byteSize: 4 });
    properties.push({ type: 'float', name: 'f_dc_0', storage: fields['f_dc_0'], byteSize: 4 });
    properties.push({ type: 'float', name: 'f_dc_1', storage: fields['f_dc_1'], byteSize: 4 });
    properties.push({ type: 'float', name: 'f_dc_2', storage: fields['f_dc_2'], byteSize: 4 });

    // Opacity: store as logit value directly
    // PlayCanvas GSplatResource will apply sigmoid when creating the color texture
    // So we should NOT do sigmoid here to avoid double sigmoid!
    const opacityLogit = fields['opacity_logit'];
    properties.push({ type: 'float', name: 'opacity', storage: opacityLogit, byteSize: 4 });

    // Dynamic properties
    properties.push({ type: 'float', name: 'motion_0', storage: fields['motion_0'], byteSize: 4 });
    properties.push({ type: 'float', name: 'motion_1', storage: fields['motion_1'], byteSize: 4 });
    properties.push({ type: 'float', name: 'motion_2', storage: fields['motion_2'], byteSize: 4 });
    properties.push({ type: 'float', name: 'trbf_center', storage: fields['trbf_center'], byteSize: 4 });
    properties.push({ type: 'float', name: 'trbf_scale', storage: fields['trbf_scale'], byteSize: 4 });
    
    // Diagnostic: check trbf_center range
    const trbfCenters = fields['trbf_center'];
    let minTrbf = Infinity, maxTrbf = -Infinity;
    for (let i = 0; i < numSplats; i++) {
        minTrbf = Math.min(minTrbf, trbfCenters[i]);
        maxTrbf = Math.max(maxTrbf, trbfCenters[i]);
    }
    console.log(`📊 TRBF center range: [${minTrbf.toFixed(3)}, ${maxTrbf.toFixed(3)}]`);
    console.log(`📊 Manifest: start=${manifest.start.toFixed(3)}, duration=${manifest.duration.toFixed(3)}, fps=${manifest.fps}`);

    // SH rest coefficients
    if (shDegree > 0 && fields['sh_rest']) {
        properties.push({ type: 'float', name: 'sh_rest', storage: fields['sh_rest'], byteSize: 4 });
    }

    const gsplatData = new GSplatData([{
        name: 'vertex',
        count: numSplats,
        properties
    }]);

    // Store manifest and dynamic config in the resource for later use
    (gsplatData as any).dynManifest = manifest;

    return gsplatData;
};

// Load dynamic gaussian splatting assets
const loadDyn = async (assets: AssetRegistry, assetSource: AssetSource, device: any): Promise<Asset> => {
    // First, load and parse the manifest
    const manifestSource = await createReadSource(assetSource);
    const manifestBuffer = await manifestSource.arrayBuffer();
    const manifestText = new TextDecoder().decode(manifestBuffer);
    const manifest: DynManifest = JSON.parse(manifestText);

    if (manifest.type !== 'dyn') {
        throw new Error(`Expected type 'dyn', got '${manifest.type}'`);
    }

    // Determine base URL from manifest file location
    const manifestUrl = assetSource.url ?? assetSource.filename ?? '';
    let baseUrl = '';
    if (manifestUrl) {
        const lastSlash = manifestUrl.lastIndexOf('/');
        if (lastSlash >= 0) {
            baseUrl = manifestUrl.substring(0, lastSlash + 1);
        } else {
            // If no slash, assume current directory
            baseUrl = './';
        }
    } else {
        // If no URL, assume current directory
        baseUrl = './';
    }

    // Load global.bin
    const globalBinUrl = baseUrl + manifest.global.url;
    const globalBinResponse = await fetch(globalBinUrl);
    if (!globalBinResponse.ok) {
        throw new Error(`Failed to load global.bin: ${globalBinResponse.statusText}`);
    }
    const globalBinData = await globalBinResponse.arrayBuffer();

    // Parse global.bin
    const gsplatData = await parseGlobalBin(globalBinData, manifest);

    // Create asset
    const file = {
        url: assetSource.contents ? `local-asset-${getNextAssetId()}` : manifestUrl,
        filename: assetSource.filename,
        contents: assetSource.contents
    };

    return new Promise<Asset>((resolve, reject) => {
        const asset = new Asset(
            assetSource.filename || assetSource.url || 'dynamic-splat',
            'gsplat',
            // @ts-ignore
            file
        );

        // Validate required properties first
        const required = [
            'x', 'y', 'z',
            'scale_0', 'scale_1', 'scale_2',
            'rot_0', 'rot_1', 'rot_2', 'rot_3',
            'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
            'motion_0', 'motion_1', 'motion_2',
            'trbf_center', 'trbf_scale'
        ];
        const missing = required.filter(x => !gsplatData.getProp(x));
        if (missing.length > 0) {
            reject(new Error(`Dynamic gaussian splatting data is missing required properties: ${missing.join(', ')}`));
            return;
        }

        // Create resource and store dynamic metadata
        const resource = new GSplatResource(device, gsplatData);
        (resource as any).dynManifest = manifest;
        (resource as any).dynBaseUrl = baseUrl;
        
        asset.resource = resource;

        // Add asset to registry
        assets.add(asset);
        
        // Trigger load:data event to ensure GSplatResource initializes textures
        // This is similar to what loadGsplat does
        asset.fire('load:data', gsplatData);
        
        // Mark asset as loaded and trigger load event
        // We don't call assets.load() because:
        // 1. The resource is already created and assigned
        // 2. assets.load() might trigger SOG loader if URL ends with .json
        // 3. We manually trigger the load event to complete the loading process
        (asset as any)._loaded = true;
        (asset as any)._loading = false;
        
        // Use setTimeout to ensure event handlers are registered first
        setTimeout(() => {
            asset.fire('load', asset);
            resolve(asset);
        }, 0);
    });
};

export { loadDyn };
export type { DynManifest };

