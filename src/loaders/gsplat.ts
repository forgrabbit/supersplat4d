import { Asset, AssetRegistry, GSplatData, GSplatResource, PIXELFORMAT_R32F, PIXELFORMAT_RGBA32F } from 'playcanvas';

import { getNextAssetId } from './asset-id-counter';
import { AssetSource } from './asset-source';

const uploadVisibilitySH = (resource: GSplatResource, splatData: GSplatData) => {
    const v0 = splatData.getProp('v_sh_0') as Float32Array | null;
    if (!v0) {
        return false;
    }

    const format = (resource as any).format;
    if (!format || typeof (format as any).addExtraStreams !== 'function') {
        // Older PlayCanvas GSplatFormat does not support extra streams.
        // In this case we skip GPU visibility integration and fall back
        // to standard opacity-only rendering.
        // This keeps loading working even on older engine versions.
        console.warn('GSplat format does not support extra streams; visibility SH will be ignored for this asset.');
        return false;
    }

    format.addExtraStreams([
        { name: 'splatVisibilitySH0', format: PIXELFORMAT_RGBA32F },
        { name: 'splatVisibilitySH1', format: PIXELFORMAT_RGBA32F },
        { name: 'splatVisibilitySH2', format: PIXELFORMAT_RGBA32F },
        { name: 'splatVisibilitySH3', format: PIXELFORMAT_RGBA32F },
        { name: 'splatFrozenOpacity', format: PIXELFORMAT_R32F }
    ]);

    const streams = (resource as any).streams;
    if (!streams?.syncWithFormat) {
        throw new Error('GSplat resource streams cannot sync with format');
    }
    streams.syncWithFormat(format);

    const tex0 = streams.getTexture('splatVisibilitySH0');
    const tex1 = streams.getTexture('splatVisibilitySH1');
    const tex2 = streams.getTexture('splatVisibilitySH2');
    const tex3 = streams.getTexture('splatVisibilitySH3');

    if (!tex0 || !tex1 || !tex2 || !tex3) {
        throw new Error('Visibility SH textures were not created');
    }

    const v: Float32Array[] = [];
    for (let i = 0; i < 16; i++) {
        const arr = splatData.getProp(`v_sh_${i}`) as Float32Array | null;
        if (!arr) {
            throw new Error(`Missing visibility SH property v_sh_${i}`);
        }
        v.push(arr);
    }

    const numSplats = splatData.numSplats;

    const fillTex = (tex: any, baseCoeff: number) => {
        const locked = tex.lock() as unknown as Float32Array | ArrayBufferView;
        const data = locked instanceof Float32Array ? locked : new Float32Array((locked as any).buffer);

        data.fill(0);

        for (let i = 0; i < numSplats; i++) {
            const o = i * 4;
            data[o + 0] = v[baseCoeff + 0][i];
            data[o + 1] = v[baseCoeff + 1][i];
            data[o + 2] = v[baseCoeff + 2][i];
            data[o + 3] = v[baseCoeff + 3][i];
        }

        tex.unlock();
    };

    fillTex(tex0, 0);
    fillTex(tex1, 4);
    fillTex(tex2, 8);
    fillTex(tex3, 12);

    (resource as any).hasVisibilitySH = true;
    return true;
};

// use the engine to load a gsplat asset (ply, compressed.ply, sog, sog-bundle)
const loadGsplat = (assets: AssetRegistry, assetSource: AssetSource) => {
    const totalStartTime = performance.now();
    console.log('🔄 Loading PLY file...');
    const contents = assetSource.contents && (assetSource.contents instanceof Response ? assetSource.contents : new Response(assetSource.contents));

    const file = {
        // we must construct a unique url if contents is provided
        url: contents ? `local-asset-${getNextAssetId()}` : assetSource.url ?? assetSource.filename,
        filename: assetSource.filename,
        contents
    };

    const data = {
        // decompress data on load
        decompress: true,
        // disable morton re-ordering when loading animation frames
        reorder: !(assetSource.animationFrame ?? false)
    };

    const options = {
        mapUrl: assetSource.mapUrl
    };

    return new Promise<Asset>((resolve, reject) => {
        const asset = new Asset(
            assetSource.filename || assetSource.url,
            'gsplat',
            // @ts-ignore
            file,
            data,
            options
        );

        asset.on('load:data', (data: GSplatData) => {
            // support loading 2d splats by adding scale_2 property with almost 0 scale
            if (data instanceof GSplatData && data.getProp('scale_0') && data.getProp('scale_1') && !data.getProp('scale_2')) {
                const scale2 = new Float32Array(data.numSplats).fill(Math.log(1e-6));
                data.addProp('scale_2', scale2);

                // place the new scale_2 property just after scale_1
                const props = data.getElement('vertex').properties;
                props.splice(props.findIndex((prop: any) => prop.name === 'scale_1') + 1, 0, props.splice(props.length - 1, 1)[0]);
            }
        });

        asset.on('load', () => {
            // check the PLY contains minimal set of we expect
            const required = [
                'x', 'y', 'z',
                'scale_0', 'scale_1', 'scale_2',
                'rot_0', 'rot_1', 'rot_2', 'rot_3',
                'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity'
            ];
            const splatData = (asset.resource as GSplatResource).gsplatData as GSplatData;
            const missing = required.filter(x => !splatData.getProp(x));
            if (missing.length > 0) {
                reject(new Error(`This file does not contain gaussian splatting data. The following properties are missing: ${missing.join(', ')}`));
            } else {
                try {
                    uploadVisibilitySH(asset.resource as GSplatResource, splatData);
                } catch (e) {
                    reject(e instanceof Error ? e : new Error(String(e)));
                    return;
                }

                const totalTime = performance.now() - totalStartTime;
                console.log(`⏱️  PLY loading total time: ${totalTime.toFixed(2)}ms`);
                resolve(asset);
            }
        });

        asset.on('error', (err: string) => {
            reject(err);
        });

        assets.add(asset);
        assets.load(asset);
    });
};

export { loadGsplat };
