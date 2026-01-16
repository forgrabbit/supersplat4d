import { AppBase, Asset, GSplatData, GSplatResource, Vec3 } from 'playcanvas';

import { Events } from './events';
import { AssetSource } from './loaders/asset-source';
import { loadDyn } from './loaders/dyn';
import { checkPlyIsDynamic, loadDynamicPly, DynamicPlyParams } from './loaders/dynamic-ply';
import { loadGsplat } from './loaders/gsplat';
import { loadLcc } from './loaders/lcc';
import { loadSog4d } from './loaders/sog4d';
import { loadSplat } from './loaders/splat';
import { Splat } from './splat';

const defaultOrientation = new Vec3(0, 0, 180);
const lccOrientation = new Vec3(90, 0, 180);

// handles loading gltf container assets
class AssetLoader {
    app: AppBase;
    events: Events;
    defaultAnisotropy: number;
    loadAllData = true;

    constructor(app: AppBase, events: Events, defaultAnisotropy?: number) {
        this.app = app;
        this.events = events;
        this.defaultAnisotropy = defaultAnisotropy || 1;
    }

    async load(assetSource: AssetSource) {
        const wrap = (gsplatData: GSplatData) => {
            const asset = new Asset(assetSource.filename || assetSource.url, 'gsplat', {
                url: assetSource.contents ? `local-asset-${Date.now()}` : assetSource.url ?? assetSource.filename,
                filename: assetSource.filename
            });
            this.app.assets.add(asset);
            asset.resource = new GSplatResource(this.app.graphicsDevice, gsplatData);
            return asset;
        };

        if (!assetSource.animationFrame) {
            this.events.fire('startSpinner');
        }

        try {
            const filename = (assetSource.filename || assetSource.url).toLowerCase();

            let asset;
            let orientation = defaultOrientation;

            if (filename.endsWith('.splat')) {
                asset = wrap(await loadSplat(assetSource));
            } else if (filename.endsWith('.lcc')) {
                asset = wrap(await loadLcc(assetSource));
                orientation = lccOrientation;
            } else if (filename.endsWith('.dyn.json')) {
                asset = await loadDyn(this.app.assets, assetSource, this.app.graphicsDevice);
            } else if (filename.endsWith('.sog4d')) {
                asset = await loadSog4d(this.app.assets, assetSource, this.app.graphicsDevice, this.events);
            } else if (filename.endsWith('.ply')) {
                // Check if PLY is dynamic (has trbf_center, trbf_scale, motion_*)
                const { isDynamic, cfgArgs } = await checkPlyIsDynamic(assetSource);
                
                if (isDynamic) {
                    let params: DynamicPlyParams | null = cfgArgs;
                    
                    // If no cfg_args in PLY header, ask user for parameters
                    if (!params) {
                        // Hide spinner while dialog is shown
                        this.events.fire('stopSpinner');
                        params = await this.events.invoke('showDynamicParamsDialog', assetSource.filename || 'unknown.ply');
                        this.events.fire('startSpinner');
                    }
                    
                    if (params) {
                        console.log('📊 Loading dynamic PLY with params:', params);
                        asset = await loadDynamicPly(this.app.assets, assetSource, params);
                    } else {
                        // User cancelled, load as static
                        console.log('⚠️ User cancelled dynamic params dialog, loading as static PLY');
                        asset = await loadGsplat(this.app.assets, assetSource);
                    }
                } else {
                    // Not dynamic, load as regular PLY
                    asset = await loadGsplat(this.app.assets, assetSource);
                }
            } else {
                asset = await loadGsplat(this.app.assets, assetSource);
            }

            return new Splat(asset, orientation);
        } finally {
            if (!assetSource.animationFrame) {
                this.events.fire('stopSpinner');
            }
        }
    }
}

export { AssetLoader };
