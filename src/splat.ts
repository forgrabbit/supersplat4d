import {
    ADDRESS_CLAMP_TO_EDGE,
    BLENDEQUATION_ADD,
    BLENDMODE_ONE,
    BLENDMODE_ONE_MINUS_SRC_ALPHA,
    FILTER_NEAREST,
    PIXELFORMAT_R8,
    PIXELFORMAT_R16U,
    PIXELFORMAT_RGBA32F,
    Asset,
    BlendState,
    BoundingBox,
    Color,
    Entity,
    GSplatData,
    GSplatResource,
    Mat4,
    Quat,
    Texture,
    Vec3,
    MeshInstance
} from 'playcanvas';

import { Element, ElementType } from './element';
import { Serializer } from './serializer';
import { vertexShader, fragmentShader, gsplatCenter } from './shaders/splat-shader';
import { State } from './splat-state';
import { Transform } from './transform';
import { TransformPalette } from './transform-palette';
import type { DynManifest } from './loaders/dyn';

const vec = new Vec3();
const veca = new Vec3();
const vecb = new Vec3();

const boundingPoints =
    [-1, 1].map((x) => {
        return [-1, 1].map((y) => {
            return [-1, 1].map((z) => {
                return [
                    new Vec3(x, y, z), new Vec3(x * 0.75, y, z),
                    new Vec3(x, y, z), new Vec3(x, y * 0.75, z),
                    new Vec3(x, y, z), new Vec3(x, y, z * 0.75)
                ];
            });
        });
    }).flat(3);

class Splat extends Element {
    asset: Asset;
    splatData: GSplatData;
    numSplats = 0;
    numDeleted = 0;
    numLocked = 0;
    numSelected = 0;
    entity: Entity;
    changedCounter = 0;
    stateTexture: Texture;
    transformTexture: Texture;
    motionTexture: Texture | null = null;  // For dynamic gaussians: motion_0, motion_1, motion_2
    trbfTexture: Texture | null = null;    // For dynamic gaussians: trbf_center, trbf_scale
    selectionBoundStorage: BoundingBox;
    localBoundStorage: BoundingBox;
    worldBoundStorage: BoundingBox;
    selectionBoundDirty = true;
    localBoundDirty = true;
    worldBoundDirty = true;
    _visible = true;
    transformPalette: TransformPalette;

    selectionAlpha = 1;

    _name = '';
    _tintClr = new Color(1, 1, 1);
    _temperature = 0;
    _saturation = 1;
    _brightness = 0;
    _blackPoint = 0;
    _whitePoint = 1;
    _transparency = 1;

    measurePoints: Vec3[] = [];
    measureSelection = -1;

    rebuildMaterial: (bands: number) => void;

    // Dynamic gaussian support
    isDynamic = false;
    dynManifest: DynManifest | null = null;
    dynBaseUrl = '';
    sog4dSegments: Map<string, ArrayBuffer> | null = null;  // Preloaded segments from SOG4D
    
    // Segment management
    segmentCache = new Map<number, Uint32Array>();
    loadingSegments = new Set<number>();
    currentSegmentIndex = -1;
    activeIndices: Uint32Array | null = null;  // Current segment's active splats
    
    // Frame tracking
    lastSortedFrame = -1;
    lastSortedTime = NaN;
    displayFrame = -1;  // Currently displayed frame (stable playback)
    pendingSort = false;  // Whether a sort is pending
    
    // For initial setup
    currentTime = 0;  // Used during initialization
    
    // Cached dynamic data arrays (for fast center updates)
    _dyn_x0: Float32Array | null = null;
    _dyn_y0: Float32Array | null = null;
    _dyn_z0: Float32Array | null = null;
    _dyn_m0: Float32Array | null = null;
    _dyn_m1: Float32Array | null = null;
    _dyn_m2: Float32Array | null = null;
    _dyn_tc: Float32Array | null = null;

    constructor(asset: Asset, orientation: Vec3) {
        super(ElementType.splat);

        const splatResource = asset.resource as GSplatResource;
        const splatData = splatResource.gsplatData;
        const { device } = splatResource;

        this._name = (asset.file as any).filename;
        this.asset = asset;
        this.splatData = splatData as GSplatData;
        this.numSplats = splatData.numSplats;

        // Check if this is a dynamic gaussian
        const resource = asset.resource as GSplatResource;
        if ((resource as any).dynManifest) {
            this.isDynamic = true;
            this.dynManifest = (resource as any).dynManifest as DynManifest;
            this.dynBaseUrl = (resource as any).dynBaseUrl || '';
            // Check for preloaded SOG4D segments
            if ((resource as any).sog4dSegments) {
                this.sog4dSegments = (resource as any).sog4dSegments as Map<string, ArrayBuffer>;
            }
        }

        this.entity = new Entity('splatEntitiy');
        this.entity.setEulerAngles(orientation);
        this.entity.addComponent('gsplat', { asset });

        // Wait for instance to be created if needed
        let instance = this.entity.gsplat.instance;
        if (!instance) {
            // If instance is not immediately available, it might be created asynchronously
            // Check if gsplat component exists
            if (!this.entity.gsplat) {
                throw new Error('Failed to create gsplat component. Asset may not be properly loaded.');
            }
            // Try to access instance again after a brief delay
            // In most cases, instance should be available immediately if asset is loaded
            throw new Error('Failed to create gsplat instance. Asset may not be properly loaded. Make sure the asset has been loaded before creating Splat.');
        }

        // use custom render order distance calculation for splats
        instance.meshInstance.calculateSortDistance = (meshInstance: MeshInstance, pos: Vec3, dir: Vec3) => {
            const bound = this.localBound;
            const mat = this.entity.getWorldTransform();
            let maxDist;
            for (let i = 0; i < 8; ++i) {
                vec.x = bound.center.x + bound.halfExtents.x * (i & 1 ? 1 : -1);
                vec.y = bound.center.y + bound.halfExtents.y * (i & 2 ? 1 : -1);
                vec.z = bound.center.z + bound.halfExtents.z * (i & 4 ? 1 : -1);
                mat.transformPoint(vec, vec);
                const dist = vec.sub(pos).dot(dir);
                if (i === 0 || dist > maxDist) {
                    maxDist = dist;
                }
            }
            return maxDist;
        };

        // added per-splat state channel
        // bit 1: selected
        // bit 2: deleted
        // bit 3: locked
        if (!this.splatData.getProp('state')) {
            this.splatData.getElement('vertex').properties.push({
                type: 'uchar',
                name: 'state',
                storage: new Uint8Array(this.splatData.numSplats),
                byteSize: 1
            });
        }

        // per-splat transform matrix
        this.splatData.getElement('vertex').properties.push({
            type: 'ushort',
            name: 'transform',
            storage: new Uint16Array(this.splatData.numSplats),
            byteSize: 2
        });

        // Get texture dimensions from resource
        // GSplatResource should have colorTexture after initialization
        const { width, height } = splatResource.colorTexture;

        // pack spherical harmonic data
        const createTexture = (name: string, format: number) => {
            return new Texture(device, {
                name: name,
                width: width,
                height: height,
                format: format,
                mipmaps: false,
                minFilter: FILTER_NEAREST,
                magFilter: FILTER_NEAREST,
                addressU: ADDRESS_CLAMP_TO_EDGE,
                addressV: ADDRESS_CLAMP_TO_EDGE
            });
        };

        // create the state texture
        this.stateTexture = createTexture('splatState', PIXELFORMAT_R8);
        this.transformTexture = createTexture('splatTransform', PIXELFORMAT_R16U);

        // Create dynamic gaussian textures if needed
        if (this.isDynamic) {
            // Motion texture: RGBA = motion_0, motion_1, motion_2, unused
            this.motionTexture = new Texture(device, {
                name: 'splatMotion',
                width: width,
                height: height,
                format: PIXELFORMAT_RGBA32F,
                mipmaps: false,
                minFilter: FILTER_NEAREST,
                magFilter: FILTER_NEAREST,
                addressU: ADDRESS_CLAMP_TO_EDGE,
                addressV: ADDRESS_CLAMP_TO_EDGE
            });

            // TRBF texture: RG = trbf_center, trbf_scale (using RGBA32F, only using RG)
            this.trbfTexture = new Texture(device, {
                name: 'splatTrbf',
                width: width,
                height: height,
                format: PIXELFORMAT_RGBA32F,
                mipmaps: false,
                minFilter: FILTER_NEAREST,
                magFilter: FILTER_NEAREST,
                addressU: ADDRESS_CLAMP_TO_EDGE,
                addressV: ADDRESS_CLAMP_TO_EDGE
            });

            // Upload motion and trbf data to textures
            this.updateDynamicTextures();
        }

        // create the transform palette
        this.transformPalette = new TransformPalette(device);

        // blend mode for splats
        const blendState = new BlendState(true, BLENDEQUATION_ADD, BLENDMODE_ONE, BLENDMODE_ONE_MINUS_SRC_ALPHA);

        this.rebuildMaterial = (bands: number) => {
            const { material } = instance;
            // material.blendState = blendState;
            const { glsl } = material.shaderChunks;
            glsl.set('gsplatVS', vertexShader);
            glsl.set('gsplatPS', fragmentShader);
            glsl.set('gsplatCenterVS', gsplatCenter);

            material.setDefine('SH_BANDS', `${Math.min(bands, (instance.resource as GSplatResource).shBands)}`);
            material.setParameter('splatState', this.stateTexture);
            material.setParameter('splatTransform', this.transformTexture);
            
            // Set dynamic gaussian parameters
            if (this.isDynamic) {
                material.setDefine('DYNAMIC_MODE', true);
                material.setParameter('uIsDynamic', 1.0);
                // Ensure currentTime is initialized
                if (this.currentTime === 0 && this.dynManifest) {
                    this.currentTime = this.dynManifest.start;
                }
                material.setParameter('uCurrentTime', this.currentTime);
                if (this.motionTexture) {
                    material.setParameter('splatMotion', this.motionTexture);
                }
                if (this.trbfTexture) {
                    material.setParameter('splatTrbf', this.trbfTexture);
                }
            } else {
                material.setDefine('DYNAMIC_MODE', false);
                material.setParameter('uIsDynamic', 0.0);
            }
            
            material.update();
        };

        this.selectionBoundStorage = new BoundingBox();
        this.localBoundStorage = instance.resource.aabb;
        // @ts-ignore
        this.worldBoundStorage = instance.meshInstance._aabb;

        // @ts-ignore
        instance.meshInstance._updateAabb = false;

        // when sort changes, re-render the scene and mark sort complete
        instance.sorter.on('updated', () => {
            this.changedCounter++;
            if (this.pendingSort) {
                this.pendingSort = false;
                
                // Now that sorting is complete, update the shader time
                // This ensures rendering uses the same time as sorting
                if (this.isDynamic && this.lastSortedTime === this.lastSortedTime) { // not NaN
                    instance.material.setParameter('uCurrentTime', this.lastSortedTime);
                }
                
                this.scene.forceRender = true;
                this.scene.app.renderNextFrame = true;
            }
        });
        
        // Cache dynamic data arrays for fast center updates
        if (this.isDynamic) {
            this._dyn_x0 = this.splatData.getProp('x') as Float32Array;
            this._dyn_y0 = this.splatData.getProp('y') as Float32Array;
            this._dyn_z0 = this.splatData.getProp('z') as Float32Array;
            this._dyn_m0 = this.splatData.getProp('motion_0') as Float32Array;
            this._dyn_m1 = this.splatData.getProp('motion_1') as Float32Array;
            this._dyn_m2 = this.splatData.getProp('motion_2') as Float32Array;
            this._dyn_tc = this.splatData.getProp('trbf_center') as Float32Array;
        }
    }

    destroy() {
        super.destroy();
        this.entity.destroy();
        this.asset.registry.remove(this.asset);
        this.asset.unload();
        this.segmentCache.clear();
        this.loadingSegments.clear();
        if (this.motionTexture) {
            this.motionTexture.destroy();
        }
        if (this.trbfTexture) {
            this.trbfTexture.destroy();
        }
    }

    /**
     * Update sorter centers with dynamic positions: p(t) = p0 + motion * (t - trbf_center)
     * Only updates active splats (performance optimization)
     * Reference: supersplat_base's _updateCentersActive
     */
    private updateCentersForTime(centers: Float32Array, indices: Uint32Array, t_abs: number): void {
        if (!this._dyn_x0 || !this._dyn_y0 || !this._dyn_z0 || 
            !this._dyn_m0 || !this._dyn_m1 || !this._dyn_m2 || !this._dyn_tc) {
            return;
        }
        
        const x0 = this._dyn_x0;
        const y0 = this._dyn_y0;
        const z0 = this._dyn_z0;
        const m0 = this._dyn_m0;
        const m1 = this._dyn_m1;
        const m2 = this._dyn_m2;
        const tc = this._dyn_tc;
        
        // Manually unroll the hot loop for better performance (4x unroll)
        const n = indices.length;
        let i = 0;
        for (; i <= n - 4; i += 4) {
            let idx = indices[i];
            let dt = t_abs - tc[idx];
            let o = idx * 3;
            centers[o + 0] = x0[idx] + m0[idx] * dt;
            centers[o + 1] = y0[idx] + m1[idx] * dt;
            centers[o + 2] = z0[idx] + m2[idx] * dt;

            idx = indices[i + 1];
            dt = t_abs - tc[idx];
            o = idx * 3;
            centers[o + 0] = x0[idx] + m0[idx] * dt;
            centers[o + 1] = y0[idx] + m1[idx] * dt;
            centers[o + 2] = z0[idx] + m2[idx] * dt;

            idx = indices[i + 2];
            dt = t_abs - tc[idx];
            o = idx * 3;
            centers[o + 0] = x0[idx] + m0[idx] * dt;
            centers[o + 1] = y0[idx] + m1[idx] * dt;
            centers[o + 2] = z0[idx] + m2[idx] * dt;

            idx = indices[i + 3];
            dt = t_abs - tc[idx];
            o = idx * 3;
            centers[o + 0] = x0[idx] + m0[idx] * dt;
            centers[o + 1] = y0[idx] + m1[idx] * dt;
            centers[o + 2] = z0[idx] + m2[idx] * dt;
        }

        // Handle remaining splats
        for (; i < n; i++) {
            const idx = indices[i];
            const dt = t_abs - tc[idx];
            const o = idx * 3;
            centers[o + 0] = x0[idx] + m0[idx] * dt;
            centers[o + 1] = y0[idx] + m1[idx] * dt;
            centers[o + 2] = z0[idx] + m2[idx] * dt;
        }
    }

    // Update motion and trbf textures from GSplatData
    private updateDynamicTextures() {
        if (!this.isDynamic || !this.motionTexture || !this.trbfTexture) {
            return;
        }

        const motion0 = this.splatData.getProp('motion_0') as Float32Array;
        const motion1 = this.splatData.getProp('motion_1') as Float32Array;
        const motion2 = this.splatData.getProp('motion_2') as Float32Array;
        const trbfCenter = this.splatData.getProp('trbf_center') as Float32Array;
        const trbfScale = this.splatData.getProp('trbf_scale') as Float32Array;

        if (!motion0 || !motion1 || !motion2 || !trbfCenter || !trbfScale) {
            return;
        }

        const numSplats = this.splatData.numSplats;
        const { width, height } = this.motionTexture;
        const textureSize = width * height;

        // Pack motion data: RGBA = motion_0, motion_1, motion_2, unused
        const motionData = new Float32Array(textureSize * 4);
        // Pack trbf data: RGBA = trbf_center, trbf_scale, unused, unused
        const trbfData = new Float32Array(textureSize * 4);

        for (let i = 0; i < numSplats && i < textureSize; i++) {
            const idx = i * 4;
            motionData[idx] = motion0[i];
            motionData[idx + 1] = motion1[i];
            motionData[idx + 2] = motion2[i];
            motionData[idx + 3] = 0;

            trbfData[idx] = trbfCenter[i];
            trbfData[idx + 1] = trbfScale[i];
            trbfData[idx + 2] = 0;
            trbfData[idx + 3] = 0;
        }

        // Upload to textures
        const motionLock = this.motionTexture.lock() as Float32Array;
        if (motionLock) {
            motionLock.set(motionData);
            this.motionTexture.unlock();
        }

        const trbfLock = this.trbfTexture.lock() as Float32Array;
        if (trbfLock) {
            trbfLock.set(trbfData);
            this.trbfTexture.unlock();
        }
    }

    // Load a segment's active indices
    private async loadSegment(segmentIndex: number): Promise<Uint32Array | null> {
        if (!this.isDynamic || !this.dynManifest) {
            return null;
        }

        if (this.segmentCache.has(segmentIndex)) {
            // Create a fresh copy from cached ArrayBuffer to avoid detachment issues
            const cached = this.segmentCache.get(segmentIndex)!;
            return new Uint32Array(cached);
        }

        if (this.loadingSegments.has(segmentIndex)) {
            // Wait for existing load to complete
            return new Promise((resolve) => {
                const checkInterval = setInterval(() => {
                    if (this.segmentCache.has(segmentIndex)) {
                        clearInterval(checkInterval);
                        const cached = this.segmentCache.get(segmentIndex)!;
                        resolve(new Uint32Array(cached));
                    } else if (!this.loadingSegments.has(segmentIndex)) {
                        clearInterval(checkInterval);
                        resolve(null);
                    }
                }, 50);
            });
        }

        if (segmentIndex < 0 || segmentIndex >= this.dynManifest.segments.length) {
            return null;
        }

        this.loadingSegments.add(segmentIndex);
        const segment = this.dynManifest.segments[segmentIndex];

        try {
            let arrayBuffer: ArrayBuffer;

            // Check if we have preloaded SOG4D segments
            if (this.sog4dSegments && this.sog4dSegments.has(segment.url)) {
                // Use preloaded data from SOG4D
                arrayBuffer = this.sog4dSegments.get(segment.url)!;
            } else {
                // Fetch from network (for .dyn.json format)
                const segmentUrl = this.dynBaseUrl + segment.url;
                const response = await fetch(segmentUrl);
                if (!response.ok) {
                    throw new Error(`Failed to load segment ${segmentIndex}: ${response.statusText}`);
                }
                arrayBuffer = await response.arrayBuffer();
            }

            const indices = new Uint32Array(arrayBuffer);
            
            // Validate indices are within range
            const maxIndex = this.splatData.numSplats - 1;
            let invalidCount = 0;
            for (let i = 0; i < Math.min(indices.length, 10); i++) {
                if (indices[i] > maxIndex) {
                    invalidCount++;
                }
            }

            // Cache a copy to preserve the original ArrayBuffer
            const cachedCopy = new Uint32Array(indices);
            this.segmentCache.set(segmentIndex, cachedCopy);
            this.loadingSegments.delete(segmentIndex);
            // Return another copy for immediate use
            return new Uint32Array(indices);
        } catch (error) {
            this.loadingSegments.delete(segmentIndex);
            return null;
        }
    }

    // Preload next segment
    private preloadNextSegment(segmentIndex: number) {
        if (!this.isDynamic || !this.dynManifest) {
            return;
        }

        let nextIndex = segmentIndex + 1;
        if (nextIndex >= this.dynManifest.segments.length) {
            // Loop: preload first segment
            nextIndex = 0;
        }

        if (!this.segmentCache.has(nextIndex) && !this.loadingSegments.has(nextIndex)) {
            this.loadSegment(nextIndex);
        }
    }

    // Find segment index for a given absolute time
    private findSegment(absoluteTime: number): number {
        if (!this.isDynamic || !this.dynManifest) {
            return -1;
        }

        for (let i = 0; i < this.dynManifest.segments.length; i++) {
            const segment = this.dynManifest.segments[i];
            const t0 = this.dynManifest.start + segment.t0;
            const t1 = this.dynManifest.start + segment.t1;
            // Use >= for t0 and < for t1 to avoid overlap issues, except for the last segment
            if (i === this.dynManifest.segments.length - 1) {
                // Last segment includes the end time
                if (absoluteTime >= t0 && absoluteTime <= t1) {
                    return i;
                }
            } else {
                if (absoluteTime >= t0 && absoluteTime < t1) {
                    return i;
                }
            }
        }

        // Fallback to first segment
        return 0;
    }


    updateState(changedState = State.selected) {
        const state = this.splatData.getProp('state') as Uint8Array;

        // write state data to gpu texture
        const data = this.stateTexture.lock();
        data.set(state);
        this.stateTexture.unlock();

        let numSelected = 0;
        let numLocked = 0;
        let numDeleted = 0;

        for (let i = 0; i < state.length; ++i) {
            const s = state[i];
            if (s & State.deleted) {
                numDeleted++;
            } else if (s & State.locked) {
                numLocked++;
            } else if (s & State.selected) {
                numSelected++;
            }
        }

        this.numSplats = state.length - numDeleted;
        this.numLocked = numLocked;
        this.numSelected = numSelected;
        this.numDeleted = numDeleted;

        this.makeSelectionBoundDirty();

        // handle splats being added or removed
        if (changedState & State.deleted) {
            this.updateSorting();
        }

        this.scene.forceRender = true;
        this.scene.events.fire('splat.stateChanged', this);
    }

    updatePositions() {
        const data = this.scene.dataProcessor.calcPositions(this);

        // update the splat centers which are used for render-time sorting
        const state = this.splatData.getProp('state') as Uint8Array;
        const { sorter } = this.entity.gsplat.instance;
        const { centers } = sorter;
        for (let i = 0; i < this.splatData.numSplats; ++i) {
            if (state[i] === State.selected) {
                centers[i * 3 + 0] = data[i * 4];
                centers[i * 3 + 1] = data[i * 4 + 1];
                centers[i * 3 + 2] = data[i * 4 + 2];
            }
        }

        this.updateSorting();

        this.scene.forceRender = true;
        this.scene.events.fire('splat.positionsChanged', this);
    }

    updateSorting() {
        const state = this.splatData.getProp('state') as Uint8Array;

        this.makeLocalBoundDirty();

        let mapping;

        // create a sorter mapping to remove deleted splats
        if (this.numSplats !== state.length) {
            mapping = new Uint32Array(this.numSplats);
            let idx = 0;
            for (let i = 0; i < state.length; ++i) {
                if ((state[i] & State.deleted) === 0) {
                    mapping[idx++] = i;
                }
            }
        }

        // update sorting instance
        this.entity.gsplat.instance.sorter.setMapping(mapping);
    }

    get worldTransform() {
        return this.entity.getWorldTransform();
    }

    set name(newName: string) {
        if (newName !== this.name) {
            this._name = newName;
            this.scene.events.fire('splat.name', this);
        }
    }

    get name() {
        return this._name;
    }

    get filename() {
        return (this.asset.file as any).filename;
    }

    calcSplatWorldPosition(splatId: number, result: Vec3) {
        if (splatId >= this.splatData.numSplats) {
            return false;
        }

        // use centers data, which are updated when edits occur
        const { sorter } = this.entity.gsplat.instance;
        const { centers } = sorter;

        result.set(
            centers[splatId * 3 + 0],
            centers[splatId * 3 + 1],
            centers[splatId * 3 + 2]
        );

        this.worldTransform.transformPoint(result, result);

        return true;
    }

    add() {
        // add the entity to the scene
        this.scene.contentRoot.addChild(this.entity);

        this.scene.events.on('view.bands', this.rebuildMaterial, this);
        this.rebuildMaterial(this.scene.events.invoke('view.bands'));

        // we must update state in case the state data was loaded from ply
        this.updateState();

        // Initialize dynamic gaussian: load first segment and set initial time
        if (this.isDynamic && this.dynManifest) {
            // Notify timeline to switch to dynamic mode
            // Use setTimeout to ensure timeline events are registered
            setTimeout(() => {
                this.scene.events.fire('timeline.setDynamic', this.dynManifest!.duration, this.dynManifest!.fps);
                // Register dynamic gaussian control
                this.scene.events.function('scene.hasDynamicGaussian', () => true);
            }, 0);
            
            // Initialize time and load first segment
            const initialRelativeTime = 0;
            const initialFrame = 0;
            const initialFrameTime = this.dynManifest.start + initialFrame / this.dynManifest.fps;
            this.currentTime = initialFrameTime;
            
            // Find initial segment and load it
            const initialSegmentIndex = this.findSegment(initialFrameTime);
            this.currentSegmentIndex = initialSegmentIndex;
            
            // Load initial segment and update mapping
            // Use an empty mapping initially to hide all splats until segment loads
            this.entity.gsplat.instance.sorter.setMapping(new Uint32Array(0));
            
            this.loadSegment(initialSegmentIndex).then((indices) => {
                // Only apply if this is still the current segment
                if (this.currentSegmentIndex !== initialSegmentIndex) {
                    return;
                }
                
                if (indices && indices.length > 0) {
                    // Cache active indices
                    this.activeIndices = indices;
                    
                    // Update centers for initial frame
                    const sorter = this.entity.gsplat.instance.sorter;
                    this.updateCentersForTime(sorter.centers, indices, this.dynManifest!.start);
                    
                    // Mark as pending sort so that uCurrentTime is set when sorting completes
                    this.pendingSort = true;
                    this.lastSortedFrame = 0;
                    this.lastSortedTime = this.dynManifest!.start;
                    
                    // Set mapping to trigger sort
                    sorter.setMapping(indices);
                    
                    this.preloadNextSegment(initialSegmentIndex);
                }
            });
        }
    }

    remove() {
        this.scene.events.off('view.bands', this.rebuildMaterial, this);

        this.scene.contentRoot.removeChild(this.entity);
        this.scene.boundDirty = true;
    }

    serialize(serializer: Serializer) {
        serializer.packa(this.entity.getWorldTransform().data);
        serializer.pack(this.changedCounter);
        serializer.pack(this.visible);
        serializer.pack(this.tintClr.r, this.tintClr.g, this.tintClr.b);
        serializer.pack(this.temperature, this.saturation, this.brightness, this.blackPoint, this.whitePoint, this.transparency);
    }

    /**
     * onUpdate: 动态高斯核心更新流程（每帧调用，即使不渲染）
     * 
     * 核心诉求：
     * 1. 每一帧更新位置 (centers)
     * 2. 排序
     * 3. 渲染
     */
    onUpdate(deltaTime: number) {
        if (!this.isDynamic || !this.dynManifest) {
            return;
        }
        
        const events = this.scene.events;
        
        // 1. 获取当前帧 (从 timeline)
        const currentFrame = (events.invoke('timeline.frame') ?? 0) as number;
        const totalFrames = Math.ceil(this.dynManifest.duration * this.dynManifest.fps);
        const frame = currentFrame % totalFrames;
        
        // 2. 计算该帧的绝对时间
        const t_abs = this.dynManifest.start + (frame / this.dynManifest.fps);
        
        // 3. 检查是否需要更新（帧变了 && 没有正在排序）
        const needsUpdate = frame !== this.lastSortedFrame && !this.pendingSort;
        
        if (needsUpdate) {
            // 4. 找到对应的 segment
            const segmentIdx = this.findSegment(t_abs);
            
            // 5. 检查 segment 是否在缓存中
            if (this.segmentCache.has(segmentIdx)) {
                const indices = new Uint32Array(this.segmentCache.get(segmentIdx)!);
                this.activeIndices = indices;
                this.currentSegmentIndex = segmentIdx;
                
                // 6. 更新 centers: p(t) = p0 + motion * (t - trbf_center)
                const sorter = this.entity.gsplat.instance.sorter;
                this.updateCentersForTime(sorter.centers, indices, t_abs);
                
                // 7. 触发排序 (shader uniform uCurrentTime will be set when sorting completes)
                this.pendingSort = true;
                this.lastSortedFrame = frame;
                this.lastSortedTime = t_abs;
                
                sorter.setMapping(indices);
                
                // 预加载下一个 segment
                this.preloadNextSegment(segmentIdx);
                
            } else if (!this.loadingSegments.has(segmentIdx)) {
                // segment 不在缓存，异步加载
                this.loadSegment(segmentIdx).then(() => {
                    this.scene.forceRender = true;
                    this.scene.app.renderNextFrame = true;
                });
            }
        }
    }

    /**
     * onPreRender: 视觉设置（只在渲染时调用）
     */
    onPreRender() {
        const events = this.scene.events;
        const selected = this.scene.camera.renderOverlays && events.invoke('selection') === this;
        const cameraMode = events.invoke('camera.mode');
        const cameraOverlay = events.invoke('camera.overlay');
        const material = this.entity.gsplat.instance.material;

        // ========== VISUAL SETTINGS ==========
        // configure rings rendering
        material.setParameter('mode', cameraMode === 'rings' ? 1 : 0);
        material.setParameter('ringSize', (selected && cameraOverlay && cameraMode === 'rings') ? 0.04 : 0);

        const selectionAlpha = selected && !events.invoke('view.outlineSelection') ? this.selectionAlpha : 0;

        // configure colors
        const selectedClr = events.invoke('selectedClr');
        const unselectedClr = events.invoke('unselectedClr');
        const lockedClr = events.invoke('lockedClr');
        material.setParameter('selectedClr', [selectedClr.r, selectedClr.g, selectedClr.b, selectedClr.a * selectionAlpha]);
        material.setParameter('unselectedClr', [unselectedClr.r, unselectedClr.g, unselectedClr.b, unselectedClr.a]);
        material.setParameter('lockedClr', [lockedClr.r, lockedClr.g, lockedClr.b, lockedClr.a]);

        // combine black pointer, white point and brightness
        const offset = -this.blackPoint + this.brightness;
        const scale = 1 / (this.whitePoint - this.blackPoint);

        material.setParameter('clrOffset', [offset, offset, offset]);
        material.setParameter('clrScale', [
            scale * this.tintClr.r * (1 + this.temperature),
            scale * this.tintClr.g,
            scale * this.tintClr.b * (1 - this.temperature),
            this.transparency
        ]);

        material.setParameter('saturation', this.saturation);
        material.setParameter('transformPalette', this.transformPalette.texture);

        if (this.visible && selected) {
            // render bounding box
            if (events.invoke('camera.bound')) {
                const bound = this.localBound;
                const scale = new Mat4().setTRS(bound.center, Quat.IDENTITY, bound.halfExtents);
                scale.mul2(this.entity.getWorldTransform(), scale);

                for (let i = 0; i < boundingPoints.length / 2; i++) {
                    const a = boundingPoints[i * 2];
                    const b = boundingPoints[i * 2 + 1];
                    scale.transformPoint(a, veca);
                    scale.transformPoint(b, vecb);

                    this.scene.app.drawLine(veca, vecb, Color.WHITE, true, this.scene.debugLayer);
                }
            }
        }

        this.entity.enabled = this.visible;
    }

    focalPoint() {
        // GSplatData has a function for calculating an weighted average of the splat positions
        // to get a focal point for the camera, but we use bound center instead
        return this.worldBound.center;
    }

    move(position?: Vec3, rotation?: Quat, scale?: Vec3) {
        const entity = this.entity;
        if (position) {
            entity.setLocalPosition(position);
        }
        if (rotation) {
            entity.setLocalRotation(rotation);
        }
        if (scale) {
            entity.setLocalScale(scale);
        }

        this.makeWorldBoundDirty();

        this.scene.events.fire('splat.moved', this);
    }

    makeSelectionBoundDirty() {
        this.selectionBoundDirty = true;
        this.makeLocalBoundDirty();
    }

    makeLocalBoundDirty() {
        this.localBoundDirty = true;
        this.makeWorldBoundDirty();
    }

    makeWorldBoundDirty() {
        this.worldBoundDirty = true;
        this.scene.boundDirty = true;
    }

    // get the selection bound
    get selectionBound() {
        const selectionBound = this.selectionBoundStorage;
        if (this.selectionBoundDirty) {
            this.scene.dataProcessor.calcBound(this, selectionBound, true);
            this.selectionBoundDirty = false;
        }
        return selectionBound;
    }

    // get local space bound
    get localBound() {
        const localBound = this.localBoundStorage;
        if (this.localBoundDirty) {
            this.scene.dataProcessor.calcBound(this, localBound, false);
            this.localBoundDirty = false;
            this.entity.getWorldTransform().transformPoint(localBound.center, vec);
        }
        return localBound;
    }

    // get world space bound
    get worldBound() {
        const worldBound = this.worldBoundStorage;
        if (this.worldBoundDirty) {
            // calculate meshinstance aabb (transformed local bound)
            worldBound.setFromTransformedAabb(this.localBound, this.entity.getWorldTransform());

            // flag scene bound as dirty
            this.worldBoundDirty = false;
        }
        return worldBound;
    }

    set visible(value: boolean) {
        if (value !== this.visible) {
            this._visible = value;
            this.scene.events.fire('splat.visibility', this);
        }
    }

    get visible() {
        return this._visible;
    }

    set tintClr(value: Color) {
        if (!this._tintClr.equals(value)) {
            this._tintClr.set(value.r, value.g, value.b);
            this.scene.events.fire('splat.tintClr', this);
        }
    }

    get tintClr() {
        return this._tintClr;
    }

    set temperature(value: number) {
        if (value !== this._temperature) {
            this._temperature = value;
            this.scene.events.fire('splat.temperature', this);
        }
    }

    get temperature() {
        return this._temperature;
    }

    set saturation(value: number) {
        if (value !== this._saturation) {
            this._saturation = value;
            this.scene.events.fire('splat.saturation', this);
        }
    }

    get saturation() {
        return this._saturation;
    }

    set brightness(value: number) {
        if (value !== this._brightness) {
            this._brightness = value;
            this.scene.events.fire('splat.brightness', this);
        }
    }

    get brightness() {
        return this._brightness;
    }

    set blackPoint(value: number) {
        if (value !== this._blackPoint) {
            this._blackPoint = value;
            this.scene.events.fire('splat.blackPoint', this);
        }
    }

    get blackPoint() {
        return this._blackPoint;
    }

    set whitePoint(value: number) {
        if (value !== this._whitePoint) {
            this._whitePoint = value;
            this.scene.events.fire('splat.whitePoint', this);
        }
    }

    get whitePoint() {
        return this._whitePoint;
    }

    set transparency(value: number) {
        if (value !== this._transparency) {
            this._transparency = value;
            this.scene.events.fire('splat.transparency', this);
        }
    }

    get transparency() {
        return this._transparency;
    }

    getPivot(mode: 'center' | 'boundCenter', selection: boolean, result: Transform) {
        const { entity } = this;
        switch (mode) {
            case 'center':
                result.set(entity.getLocalPosition(), entity.getLocalRotation(), entity.getLocalScale());
                break;
            case 'boundCenter':
                entity.getLocalTransform().transformPoint((selection ? this.selectionBound : this.localBound).center, vec);
                result.set(vec, entity.getLocalRotation(), entity.getLocalScale());
                break;
        }
    }

    docSerialize() {
        const pack3 = (v: Vec3) => [v.x, v.y, v.z];
        const pack4 = (q: Quat) => [q.x, q.y, q.z, q.w];
        const packC = (c: Color) => [c.r, c.g, c.b, c.a];
        return {
            name: this.name,
            position: pack3(this.entity.getLocalPosition()),
            rotation: pack4(this.entity.getLocalRotation()),
            scale: pack3(this.entity.getLocalScale()),
            visible: this.visible,
            tintClr: packC(this.tintClr),
            temperature: this.temperature,
            saturation: this.saturation,
            brightness: this.brightness,
            blackPoint: this.blackPoint,
            whitePoint: this.whitePoint,
            transparency: this.transparency
        };
    }

    docDeserialize(doc: any) {
        const { name, position, rotation, scale, visible, tintClr, temperature, saturation, brightness, blackPoint, whitePoint, transparency } = doc;

        this.name = name;
        this.move(new Vec3(position), new Quat(rotation), new Vec3(scale));
        this.visible = visible;
        this.tintClr = new Color(tintClr[0], tintClr[1], tintClr[2], tintClr[3]);
        this.temperature = temperature ?? 0;
        this.saturation = saturation ?? 1;
        this.brightness = brightness;
        this.blackPoint = blackPoint;
        this.whitePoint = whitePoint;
        this.transparency = transparency;
    }
}

export { Splat };
