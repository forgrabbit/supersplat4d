import {
    ADDRESS_CLAMP_TO_EDGE,
    BLENDEQUATION_ADD,
    BLENDMODE_ONE,
    BLENDMODE_ONE_MINUS_SRC_ALPHA,
    FILTER_NEAREST,
    PIXELFORMAT_R8,
    PIXELFORMAT_R32F,
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
import type { DynManifest } from './loaders/dyn';
import { profiler } from './profiling';
import { Serializer } from './serializer';
import { vertexShader, fragmentShader, gsplatCenter } from './shaders/splat-shader';
import { State } from './splat-state';
import { Transform } from './transform';
import { TransformPalette } from './transform-palette';
import { readVisibilityData, type VisibilityData, type VisibilitySvData } from './visibility-data';

const vec = new Vec3();
const veca = new Vec3();
const vecb = new Vec3();
const vecc = new Vec3();
const mat = new Mat4();

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

const PRESORT_CULL_THRESHOLD_SCALE = 0.98;
const PRESORT_CAMERA_EPSILON = 1e-5;
const PRESORT_PROFILE_SAMPLE_RATE = 64;
const PRESORT_PROFILE_SAMPLE_MASK = PRESORT_PROFILE_SAMPLE_RATE - 1;

const estimateSampledMs = (sampleMs: number, samples: number, population: number) => {
    return samples > 0 && population > 0 ? sampleMs * population / samples : 0;
};

type VisibilitySvCpuCache = {
    siteX: Float32Array[];
    siteY: Float32Array[];
    siteZ: Float32Array[];
    tauSoftplus: Float32Array[];
};

type PreSortFilterResult = {
    mapping: Uint32Array | null;
    sourceCount: number;
    keptCount: number;
    deletedRejected: number;
    baseOpacityRejected: number;
    opacityRejected: number;
    dynamicOpacityTested: number;
    dynamicOpacityRejected: number;
    dynamicCenterUpdated: number;
    frozenOpacityTested: number;
    frozenOpacityRejected: number;
    visibilityTested: number;
    visibilityRejected: number;
    setupMs: number;
    loopMs: number;
    finalizeMs: number;
    dynamicOpacitySamples: number;
    dynamicOpacitySampleMs: number;
    dynamicOpacityEstimatedMs: number;
    dynamicCenterSamples: number;
    dynamicCenterSampleMs: number;
    dynamicCenterEstimatedMs: number;
    frozenOpacitySamples: number;
    frozenOpacitySampleMs: number;
    frozenOpacityEstimatedMs: number;
    visibilitySamples: number;
    visibilityDirectionSampleMs: number;
    visibilityDirectionEstimatedMs: number;
    visibilityEvalSampleMs: number;
    visibilityEvalEstimatedMs: number;
    visibilityApplySampleMs: number;
    visibilityApplyEstimatedMs: number;
    visibilityEstimatedMs: number;
};

type PreSortSource = {
    ready: boolean;
    frame: number;
    tAbs: number;
    segmentIdx: number;
    indices: Uint32Array | null;
};

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

    hasVisibility = false;
    visibilityMode: VisibilityData['mode'] = 'none';
    visibilityData: VisibilityData = { mode: 'none' };
    visibilityNumLobes = 0;
    visibilityShTextures: Texture[] = [];
    visibilitySVSiteValueTexture: Texture | null = null;
    visibilitySVTauTexture: Texture | null = null;
    visibilityFrozenOpacityTexture: Texture | null = null;
    visibilitySVPackAxis = 0;
    visibilitySVLobeStride = 0;
    /** Effective alpha threshold after visibility (and dynamic temporal) modulation; from PLY cfg_args or 0.005. */
    visibilityCullThreshold = 0.005;
    private _freezeOpacityHandler: ((enabled: boolean) => void) | null = null;

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
    lastDrawSplats = 0;
    lastActiveSplats = 0;

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
    _dyn_ts: Float32Array | null = null;

    private _baseOpacity: Float32Array | null = null;
    private _visibilitySvCpuCache: VisibilitySvCpuCache | null = null;
    private _preSortScratch: Uint32Array | null = null;
    private _frozenEffectiveOpacity: Float32Array | null = null;
    private _visibilityLogitScratch: number[] = [];
    private _preSortFilterRevision = 0;
    private _lastPreSortRevision = -1;
    private _lastPreSortFrame = -1;
    private _lastPreSortSegment = -2;
    private _lastPreSortTime = NaN;
    private _lastPreSortThreshold = NaN;
    private _lastPreSortFrozen = false;
    private _lastPreSortRenderCount = -1;
    private _lastPreSortCamera = new Vec3(NaN, NaN, NaN);

    constructor(asset: Asset, orientation: Vec3) {
        super(ElementType.splat);
        const initStartTime = performance.now();

        const splatResource = asset.resource as GSplatResource;
        const splatData = splatResource.gsplatData;
        const { device } = splatResource;

        this._name = (asset.file as any).filename;
        this.asset = asset;
        this.splatData = splatData as GSplatData;
        this.numSplats = splatData.numSplats;

        // Check if this is a dynamic gaussian
        const resource = asset.resource as GSplatResource;
        this.visibilityData = readVisibilityData(this.splatData);
        this.visibilityMode = this.visibilityData.mode;
        this.hasVisibility = this.visibilityMode !== 'none';
        this.visibilityNumLobes = this.visibilityData.mode === 'sv' ? this.visibilityData.numLobes : 0;
        this.visibilityCullThreshold = (resource as any).visibilityCullThreshold ?? 0.005;
        if ((resource as any).dynManifest) {
            this.isDynamic = true;
            this.dynManifest = (resource as any).dynManifest as DynManifest;
            this.dynBaseUrl = (resource as any).dynBaseUrl || '';
            // Check for preloaded SOG4D segments
            if ((resource as any).sog4dSegments) {
                this.sog4dSegments = (resource as any).sog4dSegments as Map<string, ArrayBuffer>;
            }
        }
        this.lastActiveSplats = this.isDynamic ? 0 : this.numSplats;
        this.lastDrawSplats = this.isDynamic ? 0 : this.numSplats;

        this.entity = new Entity('splatEntitiy');
        this.entity.setEulerAngles(orientation);
        this.entity.addComponent('gsplat', { asset });

        // Wait for instance to be created if needed
        const instance = this.entity.gsplat.instance;
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

        const originalSort = instance.sort.bind(instance);
        instance.sort = (cameraNode: Entity) => {
            this.applyPreSortFilter(cameraNode);
            originalSort(cameraNode);
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
        const splatColor = splatResource.getTexture('splatColor');
        if (!splatColor) {
            throw new Error('GSplat resource is missing splatColor texture');
        }
        const { width, height } = splatColor;

        const createTexture = (name: string, format: number, texWidth = width, texHeight = height) => {
            return new Texture(device, {
                name: name,
                width: texWidth,
                height: texHeight,
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

        if (this.hasVisibility) {
            this.visibilityFrozenOpacityTexture = createTexture('splatFrozenOpacity', PIXELFORMAT_R32F);
            const visibilityData = this.visibilityData;

            if (visibilityData.mode === 'sh') {
                this.visibilityShTextures = [
                    createTexture('splatVisibilitySH0', PIXELFORMAT_RGBA32F),
                    createTexture('splatVisibilitySH1', PIXELFORMAT_RGBA32F),
                    createTexture('splatVisibilitySH2', PIXELFORMAT_RGBA32F),
                    createTexture('splatVisibilitySH3', PIXELFORMAT_RGBA32F)
                ];

                const visibilitySh = visibilityData.coeffs;
                this.visibilityShTextures.forEach((texture, textureIndex) => {
                    const locked = texture.lock() as unknown as Float32Array | ArrayBufferView;
                    const data = locked instanceof Float32Array ? locked : new Float32Array((locked as any).buffer);
                    data.fill(0);

                    for (let i = 0; i < this.numSplats; i++) {
                        const o = i * 4;
                        const coeffBase = textureIndex * 4;
                        data[o + 0] = visibilitySh[coeffBase + 0][i];
                        data[o + 1] = visibilitySh[coeffBase + 1][i];
                        data[o + 2] = visibilitySh[coeffBase + 2][i];
                        data[o + 3] = visibilitySh[coeffBase + 3][i];
                    }

                    texture.unlock();
                });
            } else if (visibilityData.mode === 'sv') {
                const maxTextureSize = device.maxTextureSize;
                const svStrideVertical = height * this.visibilityNumLobes;
                const svStrideHorizontal = width * this.visibilityNumLobes;
                let packedWidth = width;
                let packedHeight = svStrideVertical;

                if (packedHeight <= maxTextureSize) {
                    this.visibilitySVPackAxis = 0;
                    this.visibilitySVLobeStride = height;
                } else if (svStrideHorizontal <= maxTextureSize) {
                    this.visibilitySVPackAxis = 1;
                    this.visibilitySVLobeStride = width;
                    packedWidth = svStrideHorizontal;
                    packedHeight = height;
                } else {
                    throw new Error(`SV visibility texture packing exceeds max texture size (${maxTextureSize})`);
                }

                const visibilitySVSiteValueTexture = createTexture('splatVisibilitySVSiteValue', PIXELFORMAT_RGBA32F, packedWidth, packedHeight);
                const visibilitySVTauTexture = createTexture('splatVisibilitySVTau', PIXELFORMAT_R32F, packedWidth, packedHeight);
                this.visibilitySVSiteValueTexture = visibilitySVSiteValueTexture;
                this.visibilitySVTauTexture = visibilitySVTauTexture;

                const siteValueLocked = visibilitySVSiteValueTexture.lock() as unknown as Float32Array | ArrayBufferView;
                const siteValueData = siteValueLocked instanceof Float32Array ? siteValueLocked : new Float32Array((siteValueLocked as any).buffer);
                const tauLocked = visibilitySVTauTexture.lock() as unknown as Float32Array | ArrayBufferView;
                const tauData = tauLocked instanceof Float32Array ? tauLocked : new Float32Array((tauLocked as any).buffer);
                siteValueData.fill(0);
                tauData.fill(0);

                for (let i = 0; i < this.numSplats; i++) {
                    const baseX = i % width;
                    const baseY = Math.floor(i / width);

                    for (let lobeIndex = 0; lobeIndex < visibilityData.numLobes; lobeIndex++) {
                        const lobe = visibilityData.lobes[lobeIndex];
                        const siteX = lobe.siteX[i];
                        const siteY = lobe.siteY[i];
                        const siteZ = lobe.siteZ[i];
                        const siteLen = Math.max(1e-6, Math.sqrt(siteX * siteX + siteY * siteY + siteZ * siteZ));
                        const packedX = this.visibilitySVPackAxis === 0 ? baseX : baseX + lobeIndex * width;
                        const packedY = this.visibilitySVPackAxis === 0 ? baseY + lobeIndex * height : baseY;
                        const packedIndex = packedY * packedWidth + packedX;
                        const siteValueOffset = packedIndex * 4;

                        siteValueData[siteValueOffset + 0] = siteX / siteLen;
                        siteValueData[siteValueOffset + 1] = siteY / siteLen;
                        siteValueData[siteValueOffset + 2] = siteZ / siteLen;
                        siteValueData[siteValueOffset + 3] = lobe.value[i];
                        tauData[packedIndex] = lobe.tau[i];
                    }
                }

                visibilitySVSiteValueTexture.unlock();
                visibilitySVTauTexture.unlock();
            }
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
            material.setDefine('HAS_VISIBILITY', this.hasVisibility);
            material.setDefine('HAS_VISIBILITY_SH', this.visibilityMode === 'sh');
            material.setDefine('HAS_VISIBILITY_SV', this.visibilityMode === 'sv');
            material.setDefine('VISIBILITY_SV_LOBES', this.visibilityMode === 'sv' ? `${this.visibilityNumLobes}` : '0');
            if (this.hasVisibility && this.scene) {
                material.setDefine('FROZEN_OPACITY', !!this.scene.events.invoke('visibility.freezeEffectiveOpacity'));
            } else {
                material.setDefine('FROZEN_OPACITY', false);
            }
            material.setParameter('uVisibilityCullThreshold', this.visibilityCullThreshold);
            if (this.hasVisibility) {
                if (this.visibilityFrozenOpacityTexture) {
                    material.setParameter('splatFrozenOpacity', this.visibilityFrozenOpacityTexture);
                }
                if (this.visibilityMode === 'sh') {
                    this.visibilityShTextures.forEach((texture, textureIndex) => {
                        material.setParameter(`splatVisibilitySH${textureIndex}`, texture);
                    });
                } else if (this.visibilityMode === 'sv') {
                    if (this.visibilitySVSiteValueTexture) {
                        material.setParameter('splatVisibilitySVSiteValue', this.visibilitySVSiteValueTexture);
                    }
                    if (this.visibilitySVTauTexture) {
                        material.setParameter('splatVisibilitySVTau', this.visibilitySVTauTexture);
                    }
                    material.setParameter('uVisibilitySVLobeStride', this.visibilitySVLobeStride);
                    material.setParameter('uVisibilitySVPackAxis', this.visibilitySVPackAxis);
                }
            }
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
        instance.sorter.on('updated', (count?: number, sortDetails?: Record<string, unknown>) => {
            if (typeof count === 'number') {
                this.lastDrawSplats = count;
                profiler.addFrameValue('drawSplatsUpdated', count);
            }
            profiler.event('splat.sort.updated', {
                splat: this.name,
                drawSplats: typeof count === 'number' ? count : undefined,
                activeSplats: this.lastActiveSplats,
                segmentIndex: this.currentSegmentIndex,
                ...(sortDetails ?? {})
            });
            this.changedCounter++;
            if (this.pendingSort) {
                this.pendingSort = false;

                // Now that sorting is complete, update the shader time
                // This ensures rendering uses the same time as sorting
                if (this.isDynamic && !Number.isNaN(this.lastSortedTime)) {
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
            this._dyn_ts = this.splatData.getProp('trbf_scale') as Float32Array;
        }

        const initTime = performance.now() - initStartTime;
        console.log(`⏱️  Splat constructor initialization: ${initTime.toFixed(2)}ms`);
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
        this.visibilityShTextures.forEach(texture => texture.destroy());
        if (this.visibilitySVSiteValueTexture) {
            this.visibilitySVSiteValueTexture.destroy();
        }
        if (this.visibilitySVTauTexture) {
            this.visibilitySVTauTexture.destroy();
        }
        if (this.visibilityFrozenOpacityTexture) {
            this.visibilityFrozenOpacityTexture.destroy();
        }
    }

    // Update motion and trbf textures from GSplatData
    private updateDynamicTextures() {
        if (!this.isDynamic || !this.motionTexture || !this.trbfTexture) {
            return;
        }

        const totalStart = profiler.now();
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

        const packStart = profiler.now();
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
        const packMs = profiler.now() - packStart;

        // Upload to textures
        let motionUploadMs = 0;
        let trbfUploadMs = 0;
        const motionLock = this.motionTexture.lock() as Float32Array;
        if (motionLock) {
            motionLock.set(motionData);
            const motionUploadStart = profiler.now();
            this.motionTexture.unlock();
            motionUploadMs = profiler.now() - motionUploadStart;
        }

        const trbfLock = this.trbfTexture.lock() as Float32Array;
        if (trbfLock) {
            trbfLock.set(trbfData);
            const trbfUploadStart = profiler.now();
            this.trbfTexture.unlock();
            trbfUploadMs = profiler.now() - trbfUploadStart;
        }

        const motionTexBytes = (this.motionTexture as Texture & { gpuSize?: number }).gpuSize ?? 0;
        const trbfTexBytes = (this.trbfTexture as Texture & { gpuSize?: number }).gpuSize ?? 0;
        profiler.setFrameValues({
            motionTexBytes,
            trbfTexBytes,
            dynamicTexBytes: motionTexBytes + trbfTexBytes
        });
        profiler.event('dynamicTexture.upload', {
            splat: this.name,
            numSplats,
            width,
            height,
            texturePixels: textureSize,
            packMs,
            motionUploadMs,
            trbfUploadMs,
            totalMs: profiler.now() - totalStart,
            motionTexBytes,
            trbfTexBytes
        });
    }

    // Load a segment's active indices
    private async loadSegment(segmentIndex: number): Promise<Uint32Array | null> {
        const loadStart = profiler.now();
        if (!this.isDynamic || !this.dynManifest) {
            return null;
        }

        if (this.segmentCache.has(segmentIndex)) {
            // Create a fresh copy from cached ArrayBuffer to avoid detachment issues
            const copyStart = profiler.now();
            const cached = this.segmentCache.get(segmentIndex)!;
            const result = new Uint32Array(cached);
            profiler.event('segment.load', {
                splat: this.name,
                segmentIndex,
                source: 'cache',
                activeSplats: result.length,
                bytes: result.byteLength,
                copyMs: profiler.now() - copyStart,
                totalMs: profiler.now() - loadStart
            });
            return result;
        }

        if (this.loadingSegments.has(segmentIndex)) {
            // Wait for existing load to complete
            return new Promise((resolve) => {
                const checkInterval = setInterval(() => {
                    if (this.segmentCache.has(segmentIndex)) {
                        clearInterval(checkInterval);
                        const copyStart = profiler.now();
                        const cached = this.segmentCache.get(segmentIndex)!;
                        const result = new Uint32Array(cached);
                        profiler.event('segment.load', {
                            splat: this.name,
                            segmentIndex,
                            source: 'pending-wait',
                            activeSplats: result.length,
                            bytes: result.byteLength,
                            copyMs: profiler.now() - copyStart,
                            totalMs: profiler.now() - loadStart
                        });
                        resolve(result);
                    } else if (!this.loadingSegments.has(segmentIndex)) {
                        clearInterval(checkInterval);
                        profiler.event('segment.load', {
                            splat: this.name,
                            segmentIndex,
                            source: 'pending-wait',
                            failed: true,
                            totalMs: profiler.now() - loadStart
                        });
                        resolve(null);
                    }
                }, 50);
            });
        }

        if (segmentIndex < 0 || segmentIndex >= this.dynManifest.segments.length) {
            profiler.event('segment.load', {
                splat: this.name,
                segmentIndex,
                source: 'invalid',
                failed: true,
                totalMs: profiler.now() - loadStart
            });
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
                const fetchStart = profiler.now();
                const segmentUrl = this.dynBaseUrl + segment.url;
                const response = await fetch(segmentUrl);
                if (!response.ok) {
                    throw new Error(`Failed to load segment ${segmentIndex}: ${response.statusText}`);
                }
                arrayBuffer = await response.arrayBuffer();
                profiler.event('segment.fetch', {
                    splat: this.name,
                    segmentIndex,
                    url: segment.url,
                    bytes: arrayBuffer.byteLength,
                    ms: profiler.now() - fetchStart
                });
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
            const result = new Uint32Array(indices);
            profiler.event('segment.load', {
                splat: this.name,
                segmentIndex,
                source: this.sog4dSegments && this.sog4dSegments.has(segment.url) ? 'preloaded' : 'network',
                activeSplats: result.length,
                bytes: result.byteLength,
                totalMs: profiler.now() - loadStart
            });
            return result;
        } catch (error) {
            this.loadingSegments.delete(segmentIndex);
            profiler.event('segment.load', {
                splat: this.name,
                segmentIndex,
                source: 'error',
                failed: true,
                message: error instanceof Error ? error.message : String(error),
                totalMs: profiler.now() - loadStart
            });
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

    private markPreSortFilterDirty() {
        this._preSortFilterRevision++;
    }

    private needsPreSortOpacityFilter() {
        return this.isDynamic || this.hasVisibility;
    }

    private preSortCullThreshold() {
        return Math.max(0, this.visibilityCullThreshold * PRESORT_CULL_THRESHOLD_SCALE);
    }

    private getBaseOpacity() {
        const opacity = this.splatData.getProp('opacity') as Float32Array | null;
        if (!opacity) {
            return null;
        }

        if (this._baseOpacity && this._baseOpacity.length === opacity.length) {
            return this._baseOpacity;
        }

        const baseOpacity = new Float32Array(opacity.length);
        for (let i = 0; i < opacity.length; i++) {
            baseOpacity[i] = this.sigmoid(opacity[i]);
        }
        this._baseOpacity = baseOpacity;
        return baseOpacity;
    }

    private getVisibilitySvCpuCache() {
        const visibilityData = this.visibilityData;
        if (visibilityData.mode !== 'sv') {
            return null;
        }

        if (this._visibilitySvCpuCache) {
            return this._visibilitySvCpuCache;
        }

        const cache: VisibilitySvCpuCache = {
            siteX: [],
            siteY: [],
            siteZ: [],
            tauSoftplus: []
        };
        const numSplats = this.splatData.numSplats;

        for (let lobeIndex = 0; lobeIndex < visibilityData.numLobes; lobeIndex++) {
            const lobe = visibilityData.lobes[lobeIndex];
            const siteX = new Float32Array(numSplats);
            const siteY = new Float32Array(numSplats);
            const siteZ = new Float32Array(numSplats);
            const tauSoftplus = new Float32Array(numSplats);

            for (let i = 0; i < numSplats; i++) {
                const sx = lobe.siteX[i];
                const sy = lobe.siteY[i];
                const sz = lobe.siteZ[i];
                const invLen = 1 / Math.max(1e-6, Math.sqrt(sx * sx + sy * sy + sz * sz));
                siteX[i] = sx * invLen;
                siteY[i] = sy * invLen;
                siteZ[i] = sz * invLen;
                tauSoftplus[i] = this.softplus(lobe.tau[i]);
            }

            cache.siteX.push(siteX);
            cache.siteY.push(siteY);
            cache.siteZ.push(siteZ);
            cache.tauSoftplus.push(tauSoftplus);
        }

        this._visibilitySvCpuCache = cache;
        return cache;
    }

    private evalVisibilitySVDeg3Cached(dx: number, dy: number, dz: number, cache: VisibilitySvCpuCache, i: number) {
        const dirLen = Math.max(1e-6, Math.sqrt(dx * dx + dy * dy + dz * dz));
        const vx = dx / dirLen;
        const vy = dy / dirLen;
        const vz = dz / dirLen;
        const logits = this._visibilityLogitScratch;

        let maxLogit = -Infinity;
        for (let lobeIndex = 0; lobeIndex < cache.siteX.length; lobeIndex++) {
            const sx = cache.siteX[lobeIndex][i];
            const sy = cache.siteY[lobeIndex][i];
            const sz = cache.siteZ[lobeIndex][i];
            const dist = Math.sqrt((sx - vx) * (sx - vx) + (sy - vy) * (sy - vy) + (sz - vz) * (sz - vz));
            const logit = -cache.tauSoftplus[lobeIndex][i] * dist;
            logits[lobeIndex] = logit;
            maxLogit = Math.max(maxLogit, logit);
        }

        const visibilityData = this.visibilityData as VisibilitySvData;
        let weightedValue = 0;
        let totalWeight = 0;
        for (let lobeIndex = 0; lobeIndex < cache.siteX.length; lobeIndex++) {
            const weight = Math.exp(logits[lobeIndex] - maxLogit);
            weightedValue += weight * visibilityData.lobes[lobeIndex].value[i];
            totalWeight += weight;
        }

        return weightedValue / Math.max(totalWeight, 1e-6);
    }

    private getCameraPositionInSplatSpace(cameraNode: Entity | null, result: Vec3) {
        if (cameraNode) {
            cameraNode.getWorldTransform().getTranslation(result);
        } else {
            result.copy(this.scene.camera.entity.getPosition());
        }

        mat.copy(this.entity.getWorldTransform()).invert();
        mat.transformPoint(result, result);
        return result;
    }

    private currentPreSortSource(): PreSortSource {
        if (!this.isDynamic || !this.dynManifest) {
            return {
                ready: true,
                frame: -1,
                tAbs: 0,
                segmentIdx: -1,
                indices: null
            };
        }

        const currentFrame = (this.scene.events.invoke('timeline.frame') ?? 0) as number;
        const totalFrames = Math.max(1, Math.ceil(this.dynManifest.duration * this.dynManifest.fps));
        const frame = currentFrame % totalFrames;
        const tAbs = this.dynManifest.start + (frame / this.dynManifest.fps);
        const segmentIdx = this.findSegment(tAbs);

        let indices: Uint32Array | null = null;
        if (this.currentSegmentIndex === segmentIdx && this.activeIndices) {
            indices = this.activeIndices;
        } else if (this.segmentCache.has(segmentIdx)) {
            indices = this.segmentCache.get(segmentIdx)!;
            this.activeIndices = indices;
            this.currentSegmentIndex = segmentIdx;
        } else {
            if (!this.loadingSegments.has(segmentIdx)) {
                this.loadSegment(segmentIdx).then((loadedIndices) => {
                    if (!loadedIndices) {
                        return;
                    }

                    this.activeIndices = loadedIndices;
                    this.currentSegmentIndex = segmentIdx;
                    this.markPreSortFilterDirty();
                    this.scene.forceRender = true;
                    this.scene.app.renderNextFrame = true;
                });
            }

            return {
                ready: false,
                frame,
                tAbs,
                segmentIdx,
                indices: null
            };
        }

        this.currentTime = tAbs;

        return {
            ready: true,
            frame,
            tAbs,
            segmentIdx,
            indices
        };
    }

    private buildPreSortMapping(
        sourceIndices: Uint32Array | null,
        tAbs: number,
        cameraPosition: Vec3,
        threshold: number,
        frozen: boolean,
        centers: Float32Array
    ): PreSortFilterResult {
        const profileDetails = profiler.enabled;
        const setupStart = profileDetails ? profiler.now() : 0;
        const state = this.splatData.getProp('state') as Uint8Array | null;
        const baseOpacity = this.getBaseOpacity();
        const totalSplats = this.splatData.numSplats;
        const sourceCount = sourceIndices ? sourceIndices.length : totalSplats;
        const scratch = this._preSortScratch && this._preSortScratch.length >= sourceCount ?
            this._preSortScratch :
            new Uint32Array(sourceCount);

        this._preSortScratch = scratch;

        const useDynamic =
            this.isDynamic &&
            !!this._dyn_x0 && !!this._dyn_y0 && !!this._dyn_z0 &&
            !!this._dyn_m0 && !!this._dyn_m1 && !!this._dyn_m2 &&
            !!this._dyn_tc && !!this._dyn_ts &&
            Number.isFinite(tAbs);

        const x0 = this._dyn_x0 as Float32Array;
        const y0 = this._dyn_y0 as Float32Array;
        const z0 = this._dyn_z0 as Float32Array;
        const m0 = this._dyn_m0 as Float32Array;
        const m1 = this._dyn_m1 as Float32Array;
        const m2 = this._dyn_m2 as Float32Array;
        const tc = this._dyn_tc as Float32Array;
        const ts = this._dyn_ts as Float32Array;
        const frozenOpacity = frozen ? this._frozenEffectiveOpacity : null;
        const useFrozenOpacity = !!frozenOpacity;
        const doVisibility = this.hasVisibility && !useFrozenOpacity;
        const visibilityData = this.visibilityData;
        const svCache = doVisibility && visibilityData.mode === 'sv' ? this.getVisibilitySvCpuCache() : null;
        const dynamicOpacityThreshold = threshold;

        let keptCount = 0;
        let deletedRejected = 0;
        let baseOpacityRejected = 0;
        let opacityRejected = 0;
        let dynamicOpacityTested = 0;
        let dynamicOpacityRejected = 0;
        let dynamicCenterUpdated = 0;
        let frozenOpacityTested = 0;
        let frozenOpacityRejected = 0;
        let visibilityTested = 0;
        let visibilityRejected = 0;
        let dynamicOpacitySamples = 0;
        let dynamicOpacitySampleMs = 0;
        let dynamicCenterSamples = 0;
        let dynamicCenterSampleMs = 0;
        let frozenOpacitySamples = 0;
        let frozenOpacitySampleMs = 0;
        let visibilitySamples = 0;
        let visibilityDirectionSampleMs = 0;
        let visibilityEvalSampleMs = 0;
        let visibilityApplySampleMs = 0;

        const setupMs = profileDetails ? profiler.now() - setupStart : 0;
        const loopStart = profileDetails ? profiler.now() : 0;

        for (let source = 0; source < sourceCount; source++) {
            const index = sourceIndices ? sourceIndices[source] : source;
            const sampleProfile = profileDetails && (source & PRESORT_PROFILE_SAMPLE_MASK) === 0;

            if (state && (state[index] & State.deleted) !== 0) {
                deletedRejected++;
                continue;
            }

            let opacity = baseOpacity ? baseOpacity[index] : 1;
            let cx = 0;
            let cy = 0;
            let cz = 0;
            let centerReady = false;

            if (useDynamic) {
                dynamicOpacityTested++;
                const dynamicOpacityStart = sampleProfile ? profiler.now() : 0;
                const dt = tAbs - tc[index];
                const dtScaled = dt / Math.max(ts[index], 1e-6);
                opacity *= Math.exp(-(dtScaled * dtScaled));
                if (sampleProfile) {
                    dynamicOpacitySamples++;
                    dynamicOpacitySampleMs += profiler.now() - dynamicOpacityStart;
                }

                if (opacity < dynamicOpacityThreshold) {
                    opacityRejected++;
                    dynamicOpacityRejected++;
                    continue;
                }

                const dynamicCenterStart = sampleProfile ? profiler.now() : 0;
                cx = x0[index] + m0[index] * dt;
                cy = y0[index] + m1[index] * dt;
                cz = z0[index] + m2[index] * dt;
                const centerOffset = index * 3;
                centers[centerOffset + 0] = cx;
                centers[centerOffset + 1] = cy;
                centers[centerOffset + 2] = cz;
                centerReady = true;
                dynamicCenterUpdated++;
                if (sampleProfile) {
                    dynamicCenterSamples++;
                    dynamicCenterSampleMs += profiler.now() - dynamicCenterStart;
                }
            } else if (!useFrozenOpacity && opacity < threshold) {
                opacityRejected++;
                baseOpacityRejected++;
                continue;
            }

            if (useFrozenOpacity) {
                frozenOpacityTested++;
                const frozenOpacityStart = sampleProfile ? profiler.now() : 0;
                opacity = frozenOpacity![index];
                if (sampleProfile) {
                    frozenOpacitySamples++;
                    frozenOpacitySampleMs += profiler.now() - frozenOpacityStart;
                }
                if (opacity < threshold) {
                    opacityRejected++;
                    frozenOpacityRejected++;
                    continue;
                }
            }

            if (doVisibility) {
                const visibilityDirectionStart = sampleProfile ? profiler.now() : 0;
                if (!centerReady) {
                    const centerOffset = index * 3;
                    cx = centers[centerOffset + 0];
                    cy = centers[centerOffset + 1];
                    cz = centers[centerOffset + 2];
                }

                const dx = cx - cameraPosition.x;
                const dy = cy - cameraPosition.y;
                const dz = cz - cameraPosition.z;
                const invLen = 1 / Math.max(1e-6, Math.sqrt(dx * dx + dy * dy + dz * dz));
                const vx = dx * invLen;
                const vy = dy * invLen;
                const vz = dz * invLen;
                if (sampleProfile) {
                    visibilityDirectionSampleMs += profiler.now() - visibilityDirectionStart;
                }

                let visRaw = 0;
                const visibilityEvalStart = sampleProfile ? profiler.now() : 0;
                if (visibilityData.mode === 'sv' && svCache) {
                    visRaw = this.evalVisibilitySVDeg3Cached(vx, vy, vz, svCache, index);
                } else if (visibilityData.mode === 'sh') {
                    visRaw = this.evalVisibilitySHDeg3(vx, vy, vz, visibilityData.coeffs, index);
                }
                if (sampleProfile) {
                    visibilityEvalSampleMs += profiler.now() - visibilityEvalStart;
                }

                visibilityTested++;
                const visibilityApplyStart = sampleProfile ? profiler.now() : 0;
                opacity *= this.sigmoid(visRaw);
                if (sampleProfile) {
                    visibilitySamples++;
                    visibilityApplySampleMs += profiler.now() - visibilityApplyStart;
                }
                if (opacity < threshold) {
                    visibilityRejected++;
                    continue;
                }
            }

            scratch[keptCount++] = index;
        }

        const loopMs = profileDetails ? profiler.now() - loopStart : 0;
        const finalizeStart = profileDetails ? profiler.now() : 0;
        const allSourceSplats = !sourceIndices && keptCount === sourceCount && deletedRejected === 0;
        const mapping = allSourceSplats ? null : scratch.slice(0, keptCount);
        const finalizeMs = profileDetails ? profiler.now() - finalizeStart : 0;
        const dynamicOpacityEstimatedMs = estimateSampledMs(dynamicOpacitySampleMs, dynamicOpacitySamples, dynamicOpacityTested);
        const dynamicCenterEstimatedMs = estimateSampledMs(dynamicCenterSampleMs, dynamicCenterSamples, dynamicCenterUpdated);
        const frozenOpacityEstimatedMs = estimateSampledMs(frozenOpacitySampleMs, frozenOpacitySamples, frozenOpacityTested);
        const visibilityDirectionEstimatedMs = estimateSampledMs(visibilityDirectionSampleMs, visibilitySamples, visibilityTested);
        const visibilityEvalEstimatedMs = estimateSampledMs(visibilityEvalSampleMs, visibilitySamples, visibilityTested);
        const visibilityApplyEstimatedMs = estimateSampledMs(visibilityApplySampleMs, visibilitySamples, visibilityTested);
        const visibilityEstimatedMs = visibilityDirectionEstimatedMs + visibilityEvalEstimatedMs + visibilityApplyEstimatedMs;

        return {
            mapping,
            sourceCount,
            keptCount,
            deletedRejected,
            baseOpacityRejected,
            opacityRejected,
            dynamicOpacityTested,
            dynamicOpacityRejected,
            dynamicCenterUpdated,
            frozenOpacityTested,
            frozenOpacityRejected,
            visibilityTested,
            visibilityRejected,
            setupMs,
            loopMs,
            finalizeMs,
            dynamicOpacitySamples,
            dynamicOpacitySampleMs,
            dynamicOpacityEstimatedMs,
            dynamicCenterSamples,
            dynamicCenterSampleMs,
            dynamicCenterEstimatedMs,
            frozenOpacitySamples,
            frozenOpacitySampleMs,
            frozenOpacityEstimatedMs,
            visibilitySamples,
            visibilityDirectionSampleMs,
            visibilityDirectionEstimatedMs,
            visibilityEvalSampleMs,
            visibilityEvalEstimatedMs,
            visibilityApplySampleMs,
            visibilityApplyEstimatedMs,
            visibilityEstimatedMs
        };
    }

    private applyPreSortFilter(cameraNode: Entity | null = null, force = false) {
        if (!this.needsPreSortOpacityFilter()) {
            return false;
        }

        const source = this.currentPreSortSource();
        if (!source.ready) {
            return false;
        }

        const sorter = this.entity.gsplat.instance.sorter;
        if (!sorter) {
            return false;
        }

        const frozen = this.hasVisibility && !!this.scene.events.invoke('visibility.freezeEffectiveOpacity');
        if (frozen && !this._frozenEffectiveOpacity) {
            this.freezeEffectiveOpacity();
        }

        const threshold = this.preSortCullThreshold();
        this.getCameraPositionInSplatSpace(cameraNode, vecc);

        const cameraSensitive = this.hasVisibility && !frozen;
        const cameraChanged = cameraSensitive && (
            Math.abs(vecc.x - this._lastPreSortCamera.x) > PRESORT_CAMERA_EPSILON ||
            Math.abs(vecc.y - this._lastPreSortCamera.y) > PRESORT_CAMERA_EPSILON ||
            Math.abs(vecc.z - this._lastPreSortCamera.z) > PRESORT_CAMERA_EPSILON
        );

        const needsUpdate =
            force ||
            this._lastPreSortRevision !== this._preSortFilterRevision ||
            this._lastPreSortFrame !== source.frame ||
            this._lastPreSortSegment !== source.segmentIdx ||
            this._lastPreSortTime !== source.tAbs ||
            this._lastPreSortThreshold !== threshold ||
            this._lastPreSortFrozen !== frozen ||
            cameraChanged;

        if (!needsUpdate) {
            return false;
        }

        const start = profiler.now();
        const result = this.buildPreSortMapping(source.indices, source.tAbs, vecc, threshold, frozen, sorter.centers);
        const buildMs = profiler.now() - start;

        this._lastPreSortRevision = this._preSortFilterRevision;
        this._lastPreSortFrame = source.frame;
        this._lastPreSortSegment = source.segmentIdx;
        this._lastPreSortTime = source.tAbs;
        this._lastPreSortThreshold = threshold;
        this._lastPreSortFrozen = frozen;
        this._lastPreSortRenderCount = result.keptCount;
        this._lastPreSortCamera.copy(vecc);
        this.lastActiveSplats = result.sourceCount;
        this.lastDrawSplats = result.keptCount;

        const profileData = {
            splat: this,
            frame: source.frame,
            segment: source.segmentIdx,
            isDynamic: this.isDynamic,
            visibilityMode: this.visibilityMode,
            visibilityNumLobes: this.visibilityNumLobes,
            profileSampleRate: PRESORT_PROFILE_SAMPLE_RATE,
            sourceCount: result.sourceCount,
            keptCount: result.keptCount,
            deletedRejected: result.deletedRejected,
            baseOpacityRejected: result.baseOpacityRejected,
            opacityRejected: result.opacityRejected,
            dynamicOpacityTested: result.dynamicOpacityTested,
            dynamicOpacityRejected: result.dynamicOpacityRejected,
            dynamicCenterUpdated: result.dynamicCenterUpdated,
            frozenOpacityTested: result.frozenOpacityTested,
            frozenOpacityRejected: result.frozenOpacityRejected,
            visibilityTested: result.visibilityTested,
            visibilityRejected: result.visibilityRejected,
            threshold,
            frozen,
            setupMs: result.setupMs,
            loopMs: result.loopMs,
            finalizeMs: result.finalizeMs,
            dynamicOpacitySamples: result.dynamicOpacitySamples,
            dynamicOpacitySampleMs: result.dynamicOpacitySampleMs,
            dynamicOpacityEstimatedMs: result.dynamicOpacityEstimatedMs,
            dynamicCenterSamples: result.dynamicCenterSamples,
            dynamicCenterSampleMs: result.dynamicCenterSampleMs,
            dynamicCenterEstimatedMs: result.dynamicCenterEstimatedMs,
            frozenOpacitySamples: result.frozenOpacitySamples,
            frozenOpacitySampleMs: result.frozenOpacitySampleMs,
            frozenOpacityEstimatedMs: result.frozenOpacityEstimatedMs,
            visibilitySamples: result.visibilitySamples,
            visibilityDirectionSampleMs: result.visibilityDirectionSampleMs,
            visibilityDirectionEstimatedMs: result.visibilityDirectionEstimatedMs,
            visibilityEvalSampleMs: result.visibilityEvalSampleMs,
            visibilityEvalEstimatedMs: result.visibilityEvalEstimatedMs,
            visibilityApplySampleMs: result.visibilityApplySampleMs,
            visibilityApplyEstimatedMs: result.visibilityApplyEstimatedMs,
            visibilityEstimatedMs: result.visibilityEstimatedMs,
            buildMs
        };

        profiler.addFrameValue('prefilterMs', buildMs);
        profiler.addFrameValue('prefilterSetupMs', result.setupMs);
        profiler.addFrameValue('prefilterLoopMs', result.loopMs);
        profiler.addFrameValue('prefilterFinalizeMs', result.finalizeMs);
        profiler.addFrameValue('prefilterDynamicOpacityEstimatedMs', result.dynamicOpacityEstimatedMs);
        profiler.addFrameValue('prefilterDynamicCenterEstimatedMs', result.dynamicCenterEstimatedMs);
        profiler.addFrameValue('prefilterFrozenOpacityEstimatedMs', result.frozenOpacityEstimatedMs);
        profiler.addFrameValue('prefilterVisibilityDirectionEstimatedMs', result.visibilityDirectionEstimatedMs);
        profiler.addFrameValue('prefilterVisibilityEvalEstimatedMs', result.visibilityEvalEstimatedMs);
        profiler.addFrameValue('prefilterVisibilityApplyEstimatedMs', result.visibilityApplyEstimatedMs);
        profiler.addFrameValue('prefilterVisibilityEstimatedMs', result.visibilityEstimatedMs);
        profiler.maxFrameValue('prefilterSourceSplats', result.sourceCount);
        profiler.maxFrameValue('prefilterKeptSplats', result.keptCount);
        profiler.addFrameValue('prefilterBaseOpacityRejected', result.baseOpacityRejected);
        profiler.addFrameValue('prefilterOpacityRejected', result.opacityRejected);
        profiler.addFrameValue('prefilterDynamicOpacityTested', result.dynamicOpacityTested);
        profiler.addFrameValue('prefilterDynamicOpacityRejected', result.dynamicOpacityRejected);
        profiler.addFrameValue('prefilterDynamicCenterUpdated', result.dynamicCenterUpdated);
        profiler.addFrameValue('prefilterFrozenOpacityTested', result.frozenOpacityTested);
        profiler.addFrameValue('prefilterFrozenOpacityRejected', result.frozenOpacityRejected);
        profiler.addFrameValue('prefilterVisibilityTested', result.visibilityTested);
        profiler.addFrameValue('prefilterVisibilityRejected', result.visibilityRejected);
        profiler.setFrameValues({
            segmentIndex: source.segmentIdx,
            currentTime: source.tAbs,
            activeSplats: result.sourceCount,
            drawSplats: result.keptCount
        });
        profiler.event('splat.prefilter', {
            splat: this.name,
            frame: source.frame,
            segment: source.segmentIdx,
            isDynamic: this.isDynamic,
            visibilityMode: this.visibilityMode,
            visibilityNumLobes: this.visibilityNumLobes,
            profileSampleRate: PRESORT_PROFILE_SAMPLE_RATE,
            sourceCount: result.sourceCount,
            keptCount: result.keptCount,
            deletedRejected: result.deletedRejected,
            baseOpacityRejected: result.baseOpacityRejected,
            opacityRejected: result.opacityRejected,
            dynamicOpacityTested: result.dynamicOpacityTested,
            dynamicOpacityRejected: result.dynamicOpacityRejected,
            dynamicCenterUpdated: result.dynamicCenterUpdated,
            frozenOpacityTested: result.frozenOpacityTested,
            frozenOpacityRejected: result.frozenOpacityRejected,
            visibilityTested: result.visibilityTested,
            visibilityRejected: result.visibilityRejected,
            threshold,
            frozen,
            setupMs: result.setupMs,
            loopMs: result.loopMs,
            finalizeMs: result.finalizeMs,
            dynamicOpacitySamples: result.dynamicOpacitySamples,
            dynamicOpacitySampleMs: result.dynamicOpacitySampleMs,
            dynamicOpacityEstimatedMs: result.dynamicOpacityEstimatedMs,
            dynamicCenterSamples: result.dynamicCenterSamples,
            dynamicCenterSampleMs: result.dynamicCenterSampleMs,
            dynamicCenterEstimatedMs: result.dynamicCenterEstimatedMs,
            frozenOpacitySamples: result.frozenOpacitySamples,
            frozenOpacitySampleMs: result.frozenOpacitySampleMs,
            frozenOpacityEstimatedMs: result.frozenOpacityEstimatedMs,
            visibilitySamples: result.visibilitySamples,
            visibilityDirectionSampleMs: result.visibilityDirectionSampleMs,
            visibilityDirectionEstimatedMs: result.visibilityDirectionEstimatedMs,
            visibilityEvalSampleMs: result.visibilityEvalSampleMs,
            visibilityEvalEstimatedMs: result.visibilityEvalEstimatedMs,
            visibilityApplySampleMs: result.visibilityApplySampleMs,
            visibilityApplyEstimatedMs: result.visibilityApplyEstimatedMs,
            visibilityEstimatedMs: result.visibilityEstimatedMs,
            buildMs
        });

        this.scene.events.fire('splat.prefilterProfile', profileData);

        const setMappingStart = profiler.now();
        sorter.setMapping(result.mapping);
        profiler.addFrameValue('setMappingCallMs', profiler.now() - setMappingStart);
        return true;
    }

    get renderSplats() {
        return this._lastPreSortRenderCount >= 0 ? this._lastPreSortRenderCount : this.numSplats;
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
            this.markPreSortFilterDirty();
            if (!this.needsPreSortOpacityFilter()) {
                this.updateSorting();
            } else {
                this.applyPreSortFilter(null, true);
            }
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

        this.markPreSortFilterDirty();
        if (this.needsPreSortOpacityFilter()) {
            this.applyPreSortFilter(null, true);
        } else {
            this.updateSorting();
        }

        this.scene.forceRender = true;
        this.scene.events.fire('splat.positionsChanged', this);
    }

    updateSorting() {
        if (this.needsPreSortOpacityFilter()) {
            this.markPreSortFilterDirty();
            if (this.applyPreSortFilter(null, true) || this.isDynamic) {
                return;
            }
        }

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

        if (this.hasVisibility) {
            const initialFrozen = !!this.scene.events.invoke('visibility.freezeEffectiveOpacity');
            if (initialFrozen) {
                this.freezeEffectiveOpacity();
            }

            this._freezeOpacityHandler = (enabled: boolean) => {
                const material = this.entity.gsplat.instance.material;
                material.setDefine('FROZEN_OPACITY', enabled);
                material.update();

                if (enabled) {
                    this.freezeEffectiveOpacity();
                }

                this.markPreSortFilterDirty();
                this.applyPreSortFilter(null, true);
                this.scene.forceRender = true;
                this.scene.app.renderNextFrame = true;
            };
            this.scene.events.on('visibility.freezeEffectiveOpacity', this._freezeOpacityHandler);
        }

        // we must update state in case the state data was loaded from ply
        this.updateState();

        // Initialize dynamic gaussian: load first segment and set initial time
        if (this.isDynamic && this.dynManifest) {
            // Notify timeline to switch to dynamic mode
            // Use setTimeout to ensure timeline events are registered
            setTimeout(() => {
                this.scene.events.fire('timeline.setDynamic', this.dynManifest!.duration, this.dynManifest!.fps);
                // Register dynamic gaussian control (only if not already registered)
                if (!this.scene.events.functions.has('scene.hasDynamicGaussian')) {
                    this.scene.events.function('scene.hasDynamicGaussian', () => true);
                }
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
            const emptyMappingStart = profiler.now();
            this._lastPreSortRenderCount = 0;
            this.lastActiveSplats = 0;
            this.lastDrawSplats = 0;
            this.entity.gsplat.instance.sorter.setMapping(new Uint32Array(0));
            profiler.addFrameValue('setMappingCallMs', profiler.now() - emptyMappingStart);

            this.loadSegment(initialSegmentIndex).then((indices) => {
                // Only apply if this is still the current segment
                if (this.currentSegmentIndex !== initialSegmentIndex) {
                    return;
                }

                if (indices) {
                    // Cache active indices
                    this.activeIndices = indices;
                    this.lastActiveSplats = indices.length;
                    profiler.setFrameValues({
                        activeSplats: indices.length,
                        segmentIndex: initialSegmentIndex
                    });

                    // Mark as pending sort so that uCurrentTime is set when sorting completes
                    this.pendingSort = true;
                    this.lastSortedFrame = 0;
                    this.lastSortedTime = initialFrameTime;

                    this.applyPreSortFilter(null, true);

                    this.preloadNextSegment(initialSegmentIndex);
                }
            });
        }
    }

    private sigmoid(v: number) {
        if (v >= 0) {
            return 1 / (1 + Math.exp(-v));
        }
        const t = Math.exp(v);
        return t / (1 + t);
    }

    private softplus(v: number) {
        return Math.log1p(Math.exp(-Math.abs(v))) + Math.max(v, 0);
    }

    private evalVisibilitySHDeg3(dx: number, dy: number, dz: number, sh: Float32Array[], i: number) {
        const x = dx;
        const y = dy;
        const z = dz;
        const xx = x * x;
        const yy = y * y;
        const zz = z * z;

        const C0 = 0.28209479177387814;
        const C1 = 0.4886025119029199;
        const C2_0 = 1.0925484305920792;
        const C2_1 = -1.0925484305920792;
        const C2_2 = 0.31539156525252005;
        const C2_3 = -1.0925484305920792;
        const C2_4 = 0.5462742152960396;
        const C3_0 = -0.5900435899266435;
        const C3_1 = 2.890611442640554;
        const C3_2 = -0.4570457994644658;
        const C3_3 = 0.3731763325901154;
        const C3_4 = -0.4570457994644658;
        const C3_5 = 1.445305721320277;
        const C3_6 = -0.5900435899266435;

        let r = 0;
        r += C0 * sh[0][i];

        r += (-C1 * y) * sh[1][i];
        r += (C1 * z) * sh[2][i];
        r += (-C1 * x) * sh[3][i];

        r += C2_0 * (x * y) * sh[4][i];
        r += C2_1 * (y * z) * sh[5][i];
        r += C2_2 * (2 * zz - xx - yy) * sh[6][i];
        r += C2_3 * (x * z) * sh[7][i];
        r += C2_4 * (xx - yy) * sh[8][i];

        r += C3_0 * y * (3 * xx - yy) * sh[9][i];
        r += C3_1 * (x * y * z) * sh[10][i];
        r += C3_2 * y * (4 * zz - xx - yy) * sh[11][i];
        r += C3_3 * z * (2 * zz - 3 * xx - 3 * yy) * sh[12][i];
        r += C3_4 * x * (4 * zz - xx - yy) * sh[13][i];
        r += C3_5 * z * (xx - yy) * sh[14][i];
        r += C3_6 * x * (xx - 3 * yy) * sh[15][i];

        return r;
    }

    private evalVisibilitySVDeg3(dx: number, dy: number, dz: number, visibilitySv: VisibilitySvData, i: number) {
        const dirLen = Math.max(1e-6, Math.sqrt(dx * dx + dy * dy + dz * dz));
        const vx = dx / dirLen;
        const vy = dy / dirLen;
        const vz = dz / dirLen;

        const logits = new Array<number>(visibilitySv.numLobes);
        let maxLogit = -Infinity;
        for (let lobeIndex = 0; lobeIndex < visibilitySv.numLobes; lobeIndex++) {
            const lobe = visibilitySv.lobes[lobeIndex];
            const siteX = lobe.siteX[i];
            const siteY = lobe.siteY[i];
            const siteZ = lobe.siteZ[i];
            const siteLen = Math.max(1e-6, Math.sqrt(siteX * siteX + siteY * siteY + siteZ * siteZ));
            const sx = siteX / siteLen;
            const sy = siteY / siteLen;
            const sz = siteZ / siteLen;
            const dist = Math.sqrt((sx - vx) * (sx - vx) + (sy - vy) * (sy - vy) + (sz - vz) * (sz - vz));
            const logit = -this.softplus(lobe.tau[i]) * dist;
            logits[lobeIndex] = logit;
            maxLogit = Math.max(maxLogit, logit);
        }

        let weightedValue = 0;
        let totalWeight = 0;
        for (let lobeIndex = 0; lobeIndex < visibilitySv.numLobes; lobeIndex++) {
            const weight = Math.exp(logits[lobeIndex] - maxLogit);
            weightedValue += weight * visibilitySv.lobes[lobeIndex].value[i];
            totalWeight += weight;
        }

        return weightedValue / Math.max(totalWeight, 1e-6);
    }

    freezeEffectiveOpacity() {
        if (!this.hasVisibility) {
            return;
        }

        const frozenTex = this.visibilityFrozenOpacityTexture ?? undefined;
        if (!frozenTex) {
            throw new Error('Frozen opacity texture not available');
        }

        const x = this.splatData.getProp('x') as Float32Array;
        const y = this.splatData.getProp('y') as Float32Array;
        const z = this.splatData.getProp('z') as Float32Array;
        const opacity = this.splatData.getProp('opacity') as Float32Array;
        const baseOpacity = this.getBaseOpacity();

        const motion0 = this.splatData.getProp('motion_0') as Float32Array | null;
        const motion1 = this.splatData.getProp('motion_1') as Float32Array | null;
        const motion2 = this.splatData.getProp('motion_2') as Float32Array | null;
        const trbfCenter = this.splatData.getProp('trbf_center') as Float32Array | null;
        const trbfScale = this.splatData.getProp('trbf_scale') as Float32Array | null;

        const visibilityData = this.visibilityData;

        // Camera position in model space (same space as x/y/z)
        const camWorld = this.scene.camera.entity.getPosition();
        mat.copy(this.entity.getWorldTransform()).invert();
        mat.transformPoint(camWorld, vecc);
        const camX = vecc.x;
        const camY = vecc.y;
        const camZ = vecc.z;

        const useDynamic =
            this.isDynamic &&
            !!this.dynManifest &&
            !!motion0 && !!motion1 && !!motion2 &&
            !!trbfCenter && !!trbfScale;

        let t_abs = 0;
        if (useDynamic) {
            // Prefer the actual time used for rendering (stable playback), fallback to timeline time.
            if (!Number.isNaN(this.lastSortedTime)) {
                t_abs = this.lastSortedTime;
            } else {
                const events = this.scene.events;
                const currentFrame = (events.invoke('timeline.frame') ?? 0) as number;
                const totalFrames = Math.ceil(this.dynManifest!.duration * this.dynManifest!.fps);
                const frame = currentFrame % totalFrames;
                t_abs = this.dynManifest!.start + (frame / this.dynManifest!.fps);
            }
        }

        const numSplats = this.splatData.numSplats;

        const locked = frozenTex.lock() as unknown as Float32Array | ArrayBufferView;
        const data = locked instanceof Float32Array ? locked : new Float32Array((locked as any).buffer);
        const frozenOpacity = new Float32Array(numSplats);

        data.fill(0);
        const svCache = visibilityData.mode === 'sv' ? this.getVisibilitySvCpuCache() : null;

        for (let i = 0; i < numSplats; i++) {
            let cx = x[i];
            let cy = y[i];
            let cz = z[i];

            let opBefore = baseOpacity ? baseOpacity[i] : this.sigmoid(opacity[i]);

            if (useDynamic) {
                const dt = t_abs - (trbfCenter as Float32Array)[i];

                // Match shader center: p(t) = p0 + motion * dt
                cx += (motion0 as Float32Array)[i] * dt;
                cy += (motion1 as Float32Array)[i] * dt;
                cz += (motion2 as Float32Array)[i] * dt;

                // Match shader dynamic opacity: exp(-dt_scaled^2)
                const ts = Math.max((trbfScale as Float32Array)[i], 1e-6);
                const dtScaled = dt / ts;
                const gaussian = Math.exp(-(dtScaled * dtScaled));
                opBefore *= gaussian;
            }

            const dx = cx - camX;
            const dy = cy - camY;
            const dz = cz - camZ;
            const invLen = 1 / Math.max(1e-6, Math.sqrt(dx * dx + dy * dy + dz * dz));
            const vx = dx * invLen;
            const vy = dy * invLen;
            const vz = dz * invLen;

            let visRaw = 0;
            if (visibilityData.mode === 'sv' && svCache) {
                visRaw = this.evalVisibilitySVDeg3Cached(vx, vy, vz, svCache, i);
            } else if (visibilityData.mode === 'sh') {
                visRaw = this.evalVisibilitySHDeg3(vx, vy, vz, visibilityData.coeffs, i);
            }
            const visibility = this.sigmoid(visRaw);
            const opEff = visibility * opBefore;

            data[i] = opEff;
            frozenOpacity[i] = opEff;
        }

        frozenTex.unlock();
        this._frozenEffectiveOpacity = frozenOpacity;
        this.markPreSortFilterDirty();
    }

    remove() {
        this.scene.events.off('view.bands', this.rebuildMaterial, this);
        if (this._freezeOpacityHandler) {
            this.scene.events.off('visibility.freezeEffectiveOpacity', this._freezeOpacityHandler);
            this._freezeOpacityHandler = null;
        }

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
        if (frame !== this.lastSortedFrame && this.pendingSort) {
            profiler.addFrameValue('sortPendingFrames', 1);
            profiler.event('splat.sort.skipped', {
                splat: this.name,
                frame,
                lastSortedFrame: this.lastSortedFrame,
                reason: 'pending-sort'
            });
        }

        if (needsUpdate) {
            // 4. 找到对应的 segment
            const segmentIdx = this.findSegment(t_abs);

            // 5. 检查 segment 是否在缓存中
            if (this.segmentCache.has(segmentIdx)) {
                // 6. 更新 centers: p(t) = p0 + motion * (t - trbf_center)
                const indices = this.segmentCache.get(segmentIdx)!;
                this.activeIndices = indices;
                this.lastActiveSplats = indices.length;
                this.currentSegmentIndex = segmentIdx;
                profiler.addFrameValue('activeSplatsUpdated', indices.length);
                profiler.setFrameValues({
                    segmentIndex: segmentIdx,
                    currentTime: t_abs,
                    activeSplats: indices.length
                });

                // 7. 触发排序 (shader uniform uCurrentTime will be set when sorting completes)
                this.pendingSort = true;
                this.lastSortedFrame = frame;
                this.lastSortedTime = t_abs;
                this.applyPreSortFilter(null, true);

                // 预加载下一个 segment
                this.preloadNextSegment(segmentIdx);

            } else if (!this.loadingSegments.has(segmentIdx)) {
                // segment 不在缓存，异步加载
                profiler.event('segment.cacheMiss', {
                    splat: this.name,
                    segmentIndex: segmentIdx,
                    frame,
                    time: t_abs
                });
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

        if (this.hasVisibility) {
            const frozen = !!events.invoke('visibility.freezeEffectiveOpacity');
            if (!frozen) {
                const camWorld = this.scene.camera.entity.getPosition();
                mat.copy(this.entity.getWorldTransform()).invert();
                mat.transformPoint(camWorld, vecc);
                material.setParameter('uCameraPosition', [vecc.x, vecc.y, vecc.z]);
            }
        }

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
        this.markPreSortFilterDirty();
        this.applyPreSortFilter(null, true);

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
