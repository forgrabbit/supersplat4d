import {
    math,
    now,
    // platform,
    WebglGraphicsDevice,
    // SoundManager,
    // Lightmapper,
    // BatchManager,
    AppBase,
    AppOptions,
    // script,
    // AnimationComponentSystem,
    AnimComponentSystem,
    // AudioListenerComponentSystem,
    // AudioSourceComponentSystem,
    // ButtonComponentSystem,
    // CollisionComponentSystem,
    // ElementComponentSystem,
    // JointComponentSystem,
    // LayoutChildComponentSystem,
    // LayoutGroupComponentSystem,
    // ModelComponentSystem,
    // ParticleSystemComponentSystem,
    RenderComponentSystem,
    // RigidBodyComponentSystem,
    // ScreenComponentSystem,
    // ScriptLegacyComponentSystem,
    // ScrollViewComponentSystem,
    // ScrollbarComponentSystem,
    // SoundComponentSystem,
    // SpriteComponentSystem,
    // ZoneComponentSystem,
    CameraComponentSystem,
    LightComponentSystem,
    GSplatComponentSystem,
    // ScriptComponentSystem,
    RenderHandler,
    // AnimationHandler,
    AnimClipHandler,
    AnimStateGraphHandler,
    // AudioHandler,
    // BinaryHandler,
    ContainerHandler,
    // CssHandler,
    CubemapHandler,
    // FolderHandler,
    // FontHandler,
    GSplatHandler,
    // HierarchyHandler,
    // HtmlHandler,
    // JsonHandler,
    // MaterialHandler,
    // ModelHandler,
    // SceneHandler,
    // ScriptHandler,
    // ShaderHandler,
    // SpriteHandler,
    // TemplateHandler,
    // TextHandler,
    // TextureAtlasHandler,
    TextureHandler
    // XrManager
} from 'playcanvas';

/**
 * Same as AppBase.makeTick, but awaits __supersplat4dGpuAwait after update() so dynamic
 * WebGPU splats can read back activeCount before the frame renders.
 */
function makeSuperSplatTick(application: PCApp) {
    return async function (timestamp: number, xrFrame?: XRFrame) {
        const appAny = application as any;
        if (!application.graphicsDevice) {
            return;
        }
        if (application.frameRequestId) {
            application.xr?.session?.cancelAnimationFrame(application.frameRequestId);
            cancelAnimationFrame(application.frameRequestId);
            application.frameRequestId = undefined;
        }
        appAny._inFrameUpdate = true;
        const currentTime = application._processTimestamp(timestamp) || now();
        const ms = currentTime - (application._time || currentTime);
        let dt = ms / 1000.0;
        dt = math.clamp(dt, 0, application.maxDeltaTime);
        dt *= application.timeScale;
        application._time = currentTime;
        application.requestAnimationFrame();
        if (application.graphicsDevice.contextLost) {
            return;
        }
        appAny._fillFrameStatsBasic(currentTime, dt, ms);
        application.fire('frameupdate', ms);
        let skipUpdate = false;
        if (xrFrame) {
            skipUpdate = !application.xr?.update(xrFrame);
            (application.graphicsDevice as any).defaultFramebuffer = xrFrame.session.renderState.baseLayer.framebuffer;
        } else {
            (application.graphicsDevice as any).defaultFramebuffer = null;
        }
        if (!skipUpdate) {
            application.update(dt);
            const gpu = appAny.__supersplat4dGpuAwait as Promise<void> | undefined;
            if (gpu) {
                await gpu;
                appAny.__supersplat4dGpuAwait = null;
            }
            application.fire('framerender');
            if (application.autoRender || application.renderNextFrame) {
                application.render();
                application.renderNextFrame = false;
            }
            application.fire('frameend');
            application.stats.frameEnd();
        }
        appAny._inFrameUpdate = false;
        if (appAny._destroyRequested) {
            application.destroy();
        }
    };
}

class PCApp extends AppBase {
    constructor(canvas: HTMLCanvasElement, options: any) {
        super(canvas);

        const appOptions = new AppOptions();

        appOptions.graphicsDevice = options.graphicsDevice;
        this.addComponentSystems(appOptions);
        this.addResourceHandles(appOptions);

        appOptions.elementInput = options.elementInput;
        appOptions.keyboard = options.keyboard;
        appOptions.mouse = options.mouse;
        appOptions.touch = options.touch;
        appOptions.gamepads = options.gamepads;

        appOptions.scriptPrefix = options.scriptPrefix;
        appOptions.assetPrefix = options.assetPrefix;
        appOptions.scriptsOrder = options.scriptsOrder;

        // appOptions.soundManager = new SoundManager(options);
        // appOptions.lightmapper = Lightmapper;
        // appOptions.batchManager = BatchManager;
        // appOptions.xr = XrManager;

        this.init(appOptions);

        this.tick = makeSuperSplatTick(this);
    }

    addComponentSystems(appOptions: AppOptions) {
        appOptions.componentSystems = [
            // RigidBodyComponentSystem,
            // CollisionComponentSystem,
            // JointComponentSystem,
            // AnimationComponentSystem,
            // @ts-ignore
            AnimComponentSystem,
            // ModelComponentSystem,
            // @ts-ignore
            RenderComponentSystem,
            // @ts-ignore
            CameraComponentSystem,
            // @ts-ignore
            LightComponentSystem,
            // script.legacy ? ScriptLegacyComponentSystem : ScriptComponentSystem,
            // AudioSourceComponentSystem,
            // SoundComponentSystem,
            // AudioListenerComponentSystem,
            // ParticleSystemComponentSystem,
            // ScreenComponentSystem,
            // ElementComponentSystem,
            // ButtonComponentSystem,
            // ScrollViewComponentSystem,
            // ScrollbarComponentSystem,
            // SpriteComponentSystem,
            // LayoutGroupComponentSystem,
            // LayoutChildComponentSystem,
            // ZoneComponentSystem,
            GSplatComponentSystem
        ];
    }

    addResourceHandles(appOptions: AppOptions) {
        appOptions.resourceHandlers = [
            // @ts-ignore
            RenderHandler,
            // AnimationHandler,
            // @ts-ignore
            AnimClipHandler,
            // @ts-ignore
            AnimStateGraphHandler,
            // ModelHandler,
            // MaterialHandler,
            // @ts-ignore
            TextureHandler,
            // TextHandler,
            // JsonHandler,
            // AudioHandler,
            // ScriptHandler,
            // SceneHandler,
            // @ts-ignore
            CubemapHandler,
            // HtmlHandler,
            // CssHandler,
            // ShaderHandler,
            // HierarchyHandler,
            // FolderHandler,
            // FontHandler,
            // BinaryHandler,
            // TextureAtlasHandler,
            // SpriteHandler,
            // TemplateHandler,
            // @ts-ignore
            ContainerHandler,
            GSplatHandler
        ];
    }
}

export { PCApp };
