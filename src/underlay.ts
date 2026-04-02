import {
    BLENDEQUATION_ADD,
    BLENDMODE_ONE,
    BLENDMODE_ZERO,
    SEMANTIC_POSITION,
    BlendState,
    Color,
    drawQuadWithShader,
    Entity,
    Shader,
    ShaderUtils,
    QuadRender
} from 'playcanvas';

import { Element, ElementType } from './element';
import { vertexShader, fragmentShader } from './shaders/blit-shader';
import { isWebGPU } from './utils/graphics-backend';

class Underlay extends Element {
    entity: Entity;
    shader: Shader;
    quadRender: QuadRender;
    _blendState: BlendState;
    _blitTextureId: any;
    enabled = true;

    constructor() {
        super(ElementType.other);

        this.entity = new Entity('underlayCamera');
        this.entity.addComponent('camera');
        this.entity.camera.setShaderPass('UNDERLAY');
        this.entity.camera.clearColor = new Color(0, 0, 0, 0);
    }

    add() {
        const device = this.scene.app.graphicsDevice;

        this.entity.camera.layers = [this.scene.overlayLayer.id];
        this.scene.camera.entity.addChild(this.entity);

        this.shader = ShaderUtils.createShader(device, {
            uniqueName: 'apply-underlay',
            attributes: {
                vertex_position: SEMANTIC_POSITION
            },
            vertexGLSL: vertexShader,
            fragmentGLSL: fragmentShader
        });

        this.quadRender = new QuadRender(this.shader);

        const blitTextureId = device.scope.resolve('blitTexture');
        const blendState = new BlendState(true,
            BLENDEQUATION_ADD, BLENDMODE_ONE, BLENDMODE_ONE,
            BLENDEQUATION_ADD, BLENDMODE_ZERO, BLENDMODE_ONE
        );
        this._blendState = blendState;
        this._blitTextureId = blitTextureId;
    }

    remove() {
        this.scene.camera.entity.removeChild(this.entity);
    }

    onPreRender() {
        // copy camera properties
        const src = this.scene.camera.entity.camera;
        const dst = this.entity.camera;

        dst.projection = src.projection;
        dst.horizontalFov = src.horizontalFov;
        dst.fov = src.fov;
        dst.nearClip = src.nearClip;
        dst.farClip = src.farClip;
        dst.orthoHeight = src.orthoHeight;

        this.entity.enabled = this.enabled && !this.scene.events.invoke('view.outlineSelection');
        this.entity.camera.renderTarget = this.scene.camera.workRenderTarget;
    }

    onPostRender() {
        if (!this.entity.enabled) {
            return;
        }

        const device = this.scene.app.graphicsDevice;
        const skipBlit = isWebGPU(device);
        device.setBlendState(this._blendState);
        this._blitTextureId.setValue(this.entity.camera.renderTarget.colorBuffer);
        if (skipBlit) {
            return;
        }
        drawQuadWithShader(device, this.scene.camera.entity.camera.renderTarget, this.shader);
    }
}

export { Underlay };
