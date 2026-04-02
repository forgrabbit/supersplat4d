import {
    CULLFACE_NONE,
    SEMANTIC_POSITION,
    BlendState,
    DepthState,
    Color,
    drawQuadWithShader,
    Entity,
    Shader,
    ShaderUtils,
    QuadRender
} from 'playcanvas';

import { Element, ElementType } from './element';
import { vertexShader, fragmentShader } from './shaders/outline-shader';
import { Splat } from './splat';

class Outline extends Element {
    entity: Entity;
    shader: Shader;
    quadRender: QuadRender;
    outlineTextureId: any;
    alphaCutoffId: any;
    clrId: any;
    clrStorage = [1, 1, 1, 1];
    enabled = true;
    clr = new Color(1, 1, 1, 0.5);

    constructor() {
        super(ElementType.other);

        this.entity = new Entity('outlineCamera');
        this.entity.addComponent('camera');
        this.entity.camera.setShaderPass('OUTLINE');
        this.entity.camera.clearColor = new Color(0, 0, 0, 0);
    }

    add() {
        const device = this.scene.app.graphicsDevice;
        const layerId = this.scene.overlayLayer.id;

        // add selected splat to outline layer
        this.scene.events.on('selection.changed', (splat: Splat, prev: Splat) => {
            if (prev) {
                prev.entity.gsplat.layers = prev.entity.gsplat.layers.filter(id => id !== layerId);
            }
            if (splat) {
                splat.entity.gsplat.layers = splat.entity.gsplat.layers.concat([layerId]);
            }
        });

        // render overlay layer only
        this.entity.camera.layers = [layerId];
        this.scene.camera.entity.addChild(this.entity);

        this.shader = ShaderUtils.createShader(device, {
            uniqueName: 'apply-outline',
            attributes: {
                vertex_position: SEMANTIC_POSITION
            },
            vertexGLSL: vertexShader,
            fragmentGLSL: fragmentShader
        });

        this.quadRender = new QuadRender(this.shader);

        this.outlineTextureId = device.scope.resolve('outlineTexture');
        this.alphaCutoffId = device.scope.resolve('alphaCutoff');
        this.clrId = device.scope.resolve('clr');
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

        this.entity.enabled = this.enabled && this.scene.events.invoke('view.outlineSelection');
        this.entity.camera.renderTarget = this.scene.camera.workRenderTarget;
    }

    onPostRender() {
        if (!this.entity.enabled) {
            return;
        }

        const device = this.scene.app.graphicsDevice;
        const events = this.scene.events;

        device.setBlendState(BlendState.ALPHABLEND);
        device.setCullMode(CULLFACE_NONE);
        device.setDepthState(DepthState.NODEPTH);
        device.setStencilState(null, null);

        const selectedClr = events.invoke('selectedClr');
        this.clrStorage[0] = selectedClr.r;
        this.clrStorage[1] = selectedClr.g;
        this.clrStorage[2] = selectedClr.b;
        this.clrStorage[3] = selectedClr.a;

        this.outlineTextureId.setValue(this.entity.camera.renderTarget.colorBuffer);
        this.alphaCutoffId.setValue(events.invoke('camera.mode') === 'rings' ? 0.0 : 0.4);
        this.clrId.setValue(this.clrStorage);
        drawQuadWithShader(device, this.scene.camera.entity.camera.renderTarget, this.shader);
    }
}

export { Outline };
