import {
    Texture
} from 'playcanvas';

import { Element, ElementType } from './element';
import { Serializer } from './serializer';

class Skybox extends Element {
    texture: Texture | null = null;
    visible = true;
    previousSkybox: Texture | null = null;
    previousSkyboxMip: number = 0;
    previousSkyboxIntensity: number = 1;

    constructor() {
        super(ElementType.other);
    }

    add() {
        // Save previous skybox state
        this.previousSkybox = this.scene.app.scene.skybox;
        this.previousSkyboxMip = this.scene.app.scene.skyboxMip;
        this.previousSkyboxIntensity = this.scene.app.scene.skyboxIntensity;
        
        // Apply texture if already set
        if (this.texture && this.visible) {
            this.applySkybox();
        }
    }

    private applySkybox() {
        if (!this.texture) {
            return;
        }
        
        // Use PlayCanvas built-in skybox rendering
        this.scene.app.scene.skybox = this.texture;
        this.scene.app.scene.skyboxMip = 0; // Use highest quality mipmap
        this.scene.app.scene.skyboxIntensity = 1.0; // Full intensity
        
        // Force render update
        this.scene.forceRender = true;
    }
    
    private removeSkybox() {
        // Restore previous skybox or set to null
        this.scene.app.scene.skybox = this.previousSkybox;
        this.scene.app.scene.skyboxMip = this.previousSkyboxMip;
        this.scene.app.scene.skyboxIntensity = this.previousSkyboxIntensity;
        
        // Force render update
        this.scene.forceRender = true;
    }

    remove() {
        // Restore previous skybox state
        this.removeSkybox();
        // Don't destroy texture here, it's managed by background-handler
    }

    setTexture(texture: Texture) {
        this.texture = texture;
        
        // If already added to scene and visible, apply the skybox immediately
        if (this.scene && this.visible) {
            this.applySkybox();
        }
    }

    setVisible(visible: boolean) {
        this.visible = visible;
        
        if (visible && this.texture) {
            // Apply skybox to scene
            this.applySkybox();
        } else {
            // Remove skybox from scene
            this.removeSkybox();
        }
    }

    serialize(serializer: Serializer): void {
        serializer.pack(this.visible);
    }
}

export { Skybox };
