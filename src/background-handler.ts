import { Events } from './events';
import { Scene } from './scene';
import { Skybox } from './skybox';
import { BackgroundInfo } from './ui/background-list';
import { loadCubemapFromFile } from './cubemap-loader';

let nextBackgroundId = 1;
const backgrounds = new Map<string, BackgroundInfo>();
const skyboxes = new Map<string, Skybox>();
let activeSkybox: Skybox | null = null;

const registerBackgroundEvents = (scene: Scene, events: Events) => {
    // Helper function to import cubemap from a File object
    const importCubemapFromFile = async (file: File, autoShow: boolean = false) => {
        const filename = file.name;
        
        // Load cubemap texture
        const device = scene.graphicsDevice;
        const cubemapTexture = await loadCubemapFromFile(device, file);
        
        // Create background info
        const id = `background_${nextBackgroundId++}`;
        const backgroundInfo: BackgroundInfo = {
            id,
            name: filename,
            texture: cubemapTexture,
            visible: false
        };
        
        // Create skybox - Use PlayCanvas built-in skybox rendering
        const skybox = new Skybox();
        skybox.setTexture(cubemapTexture);
        scene.add(skybox);
        skybox.setVisible(autoShow);
        
        backgrounds.set(id, backgroundInfo);
        skyboxes.set(id, skybox);
        
        // Fire event to add to UI
        events.fire('background.added', backgroundInfo);
        
        // Auto-show if requested
        if (autoShow) {
            backgroundInfo.visible = true;
            skybox.setVisible(true);
            // Hide other backgrounds
            if (activeSkybox && activeSkybox !== skybox) {
                activeSkybox.setVisible(false);
                const prevId = Array.from(skyboxes.entries()).find(([_, s]) => s === activeSkybox)?.[0];
                if (prevId) {
                    const prevInfo = backgrounds.get(prevId);
                    if (prevInfo) {
                        prevInfo.visible = false;
                    }
                }
            }
            activeSkybox = skybox;
            scene.forceRender = true;
        }
        
        return backgroundInfo;
    };
    
    // Import background cubemap from file picker
    events.function('background.import', async () => {
        try {
            const handles = await window.showOpenFilePicker({
                id: 'BackgroundCubemapImport',
                multiple: false,
                excludeAcceptAllOption: false,
                types: [
                    {
                        description: 'Cubemap Image',
                        accept: {
                            'image/png': ['.png'],
                            'image/jpeg': ['.jpg', '.jpeg'],
                            'image/webp': ['.webp']
                        }
                    }
                ]
            });

            if (!handles || !handles[0]) {
                return;
            }

            const file = await handles[0].getFile();
            await importCubemapFromFile(file, false);
        } catch (error) {
            if (error instanceof Error && error.name !== 'AbortError') {
                console.error('Failed to import background:', error);
                await events.invoke('showPopup', {
                    type: 'error',
                    header: 'Import Failed',
                    message: `Failed to import cubemap: ${error.message ?? String(error)}`
                });
            } else if (!(error instanceof Error) && (error as any)?.name !== 'AbortError') {
                console.error('Failed to import background:', error);
                await events.invoke('showPopup', {
                    type: 'error',
                    header: 'Import Failed',
                    message: `Failed to import cubemap: ${String(error)}`
                });
            }
        }
    });

    // Handle background visibility
    events.on('background.visibility', ({ id, visible }: { id: string, visible: boolean }) => {
        const backgroundInfo = backgrounds.get(id);
        const skybox = skyboxes.get(id);

        if (!backgroundInfo || !skybox) {
            return;
        }

        backgroundInfo.visible = visible;
        skybox.setVisible(visible);

        // If this background is being shown, hide others
        if (visible) {
            if (activeSkybox && activeSkybox !== skybox) {
                activeSkybox.setVisible(false);
                const prevId = Array.from(skyboxes.entries()).find(([_, s]) => s === activeSkybox)?.[0];
                if (prevId) {
                    const prevInfo = backgrounds.get(prevId);
                    if (prevInfo) {
                        prevInfo.visible = false;
                    }
                }
            }
            activeSkybox = skybox;
        } else {
            if (activeSkybox === skybox) {
                activeSkybox = null;
            }
        }

        scene.forceRender = true;
    });

    // Handle background removal
    events.on('background.remove', (id: string) => {
        const backgroundInfo = backgrounds.get(id);
        const skybox = skyboxes.get(id);

        if (!backgroundInfo || !skybox) {
            return;
        }

        // Remove from scene
        skybox.destroy();
        scene.remove(skybox);

        // Clean up
        if (backgroundInfo.texture) {
            backgroundInfo.texture.destroy();
        }
        backgrounds.delete(id);
        skyboxes.delete(id);

        if (activeSkybox === skybox) {
            activeSkybox = null;
        }

        // Fire event to remove from UI
        events.fire('background.removed', id);

        scene.forceRender = true;
    });
    
    // Import background cubemap from File object (for programmatic import, e.g., from SOG4D)
    events.function('background.importFromFile', async (file: File) => {
        try {
            await importCubemapFromFile(file, false);
        } catch (error) {
            if (error instanceof Error && error.name !== 'AbortError') {
                console.error('Failed to import background from file:', error);
                throw error;
            }
        }
    });
    
    // Auto-show background by filename
    events.function('background.autoShow', async (filename: string) => {
        // Find background by filename
        const backgroundInfo = Array.from(backgrounds.values()).find(bg => bg.name === filename);
        if (backgroundInfo) {
            events.fire('background.visibility', { id: backgroundInfo.id, visible: true });
        } else {
            console.warn(`Background '${filename}' not found for auto-show`);
        }
    });
};

export { registerBackgroundEvents };
