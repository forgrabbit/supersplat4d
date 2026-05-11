import { loadCubemapFromFile } from './cubemap-loader';
import { Events } from './events';
import { Scene } from './scene';
import { Skybox } from './skybox';
import { BackgroundInfo } from './ui/background-list';

let nextBackgroundId = 1;
const backgrounds = new Map<string, BackgroundInfo>();
const skyboxes = new Map<string, Skybox>();
let activeSkybox: Skybox | null = null;

const registerBackgroundEvents = (scene: Scene, events: Events) => {
    const removeBackground = (id: string) => {
        const backgroundInfo = backgrounds.get(id);
        const skybox = skyboxes.get(id);

        if (!backgroundInfo || !skybox) {
            return;
        }

        skybox.destroy();
        scene.remove(skybox);

        if (backgroundInfo.texture) {
            backgroundInfo.texture.destroy();
        }

        backgrounds.delete(id);
        skyboxes.delete(id);

        if (activeSkybox === skybox) {
            activeSkybox = null;
        }

        events.fire('background.removed', id);
        scene.forceRender = true;
    };

    const clearBackgrounds = () => {
        Array.from(backgrounds.keys()).forEach((id) => {
            removeBackground(id);
        });
    };

    const importCubemapFromFile = async (file: File, autoShow = false) => {
        const filename = file.name;
        const device = scene.graphicsDevice;
        const cubemapTexture = await loadCubemapFromFile(device, file);

        const id = `background_${nextBackgroundId++}`;
        const backgroundInfo: BackgroundInfo = {
            id,
            name: filename,
            texture: cubemapTexture,
            visible: false
        };

        const skybox = new Skybox();
        skybox.setTexture(cubemapTexture);
        scene.add(skybox);
        skybox.setVisible(autoShow);

        backgrounds.set(id, backgroundInfo);
        skyboxes.set(id, skybox);
        events.fire('background.added', backgroundInfo);

        if (autoShow) {
            if (activeSkybox && activeSkybox !== skybox) {
                activeSkybox.setVisible(false);
                const prevId = Array.from(skyboxes.entries()).find(([_, s]) => s === activeSkybox)?.[0];
                if (prevId) {
                    const prevInfo = backgrounds.get(prevId);
                    if (prevInfo) {
                        prevInfo.visible = false;
                        events.fire('background.visibility', { id: prevId, visible: false });
                    }
                }
            }

            backgroundInfo.visible = true;
            skybox.setVisible(true);
            activeSkybox = skybox;
            events.fire('background.visibility', { id: backgroundInfo.id, visible: true });
            scene.forceRender = true;
        }

        return backgroundInfo;
    };

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
            await importCubemapFromFile(file, true);
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

    events.on('background.visibility', ({ id, visible }: { id: string, visible: boolean }) => {
        const backgroundInfo = backgrounds.get(id);
        const skybox = skyboxes.get(id);

        if (!backgroundInfo || !skybox) {
            return;
        }

        backgroundInfo.visible = visible;
        skybox.setVisible(visible);

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
        } else if (activeSkybox === skybox) {
            activeSkybox = null;
        }

        scene.forceRender = true;
    });

    events.on('background.remove', (id: string) => {
        removeBackground(id);
    });

    events.function('background.clear', () => {
        clearBackgrounds();
    });

    events.on('scene.clear', () => {
        clearBackgrounds();
    });

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

    events.function('background.autoShow', (filename: string) => {
        const backgroundInfo = Array.from(backgrounds.values()).find(bg => bg.name === filename);
        if (backgroundInfo) {
            events.fire('background.visibility', { id: backgroundInfo.id, visible: true });
        } else {
            console.warn(`Background '${filename}' not found for auto-show`);
        }
    });
};

export { registerBackgroundEvents };
