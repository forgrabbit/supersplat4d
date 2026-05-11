import { Button, Container, Label } from '@playcanvas/pcui';

import { Events } from '../events';
import type { PublicModel } from '../public-models';
import { localize } from './localization';
import gyroscopeSvg from './svg/gyroscope.svg';
import openSvg from './svg/open.svg';
import { Tooltips } from './tooltips';

const createSvg = (svgString: string) => {
    const decodedStr = decodeURIComponent(svgString.substring('data:image/svg+xml,'.length));
    return new DOMParser().parseFromString(decodedStr, 'image/svg+xml').documentElement;
};

const setIconSize = (element: HTMLElement) => {
    element.style.width = '24px';
    element.style.height = '24px';
    element.style.minWidth = '24px';
    element.style.minHeight = '24px';
};

class MobileToolbar extends Container {
    private expanded = false;
    private modelsExpanded = false;
    private clickOutsideHandler: ((event: PointerEvent) => void) | null = null;

    constructor(events: Events, tooltips: Tooltips, scenePanel: Container, args = {}) {
        args = {
            ...args,
            id: 'mobile-toolbar'
        };

        super(args);

        this.dom.addEventListener('pointerdown', (event) => {
            event.stopPropagation();
        });

        const floatingButton = new Button({
            id: 'mobile-toolbar-float',
            class: 'mobile-toolbar-float'
        });
        floatingButton.dom.innerHTML = '&#9776;';

        const expandedContainer = new Container({
            id: 'mobile-toolbar-expanded',
            class: 'mobile-toolbar-expanded',
            hidden: true
        });

        const actionsRow = new Container({
            class: 'mobile-toolbar-actions'
        });

        const positionScenePanel = () => {
            if (!this.expanded) {
                return;
            }

            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    const toolbarHeight = expandedContainer.dom.offsetHeight || 88;
                    const toolbarWidth = expandedContainer.dom.offsetWidth || 208;
                    const topPosition = 60 + toolbarHeight + 18;

                    scenePanel.dom.style.top = `${topPosition}px`;
                    scenePanel.dom.style.left = '12px';
                    scenePanel.dom.style.width = `${toolbarWidth}px`;
                    scenePanel.dom.style.maxWidth = `${toolbarWidth}px`;
                    scenePanel.dom.style.setProperty('display', 'block', 'important');
                });
            });
        };

        const gyroscopeButton = new Button({
            id: 'mobile-toolbar-gyroscope',
            class: ['mobile-toolbar-button', 'mobile-toolbar-button-compact']
        });
        const gyroscopeIcon = createSvg(gyroscopeSvg);
        setIconSize(gyroscopeIcon);
        gyroscopeButton.dom.appendChild(gyroscopeIcon);
        const gyroscopeLabel = document.createElement('span');
        gyroscopeLabel.textContent = 'Gyroscope';
        gyroscopeButton.dom.appendChild(gyroscopeLabel);

        const modelsButton = new Button({
            id: 'mobile-toolbar-models',
            class: ['mobile-toolbar-button', 'mobile-toolbar-button-compact']
        });
        const modelsIcon = createSvg(openSvg);
        setIconSize(modelsIcon);
        modelsButton.dom.appendChild(modelsIcon);
        const modelsLabel = document.createElement('span');
        modelsLabel.textContent = localize('menu.models');
        modelsButton.dom.appendChild(modelsLabel);

        const modelPopover = new Container({
            id: 'mobile-toolbar-model-popover',
            class: 'mobile-toolbar-model-popover',
            hidden: true
        });

        const modelListContainer = new Container({
            class: 'mobile-toolbar-model-list'
        });

        modelPopover.append(modelListContainer);
        actionsRow.append(gyroscopeButton);
        actionsRow.append(modelsButton);
        expandedContainer.append(actionsRow);
        expandedContainer.append(modelPopover);

        this.append(floatingButton);
        this.append(expandedContainer);

        let closeExpanded = () => {};

        const setModelListMessage = (text: string) => {
            modelListContainer.clear();
            modelListContainer.append(new Label({
                class: 'mobile-toolbar-model-empty',
                text
            }));
        };

        const setModelButtons = (models: PublicModel[]) => {
            modelListContainer.clear();

            if (models.length === 0) {
                setModelListMessage(localize('menu.models.none'));
                return;
            }

            models.forEach((model) => {
                const button = new Button({
                    class: 'mobile-toolbar-model-item'
                });
                const label = document.createElement('span');
                label.textContent = model.path;
                button.dom.appendChild(label);
                button.dom.title = model.path;

                button.on('click', async () => {
                    closeExpanded();
                    await events.invoke('models.load', model);
                });

                modelListContainer.append(button);
            });
        };

        const refreshModelList = async () => {
            setModelListMessage(localize('menu.models.loading'));

            try {
                const models = ((await events.invoke('models.list')) as PublicModel[]) ?? [];
                setModelButtons(models);
            } catch (error) {
                console.error('Failed to load public model list:', error);
                setModelListMessage(localize('menu.models.load-failed'));
            }
        };

        const setModelPopoverVisibility = (visible: boolean) => {
            this.modelsExpanded = visible;
            modelPopover.hidden = !visible;
            modelsButton.class[visible ? 'add' : 'remove']('active');
        };

        closeExpanded = () => {
            this.expanded = false;
            expandedContainer.hidden = true;
            setModelPopoverVisibility(false);

            scenePanel.dom.classList.remove('mobile-scene-panel');
            scenePanel.dom.style.setProperty('display', 'none', 'important');
            scenePanel.hidden = true;
            scenePanel.dom.style.removeProperty('top');
            scenePanel.dom.style.removeProperty('left');
            scenePanel.dom.style.removeProperty('width');
            scenePanel.dom.style.removeProperty('max-width');

            if (this.clickOutsideHandler) {
                document.removeEventListener('pointerdown', this.clickOutsideHandler);
                this.clickOutsideHandler = null;
            }
        };

        floatingButton.on('click', () => {
            this.expanded = !this.expanded;
            expandedContainer.hidden = !this.expanded;

            if (this.expanded) {
                scenePanel.dom.classList.add('mobile-scene-panel');
                scenePanel.dom.style.setProperty('display', 'block', 'important');
                scenePanel.hidden = false;
                positionScenePanel();

                setTimeout(() => {
                    this.clickOutsideHandler = (event: PointerEvent) => {
                        const target = event.target as Node;
                        const isInsideToolbar = this.dom.contains(target);
                        const isInsideScenePanel = scenePanel.dom.contains(target);

                        if (!isInsideToolbar && !isInsideScenePanel) {
                            closeExpanded();
                        }
                    };
                    document.addEventListener('pointerdown', this.clickOutsideHandler);
                }, 0);
            } else {
                closeExpanded();
            }
        });

        modelsButton.on('click', async () => {
            const visible = !this.modelsExpanded;
            setModelPopoverVisibility(visible);

            if (visible) {
                await refreshModelList();
            }
        });

        gyroscopeButton.on('click', () => {
            events.fire('camera.toggleGyroscope');
        });

        events.on('camera.gyroscope', (enabled: boolean) => {
            gyroscopeButton.class[enabled ? 'add' : 'remove']('active');
        });

        tooltips.register(floatingButton, 'Menu', 'right');
        tooltips.register(gyroscopeButton, localize('tooltip.right-toolbar.gyroscope'), 'right');
        tooltips.register(modelsButton, localize('menu.models'), 'right');
    }

    destroy(): void {
        if (this.clickOutsideHandler) {
            document.removeEventListener('pointerdown', this.clickOutsideHandler);
            this.clickOutsideHandler = null;
        }
        super.destroy();
    }
}

export { MobileToolbar };
