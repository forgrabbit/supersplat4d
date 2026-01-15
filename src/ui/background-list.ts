import { Container, Element as PcuiElement, Label } from '@playcanvas/pcui';

import { Events } from '../events';
import deleteSvg from './svg/delete.svg';
import hiddenSvg from './svg/hidden.svg';
import shownSvg from './svg/shown.svg';

const createSvg = (svgString: string) => {
    const decodedStr = decodeURIComponent(svgString.substring('data:image/svg+xml,'.length));
    return new DOMParser().parseFromString(decodedStr, 'image/svg+xml').documentElement;
};

interface BackgroundInfo {
    id: string;
    name: string;
    texture: any; // PlayCanvas Texture
    visible: boolean;
}

class BackgroundItem extends Container {
    getName: () => string;
    setName: (value: string) => void;
    getVisible: () => boolean;
    setVisible: (value: boolean) => void;
    destroy: () => void;
    backgroundInfo: BackgroundInfo;

    constructor(backgroundInfo: BackgroundInfo, args = {}) {
        const classList = ['background-item'];
        if (backgroundInfo.visible) {
            classList.push('visible');
        }
        
        args = {
            ...args,
            class: classList
        };

        super(args);

        this.backgroundInfo = backgroundInfo;

        const text = new Label({
            class: 'background-item-text',
            text: backgroundInfo.name
        });

        const visible = new PcuiElement({
            dom: createSvg(shownSvg),
            class: 'background-item-visible',
            hidden: !backgroundInfo.visible
        });

        const invisible = new PcuiElement({
            dom: createSvg(hiddenSvg),
            class: 'background-item-visible',
            hidden: backgroundInfo.visible
        });

        const remove = new PcuiElement({
            dom: createSvg(deleteSvg),
            class: 'background-item-delete'
        });

        this.append(text);
        this.append(visible);
        this.append(invisible);
        this.append(remove);

        this.getName = () => {
            return text.value;
        };

        this.setName = (value: string) => {
            text.value = value;
        };

        this.getVisible = () => {
            return this.class.contains('visible');
        };

        this.setVisible = (value: boolean) => {
            if (value !== this.visible) {
                visible.hidden = !value;
                invisible.hidden = value;
                if (value) {
                    this.class.add('visible');
                    this.emit('visible', this);
                } else {
                    this.class.remove('visible');
                    this.emit('invisible', this);
                }
            }
        };

        const toggleVisible = (event: MouseEvent) => {
            event.stopPropagation();
            this.visible = !this.visible;
        };

        const handleRemove = (event: MouseEvent) => {
            event.stopPropagation();
            this.emit('removeClicked', this);
        };

        // handle clicks
        visible.dom.addEventListener('click', toggleVisible);
        invisible.dom.addEventListener('click', toggleVisible);
        remove.dom.addEventListener('click', handleRemove);

        this.destroy = () => {
            visible.dom.removeEventListener('click', toggleVisible);
            invisible.dom.removeEventListener('click', toggleVisible);
            remove.dom.removeEventListener('click', handleRemove);
        };
    }

    set name(value: string) {
        this.setName(value);
    }

    get name() {
        return this.getName();
    }

    set visible(value) {
        this.setVisible(value);
    }

    get visible() {
        return this.getVisible();
    }
}

class BackgroundList extends Container {
    private items = new Map<string, BackgroundItem>();
    private activeBackgroundId: string | null = null;

    constructor(events: Events, args = {}) {
        args = {
            ...args,
            class: 'background-list'
        };

        super(args);

        // Listen for background import
        events.on('background.added', (backgroundInfo: BackgroundInfo) => {
            const item = new BackgroundItem(backgroundInfo);
            this.append(item);
            this.items.set(backgroundInfo.id, item);

            item.on('visible', () => {
                // Hide other backgrounds
                this.items.forEach((otherItem, otherId) => {
                    if (otherId !== backgroundInfo.id) {
                        otherItem.visible = false;
                        events.fire('background.visibility', { id: otherId, visible: false });
                    }
                });
                this.activeBackgroundId = backgroundInfo.id;
                events.fire('background.visibility', { id: backgroundInfo.id, visible: true });
            });

            item.on('invisible', () => {
                if (this.activeBackgroundId === backgroundInfo.id) {
                    this.activeBackgroundId = null;
                }
                events.fire('background.visibility', { id: backgroundInfo.id, visible: false });
            });
        });

        events.on('background.removed', (id: string) => {
            const item = this.items.get(id);
            if (item) {
                this.remove(item);
                item.destroy();
                this.items.delete(id);
                if (this.activeBackgroundId === id) {
                    this.activeBackgroundId = null;
                }
            }
        });

        // Note: Click handling is done through item's visible/invisible events
        // No need for separate click handler here

        this.on('removeClicked', async (item: BackgroundItem) => {
            const backgroundInfo = item.backgroundInfo;
            const result = await events.invoke('showPopup', {
                type: 'yesno',
                header: 'Remove Background',
                message: `Are you sure you want to remove '${backgroundInfo.name}'? This operation can not be undone.`
            });

            if (result?.action === 'yes') {
                events.fire('background.remove', backgroundInfo.id);
            }
        });
    }

    protected _onAppendChild(element: PcuiElement): void {
        super._onAppendChild(element);

        if (element instanceof BackgroundItem) {
            element.on('removeClicked', () => {
                this.emit('removeClicked', element);
            });
        }
    }

    protected _onRemoveChild(element: PcuiElement): void {
        if (element instanceof BackgroundItem) {
            element.unbind('removeClicked');
        }

        super._onRemoveChild(element);
    }
}

export { BackgroundList, BackgroundItem, BackgroundInfo };
