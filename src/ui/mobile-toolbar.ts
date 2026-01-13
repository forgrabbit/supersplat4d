import { Button, Container, Element, Label } from '@playcanvas/pcui';

import { Events } from '../events';
import { localize } from './localization';
import colorPanelSvg from './svg/color-panel.svg';
import gyroscopeSvg from './svg/gyroscope.svg';
import { Tooltips } from './tooltips';

const createSvg = (svgString: string) => {
    const decodedStr = decodeURIComponent(svgString.substring('data:image/svg+xml,'.length));
    return new DOMParser().parseFromString(decodedStr, 'image/svg+xml').documentElement;
};

class MobileToolbar extends Container {
    private expanded: boolean = false;
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

        // Floating button (左上角浮标)
        const floatingButton = new Button({
            id: 'mobile-toolbar-float',
            class: 'mobile-toolbar-float'
        });
        floatingButton.dom.innerHTML = '☰'; // Menu icon

        // Expanded container (展开的按钮列表)
        const expandedContainer = new Container({
            id: 'mobile-toolbar-expanded',
            class: 'mobile-toolbar-expanded',
            hidden: true
        });

        // Gyroscope button
        const gyroscopeButton = new Button({
            id: 'mobile-toolbar-gyroscope',
            class: 'mobile-toolbar-button'
        });
        const gyroscopeSvgElement = createSvg(gyroscopeSvg);
        // Set explicit size for gyroscope icon
        gyroscopeSvgElement.style.width = '24px';
        gyroscopeSvgElement.style.height = '24px';
        gyroscopeSvgElement.style.minWidth = '24px';
        gyroscopeSvgElement.style.minHeight = '24px';
        gyroscopeButton.dom.appendChild(gyroscopeSvgElement);
        const gyroscopeLabel = document.createElement('span');
        gyroscopeLabel.textContent = 'Gyroscope';
        gyroscopeButton.dom.appendChild(gyroscopeLabel);

        // Translate button
        const translateButton = new Button({
            id: 'mobile-toolbar-translate',
            class: 'mobile-toolbar-button',
            icon: 'E111'
        });
        const translateLabel = document.createElement('span');
        translateLabel.textContent = 'Translate';
        translateButton.dom.appendChild(translateLabel);

        // Rotate button
        const rotateButton = new Button({
            id: 'mobile-toolbar-rotate',
            class: 'mobile-toolbar-button',
            icon: 'E113'
        });
        const rotateLabel = document.createElement('span');
        rotateLabel.textContent = 'Rotate';
        rotateButton.dom.appendChild(rotateLabel);

        // High Precision button
        const highPrecisionButton = new Button({
            id: 'mobile-toolbar-high-precision',
            class: 'mobile-toolbar-button'
        });
        const highPrecisionSvg = createSvg(colorPanelSvg);
        // Set explicit size for high precision icon
        highPrecisionSvg.style.width = '24px';
        highPrecisionSvg.style.height = '24px';
        highPrecisionSvg.style.minWidth = '24px';
        highPrecisionSvg.style.minHeight = '24px';
        highPrecisionButton.dom.appendChild(highPrecisionSvg);
        const highPrecisionLabel = document.createElement('span');
        highPrecisionLabel.textContent = 'High Precision';
        highPrecisionButton.dom.appendChild(highPrecisionLabel);

        // Append buttons to expanded container
        expandedContainer.append(gyroscopeButton);
        expandedContainer.append(translateButton);
        expandedContainer.append(rotateButton);
        expandedContainer.append(highPrecisionButton);

        // Append to main container
        this.append(floatingButton);
        this.append(expandedContainer);

        // Helper function to close the expanded toolbar
        const closeExpanded = () => {
            this.expanded = false;
            expandedContainer.hidden = true;
            // Remove mobile-scene-panel class first
            scenePanel.dom.classList.remove('mobile-scene-panel');
            // Force hide by setting display: none with !important
            scenePanel.dom.style.setProperty('display', 'none', 'important');
            scenePanel.hidden = true;
            // Clean up positioning styles
            scenePanel.dom.style.removeProperty('top');
            scenePanel.dom.style.removeProperty('left');
            scenePanel.dom.style.removeProperty('width');
            scenePanel.dom.style.removeProperty('max-width');
            // Remove click outside handler
            if (this.clickOutsideHandler) {
                document.removeEventListener('pointerdown', this.clickOutsideHandler);
                this.clickOutsideHandler = null;
            }
        };

        // Event handlers
        floatingButton.on('click', () => {
            this.expanded = !this.expanded;
            expandedContainer.hidden = !this.expanded;
            // Show/hide scene panel below the toolbar when expanded
            if (this.expanded) {
                scenePanel.dom.classList.add('mobile-scene-panel');
                // Force display to block to override PCUI's hidden property and .collapsed rule
                scenePanel.dom.style.setProperty('display', 'block', 'important');
                scenePanel.hidden = false;
                // Position scene panel below the expanded toolbar
                // Use requestAnimationFrame to ensure DOM is updated before calculating height
                // Use double requestAnimationFrame to ensure DOM is fully updated
                requestAnimationFrame(() => {
                    requestAnimationFrame(() => {
                        // Wait for expanded container to be visible and measure its height and width
                        const toolbarHeight = expandedContainer.dom.offsetHeight || 280;
                        const toolbarWidth = expandedContainer.dom.offsetWidth || 200;
                        const topPosition = 60 + toolbarHeight + 8 + 10; // Add 10px extra spacing
                        // Set all positioning styles directly to ensure they take effect
                        scenePanel.dom.style.top = `${topPosition}px`;
                        scenePanel.dom.style.left = '12px';
                        // Match the width of the expanded toolbar container
                        scenePanel.dom.style.width = `${toolbarWidth}px`;
                        scenePanel.dom.style.maxWidth = `${toolbarWidth}px`;
                        // Ensure display is still set after all operations (override .collapsed rule)
                        scenePanel.dom.style.setProperty('display', 'block', 'important');
                        console.log('Scene panel positioned at:', topPosition, 'toolbar height:', toolbarHeight, 'toolbar width:', toolbarWidth, 'hidden:', scenePanel.hidden, 'display:', window.getComputedStyle(scenePanel.dom).display);
                    });
                });
                // Add click outside handler when expanded
                // Use setTimeout to avoid immediate trigger from the current click
                setTimeout(() => {
                    this.clickOutsideHandler = (event: PointerEvent) => {
                        const target = event.target as Node;
                        // Check if click is outside both toolbar and scene panel
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

        gyroscopeButton.on('click', () => {
            events.fire('camera.toggleGyroscope');
        });

        translateButton.on('click', () => {
            events.fire('tool.move');
        });

        rotateButton.on('click', () => {
            events.fire('tool.rotate');
        });

        highPrecisionButton.on('click', () => {
            const currentValue = events.invoke('camera.highPrecision') ?? true;
            events.fire('camera.sethighPrecision', !currentValue);
        });

        // Update button states based on events
        events.on('camera.gyroscope', (enabled: boolean) => {
            gyroscopeButton.class[enabled ? 'add' : 'remove']('active');
        });

        events.on('tool.activated', (toolName: string) => {
            translateButton.class[toolName === 'move' ? 'add' : 'remove']('active');
            rotateButton.class[toolName === 'rotate' ? 'add' : 'remove']('active');
        });

        events.on('camera.highPrecision', (enabled: boolean) => {
            highPrecisionButton.class[enabled ? 'add' : 'remove']('active');
        });

        // Register tooltips
        tooltips.register(floatingButton, 'Menu', 'right');
        tooltips.register(gyroscopeButton, localize('tooltip.right-toolbar.gyroscope'), 'right');
        tooltips.register(translateButton, localize('tooltip.bottom-toolbar.translate'), 'right');
        tooltips.register(rotateButton, localize('tooltip.bottom-toolbar.rotate'), 'right');
        tooltips.register(highPrecisionButton, localize('panel.view-options.high-precision'), 'right');
    }

    // Cleanup method to remove event listeners
    destroy(): void {
        // Remove click outside handler if it exists
        if (this.clickOutsideHandler) {
            document.removeEventListener('pointerdown', this.clickOutsideHandler);
            this.clickOutsideHandler = null;
        }
        super.destroy();
    }
}

export { MobileToolbar };
