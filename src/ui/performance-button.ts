import { Container } from '@playcanvas/pcui';
import { Events } from '../events';
import { isMobileDevice } from '../utils/device-detection';
import performanceIconSvg from './svg/performance-icon.svg';

const createSvg = (svgString: string) => {
    const decodedStr = decodeURIComponent(svgString.substring('data:image/svg+xml,'.length));
    return new DOMParser().parseFromString(decodedStr, 'image/svg+xml').documentElement;
};

const STORAGE_KEY = 'performance-button-position';
const EDGE_MARGIN = 20; // Minimum visible pixels when near edge

/**
 * Performance monitoring floating button (draggable)
 */
class PerformanceButton extends Container {
    private button: HTMLButtonElement;
    private isDragging = false;
    private isMonitoring = false;
    private dragStartX = 0;
    private dragStartY = 0;
    private buttonStartX = 0;
    private buttonStartY = 0;

    constructor(events: Events, args = {}) {
        super({
            id: 'performance-button',
            class: 'performance-button',
            ...args
        });

        // Create button with icon
        this.button = document.createElement('button');
        this.button.className = 'performance-button-icon';
        this.button.title = 'Performance Monitor (Drag to move)';
        this.button.type = 'button';
        
        const svg = createSvg(performanceIconSvg);
        this.button.appendChild(svg);

        this.dom.appendChild(this.button);

        // Ensure button is visible
        this.dom.style.display = 'block';
        this.dom.style.visibility = 'visible';
        this.dom.style.opacity = '1';

        // Load saved position or use default (bottom-right for desktop, bottom-left for mobile)
        this.loadPosition();
        
        // Debug: log button creation
        const isMobile = isMobileDevice();
        console.log(`[PerformanceButton] Created, isMobile=${isMobile}, position: left=${this.dom.style.left}, top=${this.dom.style.top}`);

        // Setup drag functionality
        this.setupDragHandlers();

        // Handle click (only if not dragging)
        this.button.addEventListener('click', (e) => {
            // Prevent click if we just finished dragging
            if (this.isDragging) {
                e.preventDefault();
                return;
            }

            if (!this.isMonitoring) {
                // Start monitoring
                events.fire('performance.start');
                this.isMonitoring = true;
                this.button.classList.add('active');
            } else {
                // Stop monitoring and save
                events.fire('performance.stop');
                this.isMonitoring = false;
                this.button.classList.remove('active');
            }
        });

        // Update position on window resize
        window.addEventListener('resize', () => {
            this.constrainPosition();
        });
    }

    private loadPosition() {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved) {
            try {
                const { x, y, screenWidth, screenHeight } = JSON.parse(saved);
                // Check if screen size changed significantly (likely different device)
                const currentWidth = window.innerWidth;
                const currentHeight = window.innerHeight;
                const widthDiff = Math.abs(currentWidth - (screenWidth || currentWidth)) / (screenWidth || currentWidth);
                const heightDiff = Math.abs(currentHeight - (screenHeight || currentHeight)) / (screenHeight || currentHeight);
                
                // If screen size changed by more than 30%, reset position (likely different device)
                if (widthDiff > 0.3 || heightDiff > 0.3) {
                    this.setDefaultPosition();
                } else {
                    this.dom.style.left = `${x}px`;
                    this.dom.style.top = `${y}px`;
                    this.constrainPosition();
                }
            } catch (e) {
                this.setDefaultPosition();
            }
        } else {
            this.setDefaultPosition();
        }
    }

    private setDefaultPosition() {
        const padding = 20;
        const buttonSize = 60; // 40px button + 20px padding
        const isMobile = isMobileDevice();
        
        if (isMobile) {
            // Mobile: bottom-left corner (avoid conflict with menu button in top-left)
            this.dom.style.left = `${padding}px`;
            this.dom.style.top = `${window.innerHeight - buttonSize - padding}px`;
        } else {
            // Desktop: bottom-right corner
            this.dom.style.left = `${window.innerWidth - buttonSize - padding}px`;
            this.dom.style.top = `${window.innerHeight - buttonSize - padding}px`;
        }
    }

    private savePosition() {
        const rect = this.dom.getBoundingClientRect();
        localStorage.setItem(STORAGE_KEY, JSON.stringify({
            x: rect.left,
            y: rect.top,
            screenWidth: window.innerWidth,
            screenHeight: window.innerHeight
        }));
    }

    private constrainPosition() {
        const rect = this.dom.getBoundingClientRect();
        const buttonWidth = rect.width;
        const buttonHeight = rect.height;
        
        let x = rect.left;
        let y = rect.top;
        
        // Constrain to viewport with minimum visible edge
        const minX = -buttonWidth + EDGE_MARGIN;
        const maxX = window.innerWidth - EDGE_MARGIN;
        const minY = 0;
        const maxY = window.innerHeight - EDGE_MARGIN;
        
        x = Math.max(minX, Math.min(maxX, x));
        y = Math.max(minY, Math.min(maxY, y));
        
        this.dom.style.left = `${x}px`;
        this.dom.style.top = `${y}px`;
    }

    private setupDragHandlers() {
        let dragMoved = false;

        const onPointerDown = (e: PointerEvent) => {
            // Only left mouse button or touch
            if (e.button !== 0 && e.pointerType === 'mouse') return;

            this.isDragging = true;
            dragMoved = false;
            
            const rect = this.dom.getBoundingClientRect();
            this.dragStartX = e.clientX;
            this.dragStartY = e.clientY;
            this.buttonStartX = rect.left;
            this.buttonStartY = rect.top;
            
            this.button.classList.add('dragging');
            e.preventDefault();
            e.stopPropagation();

            // Capture pointer
            this.button.setPointerCapture(e.pointerId);
        };

        const onPointerMove = (e: PointerEvent) => {
            if (!this.isDragging) return;

            const deltaX = e.clientX - this.dragStartX;
            const deltaY = e.clientY - this.dragStartY;
            
            // Consider it a drag if moved more than 5 pixels
            if (Math.abs(deltaX) > 5 || Math.abs(deltaY) > 5) {
                dragMoved = true;
            }
            
            const newX = this.buttonStartX + deltaX;
            const newY = this.buttonStartY + deltaY;
            
            this.dom.style.left = `${newX}px`;
            this.dom.style.top = `${newY}px`;
            
            e.preventDefault();
            e.stopPropagation();
        };

        const onPointerUp = (e: PointerEvent) => {
            if (!this.isDragging) return;

            this.isDragging = false;
            this.button.classList.remove('dragging');
            
            // Constrain final position
            this.constrainPosition();
            
            // Save position
            this.savePosition();
            
            // If we actually dragged, prevent the click event
            if (dragMoved) {
                e.preventDefault();
                e.stopPropagation();
            }

            // Release pointer capture
            this.button.releasePointerCapture(e.pointerId);
        };

        this.button.addEventListener('pointerdown', onPointerDown);
        this.button.addEventListener('pointermove', onPointerMove);
        this.button.addEventListener('pointerup', onPointerUp);
        this.button.addEventListener('pointercancel', onPointerUp);

        // Prevent context menu on drag
        this.button.addEventListener('contextmenu', (e) => {
            if (dragMoved) {
                e.preventDefault();
            }
        });
    }
}

export { PerformanceButton };
