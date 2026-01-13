import { Camera } from './camera';

// Extend DeviceOrientationEvent type for iOS permission API
interface DeviceOrientationEventWithPermission extends DeviceOrientationEvent {
    requestPermission?: () => Promise<'granted' | 'denied'>;
}

declare const DeviceOrientationEvent: {
    prototype: DeviceOrientationEvent;
    new(type: string, eventInitDict?: DeviceOrientationEventInit): DeviceOrientationEvent;
    requestPermission?: () => Promise<'granted' | 'denied'>;
};

class GyroscopeController {
    private camera: Camera;
    private _enabled: boolean = false;
    private _supported: boolean = false;

    // Initial device orientation (for calibration)
    private initialGamma: number | null = null;
    private initialBeta: number | null = null;

    // Initial camera orientation (when gyroscope was enabled)
    private initialCameraAzim: number = 0;
    private initialCameraElev: number = 0;

    // Smoothing factor (0-1, higher = less smoothing)
    private smoothingFactor: number = 0.3;

    // Current smoothed values
    private smoothedGamma: number = 0;
    private smoothedBeta: number = 0;

    private listener: ((event: DeviceOrientationEvent) => void) | null = null;

    constructor(camera: Camera) {
        this.camera = camera;

        // Check if DeviceOrientationEvent is supported
        this._supported = typeof window !== 'undefined' && 'DeviceOrientationEvent' in window;
    }

    get enabled(): boolean {
        return this._enabled;
    }

    get supported(): boolean {
        return this._supported;
    }

    /**
     * Check if permission is required (iOS 13+)
     */
    private needsPermission(): boolean {
        return typeof DeviceOrientationEvent.requestPermission === 'function';
    }

    /**
     * Request permission for device orientation (iOS 13+)
     * Must be called from a user gesture (click/tap)
     */
    async requestPermission(): Promise<boolean> {
        if (!this.needsPermission()) {
            return true;
        }

        try {
            const permission = await DeviceOrientationEvent.requestPermission();
            return permission === 'granted';
        } catch (error) {
            console.error('Failed to request device orientation permission:', error);
            return false;
        }
    }

    /**
     * Enable gyroscope control
     * Must be called from a user gesture on iOS
     */
    async enable(): Promise<boolean> {
        if (!this._supported) {
            console.warn('DeviceOrientationEvent is not supported on this device');
            return false;
        }

        if (this._enabled) {
            return true;
        }

        // Request permission if needed (iOS)
        if (this.needsPermission()) {
            const granted = await this.requestPermission();
            if (!granted) {
                console.warn('Device orientation permission denied');
                return false;
            }
        }

        // Store initial camera orientation (use actual value, not target)
        const currentAzimElev = this.camera.azimElevTween.value;
        this.initialCameraAzim = currentAzimElev.azim;
        this.initialCameraElev = currentAzimElev.elev;

        // Reset calibration
        this.initialGamma = null;
        this.initialBeta = null;

        // Create and store the listener
        this.listener = this.handleOrientation.bind(this);
        window.addEventListener('deviceorientation', this.listener);

        this._enabled = true;
        console.log('Gyroscope control enabled');

        return true;
    }

    /**
     * Disable gyroscope control
     */
    disable(): void {
        if (!this._enabled || !this.listener) {
            return;
        }

        window.removeEventListener('deviceorientation', this.listener);
        this.listener = null;
        this._enabled = false;

        console.log('Gyroscope control disabled');
    }

    /**
     * Toggle gyroscope control
     */
    async toggle(): Promise<boolean> {
        if (this._enabled) {
            this.disable();
            return false;
        } else {
            return await this.enable();
        }
    }

    /**
     * Recalibrate the gyroscope (set current orientation as reference)
     */
    recalibrate(): void {
        this.initialGamma = null;
        this.initialBeta = null;
        // Use actual value, not target
        const currentAzimElev = this.camera.azimElevTween.value;
        this.initialCameraAzim = currentAzimElev.azim;
        this.initialCameraElev = currentAzimElev.elev;
    }

    /**
     * Handle device orientation event
     */
    private handleOrientation(event: DeviceOrientationEvent): void {
        if (!this._enabled) return;

        const gamma = event.gamma; // Y-axis rotation (left-right tilt) -90 to 90
        const beta = event.beta;   // X-axis rotation (front-back tilt) -180 to 180

        if (gamma === null || beta === null) {
            return;
        }

        // Calibrate on first valid event
        if (this.initialGamma === null || this.initialBeta === null) {
            this.initialGamma = gamma;
            this.initialBeta = beta;
            this.smoothedGamma = gamma;
            this.smoothedBeta = beta;
            return;
        }

        // Apply smoothing (low-pass filter) to reduce jitter
        // Gamma and Beta are already in a bounded range, no wrap-around needed
        const gammaDiff = gamma - this.smoothedGamma;
        const betaDiff = beta - this.smoothedBeta;

        this.smoothedGamma = this.smoothedGamma + this.smoothingFactor * gammaDiff;
        this.smoothedBeta = this.smoothedBeta + this.smoothingFactor * betaDiff;

        // Calculate delta from initial position
        const deltaGamma = this.smoothedGamma - this.initialGamma;
        const deltaBeta = this.smoothedBeta - this.initialBeta;

        // Map device orientation to camera:
        // - Gamma (left-right tilt): affects camera azimuth (horizontal rotation)
        // - Beta (front-back tilt): affects camera elevation (vertical rotation)
        // Note: When device tilts forward (beta increases), user expects to see model top (elev increases)
        const newAzim = this.initialCameraAzim - deltaGamma;
        const newElev = this.initialCameraElev - deltaBeta;

        // Update camera orientation with minimal damping for responsiveness
        this.camera.setAzimElev(newAzim, newElev, 0.5);
    }

    /**
     * Clean up resources
     */
    destroy(): void {
        this.disable();
    }
}

export { GyroscopeController };
