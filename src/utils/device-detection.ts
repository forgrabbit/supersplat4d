/**
 * Detect if the current device is a mobile device
 * Uses multiple detection methods for better accuracy
 */
export const isMobileDevice = (): boolean => {
    // Method 1: Use User-Agent Client Hints API (modern browsers)
    // TypeScript doesn't have types for userAgentData yet, so we use type assertion
    const nav = navigator as any;
    if (nav.userAgentData?.mobile) {
        return true;
    }

    // Method 2: Use CSS media queries
    const hasCoarsePointer = window.matchMedia('(pointer: coarse)').matches;
    const lacksHoverSupport = window.matchMedia('(hover: none)').matches;
    if (hasCoarsePointer && lacksHoverSupport) {
        return true;
    }

    // Method 3: Check for fine pointer (desktop usually has this)
    const hasFinePointer = window.matchMedia('(pointer: fine)').matches;
    if (hasFinePointer) {
        return false;
    }

    // Method 4: Fallback to User Agent string
    return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
};
