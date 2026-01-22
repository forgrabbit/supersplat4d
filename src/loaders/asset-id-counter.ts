/**
 * Shared asset ID counter for all loaders
 * This prevents URL conflicts when multiple loaders create local assets
 */

let globalAssetId = 0;

/**
 * Get next unique asset ID
 * Used to generate unique URLs for local assets (e.g., "local-asset-0", "local-asset-1")
 */
export const getNextAssetId = (): number => {
    return globalAssetId++;
};

/**
 * Reset the counter (mainly for testing)
 */
export const resetAssetIdCounter = (): void => {
    globalAssetId = 0;
};
