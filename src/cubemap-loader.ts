import {
    Texture,
    PIXELFORMAT_RGBA8,
    ADDRESS_CLAMP_TO_EDGE,
    FILTER_LINEAR
} from 'playcanvas';

/**
 * Load a cubemap texture from a single image file
 * The image should be in horizontal cross format: [ -X | +Z | +X | -Z ]
 *                                                      [ -Y |     | +Y ]
 * @param device Graphics device
 * @param image Image element or ImageData
 * @returns Cubemap texture
 */
const loadCubemapFromImage = (device: any, image: HTMLImageElement | ImageData): Texture => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    if (!ctx) {
        throw new Error('Failed to get canvas context');
    }

    let width: number;
    let height: number;
    let imageData: ImageData;

    if (image instanceof HTMLImageElement) {
        canvas.width = image.width;
        canvas.height = image.height;
        ctx.drawImage(image, 0, 0);
        imageData = ctx.getImageData(0, 0, image.width, image.height);
        width = image.width;
        height = image.height;
    } else {
        imageData = image;
        width = image.width;
        height = image.height;
    }

    // Assume horizontal cross layout
    // The image should be 4:3 aspect ratio (4 faces horizontally, 1 face above, 1 face below)
    // Or it could be 3:4 (vertical cross)
    // Common format: horizontal cross with 4 faces in a row, 1 above, 1 below
    // Layout: [ -X | +Z | +X | -Z ]
    //         [ -Y |     | +Y ]
    
    // Calculate face size
    // For horizontal cross: width = 4 * faceSize, height = 3 * faceSize
    // So faceSize = width / 4
    const faceSize = Math.floor(width / 4);
    
    // Create canvas for each face
    const faceCanvas = document.createElement('canvas');
    faceCanvas.width = faceSize;
    faceCanvas.height = faceSize;
    const faceCtx = faceCanvas.getContext('2d');
    
    if (!faceCtx) {
        throw new Error('Failed to get face canvas context');
    }

    // Extract each face
    const extractFace = (sx: number, sy: number): ImageData => {
        const faceImageData = faceCtx.createImageData(faceSize, faceSize);
        for (let y = 0; y < faceSize; y++) {
            for (let x = 0; x < faceSize; x++) {
                const srcX = sx + x;
                const srcY = sy + y;
                const srcIdx = (srcY * width + srcX) * 4;
                const dstIdx = (y * faceSize + x) * 4;
                faceImageData.data[dstIdx] = imageData.data[srcIdx];
                faceImageData.data[dstIdx + 1] = imageData.data[srcIdx + 1];
                faceImageData.data[dstIdx + 2] = imageData.data[srcIdx + 2];
                faceImageData.data[dstIdx + 3] = imageData.data[srcIdx + 3];
            }
        }
        return faceImageData;
    };

    // Extract faces in order: -X, +Z, +X, -Z, -Y, +Y
    // Layout: [ -X | +Z | +X | -Z ]
    //         [ -Y |     | +Y ]
    const negX = extractFace(0, faceSize);
    const posZ = extractFace(faceSize, faceSize);
    const posX = extractFace(faceSize * 2, faceSize);
    const negZ = extractFace(faceSize * 3, faceSize);
    const negY = extractFace(faceSize, faceSize * 2);
    const posY = extractFace(faceSize, 0);

    // Create cubemap texture
    const cubemap = new Texture(device, {
        width: faceSize,
        height: faceSize,
        format: PIXELFORMAT_RGBA8,
        cubemap: true,
        addressU: ADDRESS_CLAMP_TO_EDGE,
        addressV: ADDRESS_CLAMP_TO_EDGE,
        minFilter: FILTER_LINEAR,
        magFilter: FILTER_LINEAR
    });

    const toCanvas = (face: ImageData) => {
        const c = document.createElement('canvas');
        c.width = faceSize;
        c.height = faceSize;
        const cctx = c.getContext('2d');
        if (!cctx) {
            throw new Error('Failed to get cubemap face canvas context');
        }
        cctx.putImageData(face, 0, 0);
        return c;
    };

    // PlayCanvas cubemap face order: +X, -X, +Y, -Y, +Z, -Z
    cubemap.setSource([
        toCanvas(posX),
        toCanvas(negX),
        toCanvas(posY),
        toCanvas(negY),
        toCanvas(posZ),
        toCanvas(negZ)
    ]);

    return cubemap;
};

/**
 * Load cubemap from a file
 */
const loadCubemapFromFile = async (device: any, file: File): Promise<Texture> => {
    return new Promise((resolve, reject) => {
        const img = new Image();
        const objectUrl = URL.createObjectURL(file);
        
        img.onload = () => {
            try {
                const cubemap = loadCubemapFromImage(device, img);
                URL.revokeObjectURL(objectUrl); // Clean up
                resolve(cubemap);
            } catch (error) {
                URL.revokeObjectURL(objectUrl); // Clean up on error
                reject(error);
            }
        };
        img.onerror = () => {
            URL.revokeObjectURL(objectUrl); // Clean up on error
            reject(new Error('Failed to load image'));
        };
        img.src = objectUrl;
    });
};

export { loadCubemapFromImage, loadCubemapFromFile };
