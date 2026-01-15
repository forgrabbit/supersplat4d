import {
    Texture,
    PIXELFORMAT_RGBA8,
    ADDRESS_CLAMP_TO_EDGE,
    FILTER_LINEAR,
    CUBEFACE_POSX,
    CUBEFACE_NEGX,
    CUBEFACE_POSY,
    CUBEFACE_NEGY,
    CUBEFACE_POSZ,
    CUBEFACE_NEGZ,
    WebglGraphicsDevice
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

    // Upload each face using WebGL API directly
    // PlayCanvas doesn't expose a direct API for cubemap face upload, so we use WebGL
    const glDevice = device as WebglGraphicsDevice;
    const gl = glDevice.gl;
    
    // Force PlayCanvas to initialize the texture by uploading dummy data
    // This ensures the WebGL texture object is created
    cubemap.lock();
    cubemap.unlock();
    
    // Get the WebGL texture handle that PlayCanvas created
    const textureHandle = (cubemap.impl as any)._glTexture;
    
    if (!textureHandle) {
        throw new Error('Failed to get WebGL texture handle from PlayCanvas');
    }
    
    // Save current texture binding
    const previousTexture = gl.getParameter(gl.TEXTURE_BINDING_CUBE_MAP);
    
    // Bind the cubemap texture
    gl.bindTexture(gl.TEXTURE_CUBE_MAP, textureHandle);
    
    const uploadFace = (face: number, imageData: ImageData) => {
        faceCanvas.width = faceSize;
        faceCanvas.height = faceSize;
        faceCtx.putImageData(imageData, 0, 0);
        
        // Map PlayCanvas CUBEFACE constants to WebGL constants
        const glFaceMap: { [key: number]: number } = {
            [CUBEFACE_POSX]: gl.TEXTURE_CUBE_MAP_POSITIVE_X,
            [CUBEFACE_NEGX]: gl.TEXTURE_CUBE_MAP_NEGATIVE_X,
            [CUBEFACE_POSY]: gl.TEXTURE_CUBE_MAP_POSITIVE_Y,
            [CUBEFACE_NEGY]: gl.TEXTURE_CUBE_MAP_NEGATIVE_Y,
            [CUBEFACE_POSZ]: gl.TEXTURE_CUBE_MAP_POSITIVE_Z,
            [CUBEFACE_NEGZ]: gl.TEXTURE_CUBE_MAP_NEGATIVE_Z
        };
        
        const glFace = glFaceMap[face];
        if (glFace) {
            gl.texImage2D(glFace, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, faceCanvas);
        }
    };

    uploadFace(CUBEFACE_NEGX, negX);
    uploadFace(CUBEFACE_POSZ, posZ);
    uploadFace(CUBEFACE_POSX, posX);
    uploadFace(CUBEFACE_NEGZ, negZ);
    uploadFace(CUBEFACE_NEGY, negY);
    uploadFace(CUBEFACE_POSY, posY);
    
    // Restore previous texture binding
    gl.bindTexture(gl.TEXTURE_CUBE_MAP, previousTexture);
    
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
