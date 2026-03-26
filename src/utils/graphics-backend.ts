import { GraphicsDevice } from 'playcanvas';

const hasFn = (target: unknown, name: string) => {
    return typeof (target as Record<string, unknown>)?.[name] === 'function';
};

const isWebGPU = (device: GraphicsDevice) => {
    return Boolean((device as any).isWebGPU);
};

const getGraphicsBackendName = (device: GraphicsDevice) => {
    if (isWebGPU(device)) {
        return 'webgpu';
    }
    if (hasFn(device, 'readPixels')) {
        return 'webgl2';
    }
    return 'unknown';
};

const supportsRawReadPixels = (device: GraphicsDevice) => {
    return hasFn(device, 'readPixels');
};

const supportsGLInternalFormatRead = (device: GraphicsDevice) => {
    return !isWebGPU(device);
};

const assertRawReadPixelsSupported = (device: GraphicsDevice, operation: string) => {
    if (!supportsRawReadPixels(device)) {
        throw new Error(`${operation} requires readPixels support on the active graphics backend (${getGraphicsBackendName(device)}).`);
    }
};

const assertGLInternalFormatReadSupported = (device: GraphicsDevice, operation: string) => {
    if (!supportsGLInternalFormatRead(device)) {
        throw new Error(`${operation} requires GL internal-format readback and is unavailable on backend ${getGraphicsBackendName(device)}.`);
    }
};

export {
    isWebGPU,
    getGraphicsBackendName,
    supportsRawReadPixels,
    supportsGLInternalFormatRead,
    assertRawReadPixelsSupported,
    assertGLInternalFormatReadSupported
};
