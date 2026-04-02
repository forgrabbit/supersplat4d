import './twgsl.js';

let twgslPromise;

const resolveWasmPath = (wasmPath) => {
    const base = new URL('./', import.meta.url);
    if (!wasmPath) {
        return new URL('twgsl.wasm', base).toString();
    }
    const resolved = new URL(wasmPath, base).toString();
    if (resolved.endsWith('/twgsl-wrapper.wasm')) {
        return new URL('twgsl.wasm', base).toString();
    }
    return resolved;
};

export default (wasmPath) => {
    if (!twgslPromise) {
        twgslPromise = (async () => {
            if (typeof globalThis.twgsl !== 'function') {
                throw new Error('twgsl global initializer is unavailable after script load.');
            }
            return globalThis.twgsl(resolveWasmPath(wasmPath));
        })();
    }
    return twgslPromise;
};
