import './glslang.js';

let glslangPromise;

const resolveWasmPath = (wasmPath) => {
    const base = new URL('./', import.meta.url);
    if (!wasmPath) {
        return new URL('glslang.wasm', base).toString();
    }
    const resolved = new URL(wasmPath, base).toString();
    if (resolved.endsWith('/glslang-wrapper.wasm')) {
        return new URL('glslang.wasm', base).toString();
    }
    return resolved;
};

export default (wasmPath) => {
    if (!glslangPromise) {
        glslangPromise = (async () => {
            if (typeof globalThis.glslang !== 'function') {
                throw new Error('glslang global initializer is unavailable after script load.');
            }
            return globalThis.glslang(resolveWasmPath(wasmPath));
        })();
    }
    return glslangPromise;
};
