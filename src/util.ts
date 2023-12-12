import core from './core/build';

export const core_ready = new Promise<null>((resolve) => {
    core.onRuntimeInitialized = () => {
        core.memory = new Uint8Array(core.HEAPU8.buffer);
        resolve(null);
    }
});
