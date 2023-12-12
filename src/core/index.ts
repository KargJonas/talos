import core from './build';
let memory: Uint8Array;

interface core_data {
    core: typeof core;
    memory: Uint8Array;
}

export default new Promise<core_data>((resolve) => {
    core.onRuntimeInitialized = () => {
        memory = new Uint8Array(core.HEAPU8.buffer);
        resolve({ core, memory });
    }
});
