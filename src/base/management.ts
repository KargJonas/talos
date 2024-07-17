import core from "./core/build";

enum  STRUCT_LAYOUT { ALLOCATED, NTENSORS }
const STRUCT_SIZE = Object.entries(STRUCT_LAYOUT).length / 2;

let view: Uint32Array;
let memory: Uint32Array;

const init_error = () => console.log("[core] error during initialization");
const init = () => {
    memory = new Uint32Array(core.HEAP32.buffer);

    // augmenting the core module with a custom memory object
    // whenever this object is accessed, we check if the buffer
    // has been detached (because of a memory growth event)
    // if so, we create a new view of the memory.
    // an alternative solution we would be to inject some code into
    // the emscripten build output to update the view when growth happens.
    // todo: test perf.
    Object.defineProperty(core, "memory", {
        get() {
            // memory buffer has detached, growth event has ocurred
            if (memory.byteLength === 0) {
                memory = new Uint32Array(core.HEAP32.buffer);
                init_mgmt();
                console.log("[core] memory growth");
            }

            return memory;
        },
      
        configurable: false,
        enumerable: true,
    });

    init_mgmt();
    console.log("[core] initialized");
};

export const core_ready = new Promise<null>((resolve) => {
    let loaded = false;

    core.onRuntimeInitialized = () => {
        try   { init(); }
        catch { init_error(); }

        loaded = true;
        resolve(null);
    };

    // todo
    //   this is a band-aid-fix
    //   the real solution is to find out why onRuntimeInitialized
    //   does not reliably trigger.
 
    // polling for core readiness
    const loading = setInterval(() => {
        if (!loaded && core.HEAPU8) {
            try   { init(); }
            catch { init_error(); }
            
            loaded = true;
            resolve(null);
        }        

        if (loaded) clearInterval(loading);
    }, 5);
});

export function init_mgmt() {
    view = new Uint32Array(core.memory.buffer, core._get_mgmt_ptr(), STRUCT_SIZE);
}

export function get_total_allocated(): number {
    return view[STRUCT_LAYOUT.ALLOCATED];
}

export function get_ntensors(): number {
    return view[STRUCT_LAYOUT.NTENSORS];
}

export function print_memory_status() {
    console.log(
        "MEMORY INFO\n" + 
        `  Number of Tensors: ${get_ntensors()}\n` +
        `  Total allocated:   ${get_total_allocated()} bytes`
    );
}
