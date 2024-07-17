import core from "./core/build";

/**
 * "Memory location agnostic array."
 * More precisely, this is an Int32Array, thats either a view of
 * a section of WASM memory or a regular typed array in js memory.
 */
export default class UnifiedArray extends Int32Array {
    /**
     * @param data Initial data in buffer
     * @param attached If true, the buffer will reside in WASM memory.
     *                 If false, it will reside in js memory.
     */
    constructor(data: Int32Array | number[], attached = false) {
        if (attached) {
            // create a attached buffer (resides in wasm memory)
            if (!(data instanceof Int32Array))
                throw new Error("Data of attached UnifiedArray is expected to be of type Int32Array.");

            super(core.memory.buffer, data.byteOffset, data.length);
        } else {
            // create a detached buffer (not in wasm memory)
            super(data.length);
            this.set(data);
        }
    }

    // returns true if two shapes are identical
    equals(other: UnifiedArray): boolean {
        if (this.length !== other.length) return false;

        for (let i = 0; i < this.length; i++) {
            if (this[i] !== other[i] || other[i] === undefined) return false;
        }
    
        return true;
    }
}
