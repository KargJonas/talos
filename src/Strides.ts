import core from "./core/build";

// todo could extract functionality to a superclass that can then be
// extended by Shape, Stride and Tensor (view)
export default class Strides extends Int32Array {
    constructor(strides: Int32Array | number[], attached = true) {
        if (attached) {
            // bind strides to a memory location (usually inside a tensor)
            const ptr = core._alloc_starr(strides.length);
            super(core.memory.buffer, ptr, strides.length);
            this.set(strides);
        } else {
            // create a detatched shape (not bound to a tensor)
            super(strides.length);
            this.set(strides);
        }
    }
}
