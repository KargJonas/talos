import core from "./core/build";

// todo could extract functionality to a superclass that can then be
// extended by Shape, Stride and Tensor (e.g. class WASMView)
export default class Strides extends Int32Array {
    constructor(strides: Int32Array | number[], attached = true) {
        if (attached) {
            if (!(strides instanceof Int32Array)) throw new Error("Shape must be Int32Array!");

            // bind shape to a memory location (usually inside a tensor)
            // const ptr = core._alloc_starr(shape.length);
            super(core.memory.buffer, strides.byteOffset, strides.length);
            // this.set([...strides]);
        } else {
            // create a detached shape (not bound to a tensor)
            super(strides.length);
            this.set(strides);
        }
    }
}
