import core from "./core/build";

// todo could extract functionality to a superclass that can then be
// extended by Shape, Stride and Tensor (e.g. class WASMView)
export default class Strides extends Int32Array {
    constructor(strides: Int32Array | number[], attached = false) {
        if (attached) {
            if (!(strides instanceof Int32Array)) throw new Error("Strides must be Int32Array!");
            super(core.memory.buffer, strides.byteOffset, strides.length);
        } else {
            // create a detached shape (not bound to a tensor)
            super(strides.length);
            this.set(strides);
        }
    }
}
