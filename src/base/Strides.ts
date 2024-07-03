import core from "./core/build";

// todo: this could be an even more versatile datatype than i had
//    originally thought. we could extend this further, to also
//    use this for the data/view arrays. later when i add WebGPU
//    support, we could add a third option such that it is possible
//    to select, if you want the data in js, wasm or gpu memory.

// todo could extract functionality to a superclass that can then be
// extended by Shape, Stride and RawTensor (e.g. class WASMView)
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
