import Shape from "./Shape";
import Strides from "./Strides";
import core from "./core/build";
import {tensor_to_string, ordinal_str, tensor_info_to_string} from "./to_string";
import {flatten, get_strides_row_major, NDArray} from "./util";
import * as ops from "./raw_tensor_operations.ts";

enum  STRUCT_LAYOUT { DATA, SHAPE, STRIDES, RANK, NELEM, NDATA, OFFSET, SIZE, ISVIEW }
const STRUCT_SIZE = Object.entries(STRUCT_LAYOUT).length / 2;

/**
 * This class is acts as an interface between wasm memory and js.
 * It contains views of the wasm memory locations where the data, shape, strides, etc.
 * are stored.
 * Additionally, it contains some metadata operations that don't need to be tracked
 * in the computation graph.
 */
export class RawTensor {
    private readonly view: Int32Array;
    data: Float32Array; // todo: should be private probably
    shape: Shape;
    strides: Strides;

    constructor(ptr: number) {
        // set up typed arrays for wasm memory access
        this.view = new Int32Array(core.memory.buffer, ptr, STRUCT_SIZE);
        this.shape   = new Shape  (new Int32Array(core.memory.buffer, this.shape_ptr, this.rank), true);
        this.strides = new Strides(new Int32Array(core.memory.buffer, this.strides_ptr, this.rank), true);
        this.data    = new Float32Array(core.memory.buffer, this.data_ptr, this.ndata);
    }

    public get rank(): number        { return this.view[STRUCT_LAYOUT.RANK]; }
    public get nelem(): number       { return this.view[STRUCT_LAYOUT.NELEM]; }
    public get offset(): number      { return this.view[STRUCT_LAYOUT.OFFSET]; }
    public get ndata(): number       { return this.view[STRUCT_LAYOUT.NDATA]; }
    public get size(): number        { return this.view[STRUCT_LAYOUT.SIZE]; }
    public get isview(): number      { return this.view[STRUCT_LAYOUT.ISVIEW]; }
    public get ptr(): number         { return this.view.byteOffset; }
    public get data_ptr(): number    { return this.view[STRUCT_LAYOUT.DATA]; }
    public get shape_ptr(): number   { return this.view[STRUCT_LAYOUT.SHAPE]; }
    public get strides_ptr(): number { return this.view[STRUCT_LAYOUT.STRIDES]; }
    public get rows(): number        { return this.get_axis_size(this.rank - 2); }
    public get cols(): number        { return this.get_axis_size(this.rank - 1); }
    public get is_scalar(): boolean  { return this.nelem === 1; }

    public get_axis_size = (axis_index: number) => this.shape.get_axis_size(axis_index);
    public toString      = (precision?: number) => tensor_to_string(this, precision);
    public print         = (precision?: number) => console.log(tensor_to_string(this, precision) + "\n---");
    public print_info    = () =>console.log(tensor_info_to_string(this));

    *get_axis_iterable(n: number): Generator<RawTensor> {
        if (n > this.rank - 2)
            throw new Error(`Cannot iterate over ${ordinal_str(n)} axis.`);

        const view = RawTensor.view_of(this, n + 1);
        const nelem = this.shape.flatten(this.rank - n)[0];

        for (let index = 0; index < nelem; index++) {
            const offset = index * this.strides[n];
            // creating separate views instead of using a single one and
            // incrementing the offset, because the user might access
            // the views even after the iteration process is done
            // todo: this leaves us with the problem of de-allocation, however
            yield RawTensor.view_of(view, 0, offset);
        }
    }

    public rand(min = -1, max = 1) {
        core._rand_f(this.data_ptr, this.data.length, min, max);
        return this;
    }

    public rand_int(min = -1, max = 1) {
        core._rand_i(this.data_ptr, this.data.length, min, max);
        return this;
    }

    public fill(value: number) {
        core._fill(this.data_ptr, this.data.length, value);
        return this;
    }

    // init/free operations
    public zeros = () => this.fill(0);
    public ones  = () => this.fill(1);
    public free  = () => ops.free(this);
    public clone = () => ops.clone(this);
    public create_view = (axis = 0, offset = 0) => RawTensor.view_of(this, axis, offset);

    // metadata operations
    public transpose  = (...permutation: number[]) => ops.transpose(this, permutation);
    public get T(): RawTensor { return ops.transpose(this); }

    // Returns the value of the tensor as a scalar if the tensor only has one element
    public get item() {
        if (this.nelem !== 1) throw new Error("RawTensor.item only works on scalar tensors.");
        return this.data[this.offset];
    }

    [Symbol.iterator]() { return this.get_axis_iterable(0); }

    // builders
    static scalar = (scalar?: number): RawTensor => RawTensor.create([1], scalar ? [scalar] : undefined);
    static like = (other: RawTensor): RawTensor => RawTensor.create([...other.shape]);
    static view_of = (a: RawTensor, axis = 0, offset = 0): RawTensor => new RawTensor(core._create_view(a.ptr, axis, offset));
    static create(shape: number[] | Shape, data?: number[]): RawTensor {
        const _shape = [...shape];
        const nelem = _shape.reduce((acc: number, val: number) => acc * val, 1);

        if (data !== undefined && data.length !== nelem)
            throw new Error(`Cannot cast array of size ${data.length} into tensor of shape [${shape}]`);

        const ptr = core._create_tensor(shape.length, nelem);
        const new_tensor = new RawTensor(ptr);
    
        if (data !== undefined) new_tensor.data.set(data);
        new_tensor.shape.set(shape);
        new_tensor.strides.set(get_strides_row_major(_shape));
    
        return new_tensor;
    }

    static from_array(_data: NDArray): RawTensor {
        const [shape, data] = flatten(_data);
        return RawTensor.create(shape, data);
    }
}
