import Shape from "./Shape";
import Strides from "./Strides";
import core from "./core/build";
import { get_row_major } from "./util";
import tensor_to_string from "./to_string";
import { ordinal_str } from "./util";
import * as ops from "./tensor_operations.ts";

enum  STRUCT_LAYOUT { DATA, SHAPE, STRIDES, RANK, NELEM, NDATA, OFFSET, SIZE, ISVIEW }
const STRUCT_SIZE = Object.entries(STRUCT_LAYOUT).length / 2;

export class RawTensor {
    private readonly view: Int32Array;
    data: Float32Array; // todo: should be private probably
    shape: Shape;
    strides: Strides;

    constructor(ptr: number) {
        // set up typed arrays for data access
        this.view = new Int32Array(core.memory.buffer, ptr, STRUCT_SIZE);
        this.shape   = new Shape  (new Int32Array(core.memory.buffer, this.shape_ptr, this.rank), true);
        this.strides = new Strides(new Int32Array(core.memory.buffer, this.strides_ptr, this.rank), true);
        this.data    = new Float32Array(core.memory.buffer, this.data_ptr, this.ndata);
    }

    public get rank()           {  return this.view[STRUCT_LAYOUT.RANK]; }
    public get nelem()          { return this.view[STRUCT_LAYOUT.NELEM]; }
    public get offset()         { return this.view[STRUCT_LAYOUT.OFFSET]; }
    public get ndata()          { return this.view[STRUCT_LAYOUT.NDATA]; }
    public get size()           { return this.view[STRUCT_LAYOUT.SIZE]; }
    public get isview()         { return this.view[STRUCT_LAYOUT.ISVIEW]; }
    public get data_ptr()       { return this.view[STRUCT_LAYOUT.DATA]; }
    public get shape_ptr()      { return this.view[STRUCT_LAYOUT.SHAPE]; }
    public get strides_ptr()    { return this.view[STRUCT_LAYOUT.STRIDES]; }
    public get rows()           { return this.get_axis_size(this.rank - 2); }
    public get cols()           { return this.get_axis_size(this.rank - 1); }
    public get is_scalar()      { return this.nelem === 1; }
    public get_axis_size    = (axis_index: number) => this.shape.get_axis_size(axis_index);

    public get ptr(): number {
        return this.view.byteOffset;
    }

    public toString = (precision?: number) => tensor_to_string(this, precision);
    public print = (precision?: number) => console.log(tensor_to_string(this, precision) + "\n---");
    public print_info(title: string = "TENSOR INFO") {
        const max_entries = 16;
        const precision = 3;

        const exp = 10 ** precision;
        const data = [...this.data.slice(0, max_entries)].map(v => Math.floor(v * exp) / exp);

        console.log(
            `${title}\n` +
            `  address: 0x${this.ptr.toString(16)}\n` +
            `  is view: ${this.isview ? "true" : "false"}\n` +
            `  shape:   [${this.shape.join(", ")}]\n` +
            `  strides: [${this.strides.join(", ")}]\n` +
            `  rank:    ${this.rank}\n` +
            `  nelem:   ${this.nelem}\n` +
            `  ndata:   ${this.ndata}\n` +
            `  size:    ${this.size} bytes\n` +
            `  offset:  ${this.offset}\n` +
            `  data: [${data.join(", ")}${this.data.length > max_entries ? ", ..." : ""}]\n`
        );
    }

    *get_axis_iterable(n: number): Generator<RawTensor> {
        if (n > this.rank - 2)
            throw new Error(`Cannot iterate over ${ordinal_str(n)} axis.`);

        const view = RawTensor.view_from(this, n + 1);
        const nelem = this.shape.flatten(this.rank - n)[0];

        for (let index = 0; index < nelem; index++) {
            const offset = index * this.strides[n];
            // creating separate views instead of using a single one and
            // incrementing the offset, because the user might access
            // the views even after the iteration process is done
            // todo: this leaves us with the problem of de-allocation, however
            yield RawTensor.view_from(view, 0, offset);
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
    public create_view = (axis = 0, offset = 0) => RawTensor.view_from(this, axis, offset);

    // metadata operations
    public transpose  = (...permutation: number[]) => ops.transpose(this, permutation);

    // Returns the value of the tensor as a scalar if the tensor only has one element
    public get item() {
        if (this.nelem !== 1) throw new Error("Tensor.item is only valid on scalar tensors.");
        return this.data[this.offset];
    }

    // operation shorthands
    public get T(): RawTensor {
        return ops.transpose(this);
    }

    [Symbol.iterator]() {
        return this.get_axis_iterable(0);
    }

    // builders
    static scalar = (scalar?: number): RawTensor => RawTensor.create([1], scalar ? [scalar] : undefined);
    static like = (other: RawTensor): RawTensor => RawTensor.create([...other.shape]);
    static view_from = (a: RawTensor, axis = 0, offset = 0): RawTensor => new RawTensor(core._create_view(a.ptr, axis, offset));
    static create(shape: number[] | Shape, data?: number[]): RawTensor {
        const _shape = [...shape];
        const nelem = _shape.reduce((acc: number, val: number) => acc * val, 1);
    
        if (data !== undefined && data.length !== nelem)
            throw new Error(`Cannot cast array of size ${data.length} into tensor of shape [${shape}]`);
    
        const ptr = core._create_tensor(shape.length, nelem);
        const new_tensor = new RawTensor(ptr);
    
        if (data !== undefined) new_tensor.data.set(data);
        new_tensor.shape.set(shape);
        new_tensor.strides.set(get_row_major(_shape));
    
        return new_tensor;
    }
}
