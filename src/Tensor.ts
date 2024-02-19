import Shape from "./Shape";
import Strides from "./Strides";
import core from "./core/build";
import { get_row_major } from "./stride_operations";
import tensor_to_string from "./to_string";
import { create_farr, ordinal_str } from "./util";
import * as ops from "./tensor_operations";

enum  STRUCT_LAYOUT { DATA, SHAPE, STRIDES, RANK, NELEM, OFFSET, ISVIEW }
const STRUCT_SIZE = Object.entries(STRUCT_LAYOUT).length / 2;

export class Tensor {
    private view: Int32Array;
    data: Float32Array;
    shape: Shape;
    strides: Strides;

    constructor(
        shape: Shape, strides: Strides, data: Float32Array,
        offset: number = 0, isview: boolean = false
    ) {
        // todo remove after complete implementation. this is just a safeguard
        if (offset !== 0 && !isview) throw new Error("This cant be right.");

        const ptr = core._create_tensor();
        this.view = new Int32Array(core.memory.buffer, ptr, STRUCT_SIZE);

        this.shape = shape;
        this.strides = strides;
        this.data = data;

        this.view[STRUCT_LAYOUT.DATA] = this.data.byteOffset;
        this.view[STRUCT_LAYOUT.SHAPE] = this.shape.byteOffset;
        this.view[STRUCT_LAYOUT.STRIDES] = this.strides.byteOffset;

        this.set_rank(shape.length);
        this.set_nelem(shape.get_nelem()); // previously used data.length but this causes issues with subtensors
        this.set_offset(offset);
        this.set_isview(isview ? 1 : 0);
    }

    public set_rank     = (rank:   number) => this.view[STRUCT_LAYOUT.RANK]   = rank;
    public set_nelem    = (nelem:  number) => this.view[STRUCT_LAYOUT.NELEM]  = nelem;
    public set_offset   = (offset: number) => this.view[STRUCT_LAYOUT.OFFSET] = offset;
    public set_isview   = (isview: number) => this.view[STRUCT_LAYOUT.ISVIEW] = isview;

    public get_rank         = () => this.view[STRUCT_LAYOUT.RANK];
    public get_nelem        = () => this.view[STRUCT_LAYOUT.NELEM];
    public get_offset       = () => this.view[STRUCT_LAYOUT.OFFSET];
    public get_isview       = () => this.view[STRUCT_LAYOUT.ISVIEW];
    public get_rows         = () => this.get_axis_size(this.get_rank() - 2);
    public get_cols         = () => this.get_axis_size(this.get_rank() - 1);
    public get_view_ptr     = () => this.view.byteOffset;
    public get_data_ptr     = () => this.data.byteOffset;
    public get_shape_ptr    = () => this.shape.byteOffset;
    public get_strides_ptr  = () => this.strides.byteOffset;
    public get_axis_size    = (axis_index: number) => this.shape.get_axis_size(axis_index);

    public toString = () => tensor_to_string(this);

    public print = () => console.log(this.toString() + "\n---");
    public print_info = (title: string = "TENSOR INFO") => console.log(
        `${title}\n` +
        `  address: 0x${this.get_view_ptr().toString(16)}\n` +
        `  is view: ${this.get_isview() ? "true" : "false"}\n` +
        `  shape:   [${this.shape}]\n` +
        `  strides: [${this.strides}]\n` +
        `  nelem:   ${this.get_nelem()}\n` +
        `  offset:  ${this.get_offset()}\n`
    );

    *get_axis_iterable(n: number): Generator<Tensor> {
        if (n > this.get_rank() - 2)
            throw new Error(`Cannot iterate over ${ordinal_str(n)} axis.`);

        // todo: add get_axis_strides() and extract superclass from Shape and Strides
        const shape = new Shape(this.shape.get_axis_shape(n + 1), true);
        const strides = new Strides([...this.strides].slice(n - this.get_rank() + 1), true);
        const nelem = this.shape.flatten(this.get_rank() - n)[0];

        for (let index = 0; index < nelem; index++) {
            const offset = index * this.strides[n];
            yield new Tensor(shape, strides, this.data, offset, true);
        }
    }

    public rand(min = -1, max = 1) {
        core._rand_f(this.get_data_ptr(), this.data.length, min, max);
        return this;
    }

    public rand_int(min = -1, max = 1) {
        core._rand_i(this.get_data_ptr(), this.data.length, min, max);
        return this;
    }

    public fill(value: number) {
        core._fill(this.get_data_ptr(), this.data.length, value);
        return this;
    }

    // init/free operations
    public zeros = () => this.fill(0);
    public ones  = () => this.fill(1);
    public free  = () => ops.free(this);
    public clone = () => ops.clone(this);

    // metadata operations
    public transpose  = (...permutation: number[]) => ops.transpose(this, permutation);

    // unary operations
    public relu       = (in_place = false) => ops.relu(this, in_place);
    public binstep    = (in_place = false) => ops.binstep(this, in_place);
    public logistic   = (in_place = false) => ops.logistic(this, in_place);
    public negate     = (in_place = false) => ops.negate(this, in_place);
    public sin        = (in_place = false) => ops.sin(this, in_place);
    public cos        = (in_place = false) => ops.cos(this, in_place);
    public tan        = (in_place = false) => ops.tan(this, in_place);
    public asin       = (in_place = false) => ops.asin(this, in_place);
    public acos       = (in_place = false) => ops.acos(this, in_place);
    public atan       = (in_place = false) => ops.atan(this, in_place);
    public sinh       = (in_place = false) => ops.sinh(this, in_place);
    public cosh       = (in_place = false) => ops.cosh(this, in_place);
    public tanh       = (in_place = false) => ops.tanh(this, in_place);
    public exp        = (in_place = false) => ops.exp(this, in_place);
    public log        = (in_place = false) => ops.log(this, in_place);
    public log10      = (in_place = false) => ops.log10(this, in_place);
    public log2       = (in_place = false) => ops.log2(this, in_place);
    public invsqrt    = (in_place = false) => ops.invsqrt(this, in_place); // careful - negative input values will produce Infinity, not NaN
    public sqrt       = (in_place = false) => ops.sqrt(this, in_place);
    public ceil       = (in_place = false) => ops.ceil(this, in_place);
    public floor      = (in_place = false) => ops.floor(this, in_place);
    public abs        = (in_place = false) => ops.abs(this, in_place);
    public reciprocal = (in_place = false) => ops.reciprocal(this, in_place);

    // binary operations
    public add        = (other: Tensor | number, in_place = false) => ops.add(this, other, in_place);
    public sub        = (other: Tensor | number, in_place = false) => ops.sub(this, other, in_place);
    public mul        = (other: Tensor | number, in_place = false) => ops.mul(this, other, in_place);
    public div        = (other: Tensor | number, in_place = false) => ops.div(this, other, in_place);
    public dot        = (other: Tensor, in_place = false) => ops.dot(this, other, in_place);
    public matmul     = (other: Tensor, in_place = false) => ops.matmul(this, other, in_place);

    // reduce operations
    public min = (): number => ops.min(this);
    public max = (): number => ops.max(this);
    public sum = (): number => ops.sum(this);

    // operation shorthands
    public get T() {
        return this.transpose();
    }
}

export default function tensor(shape: Shape | number[], data?: number[]): Tensor {
    // @ts-expect-error Obscure signature incompatibility between Int32Array.reduce() and Array.reduce()
    const nelem = shape.reduce((acc: number, val: number) => acc * val, 1);

    const _data = create_farr(nelem);
    const _shape = new Shape(shape, true);
    const _strides = new Strides(get_row_major(_shape), true);

    if (data !== undefined) {
        if (data.length !== nelem) throw new Error(`Cannot cast array of size ${data.length} into tensor of shape [${shape}]`);
        _data.set(data);
    }

    return new Tensor(_shape, _strides, _data);
}

export function tensor_like(other: Tensor) {
    return tensor(other.shape);
}

export function derive_tensor(a: Tensor, shape: number[], strides: number[], offset = 0) {
    const _shape = new Shape(shape, true);
    const _strides = new Strides(strides, true);
    return new Tensor(_shape, _strides, a.data, offset);
}
