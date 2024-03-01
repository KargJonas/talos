import Shape from "./Shape";
import Strides from "./Strides";
import core from "./core/build";
import { get_row_major } from "./util";
import tensor_to_string from "./to_string";
import { ordinal_str } from "./util";
import * as ops from "./tensor_operations";

enum  STRUCT_LAYOUT { DATA, SHAPE, STRIDES, RANK, NELEM, NDATA, OFFSET, SIZE, ISVIEW }
const STRUCT_SIZE = Object.entries(STRUCT_LAYOUT).length / 2;

export class Tensor {
    private view: Int32Array;
    data: Float32Array;
    shape: Shape;
    strides: Strides;

    constructor(ptr: number) {
        // set up typed arrays for data access
        this.view = new Int32Array(core.memory.buffer, ptr, STRUCT_SIZE);
        this.shape   = new Shape  (new Int32Array(core.memory.buffer, this.get_shape_ptr(), this.get_rank()), true);
        this.strides = new Strides(new Int32Array(core.memory.buffer, this.get_strides_ptr(), this.get_rank()), true);
        this.data    = new Float32Array(core.memory.buffer, this.get_data_ptr(), this.get_ndata());
    }

    public set_offset   = (offset: number) => this.view[STRUCT_LAYOUT.OFFSET] = offset;

    public get_view_ptr     = () => this.view.byteOffset;
    public get_rank         = () => this.view[STRUCT_LAYOUT.RANK];
    public get_nelem        = () => this.view[STRUCT_LAYOUT.NELEM];
    public get_offset       = () => this.view[STRUCT_LAYOUT.OFFSET];
    public get_ndata        = () => this.view[STRUCT_LAYOUT.NDATA];
    public get_size         = () => this.view[STRUCT_LAYOUT.SIZE];
    public get_isview       = () => this.view[STRUCT_LAYOUT.ISVIEW];
    public get_data_ptr     = () => this.view[STRUCT_LAYOUT.DATA];
    public get_shape_ptr    = () => this.view[STRUCT_LAYOUT.SHAPE];
    public get_strides_ptr  = () => this.view[STRUCT_LAYOUT.STRIDES];
    public get_rows         = () => this.get_axis_size(this.get_rank() - 2);
    public get_cols         = () => this.get_axis_size(this.get_rank() - 1);
    public get_axis_size    = (axis_index: number) => this.shape.get_axis_size(axis_index);

    public print = () => console.log(tensor_to_string(this) + "\n---");
    public print_info(title: string = "TENSOR INFO") {
        const max_entries = 16;
        const precision = 3;

        const exp = 10 ** precision;
        const data = [...this.data.slice(0, max_entries)].map(v => Math.floor(v * exp) / exp);

        console.log(
            `${title}\n` +
            `  address: 0x${this.get_view_ptr().toString(16)}\n` +
            `  is view: ${this.get_isview() ? "true" : "false"}\n` +
            `  shape:   [${this.shape.join(", ")}]\n` +
            `  strides: [${this.strides.join(", ")}]\n` +
            `  rank:    ${this.get_rank()}\n` +
            `  nelem:   ${this.get_nelem()}\n` +
            `  ndata:   ${this.get_ndata()}\n` +
            `  size:    ${this.get_size()} bytes\n` +
            `  offset:  ${this.get_offset()}\n` +
            `  data: [${data.join(", ")}${this.data.length > max_entries ? ", ..." : ""}]\n`
        );
    }

    *get_axis_iterable(n: number): Generator<Tensor> {
        if (n > this.get_rank() - 2)
            throw new Error(`Cannot iterate over ${ordinal_str(n)} axis.`);

        const view = create_view(this, n + 1);
        const nelem = this.shape.flatten(this.get_rank() - n)[0];

        for (let index = 0; index < nelem; index++) {
            const offset = index * this.strides[n];
            // creating separate views instead of using a single one and
            // incrementing the offset, because the user might access
            // the views even after the iteration process is done
            // todo: this leaves us with the problem of deallocation, however
            yield create_view(view, 0, offset);
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
    public create_view = (axis = 0, offset = 0) => create_view(this, axis, offset);

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
    public pow        = (other: Tensor | number, in_place = false) => ops.pow(this, other, in_place);
    public dot        = (other: Tensor, in_place = false) => ops.dot(this, other, in_place);
    public matmul     = (other: Tensor, in_place = false) => ops.matmul(this, other, in_place);

    // reduce operations
    public min  = (): number => ops.min(this);
    public max  = (): number => ops.max(this);
    public sum  = (): number => ops.sum(this);
    public mean = (): number => ops.mean(this);

    // operation shorthands
    public get T() {
        return ops.transpose(this);
    }

    [Symbol.iterator]() {
        return this.get_axis_iterable(0);
    }
}

export default function tensor(shape: number[] | Shape, data?: number[]): Tensor {
    const _shape = [...shape];
    const nelem = _shape.reduce((acc: number, val: number) => acc * val, 1);

    if (data !== undefined && data.length !== nelem)
        throw new Error(`Cannot cast array of size ${data.length} into tensor of shape [${shape}]`);

    const ptr = core._create_tensor(shape.length, nelem);
    const new_tensor = new Tensor(ptr);

    if (data !== undefined) new_tensor.data.set(data);
    new_tensor.shape.set(shape);
    new_tensor.strides.set(get_row_major(_shape));

    return new_tensor;
}

export function tensor_like(other: Tensor) {
    return tensor([...other.shape]);
}

// returns a view of an element in the desired axis
export function create_view(a: Tensor, axis = 0, offset = 0) {
    const ptr = core._create_view(a.get_view_ptr(), axis, offset);
    return new Tensor(ptr);
}
