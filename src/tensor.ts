import core from './core/build';
import { Shape } from './shape';
import { mat_to_string } from './util';

export class Tensor {
    public readonly shape: Shape; // [outermost axis, ..., rows, cols]
    public readonly data: Float32Array;

    // proxy props
    public readonly rank: number;
    public readonly nrows: number;
    public readonly ncols: number;

    constructor(shape: Shape, data: Float32Array) {
        this.shape = new Shape(...shape);
        this.data = data;

        this.rank = this.shape.get_ndim();
        this.nrows = this.shape.get_rows();
        this.ncols = this.shape.get_cols();
    }

    public clone(): Tensor {
        const new_tensor = tensor(this.shape);
        new_tensor.data.set(this.data);
        return new_tensor;
    }

    // data operations
    public get_ptr = () => this.data.byteOffset;
    public free = () => core._free(this.get_ptr());

    public rand(min = -1, max = 1) {
        core._rand_f(this.get_ptr(), this.data.length, min, max);
        return this;
    }

    public rand_int(min = -1, max = 1) {
        core._rand_i(this.get_ptr(), this.data.length, min, max);
        return this;
    }

    // shape operations
    flatten = (n: number): Tensor => new Tensor(this.shape.flatten(n), this.data);

    *get_axis_iterable(n: number) {
        const shape = this.shape.get_axis_shape(n + 1);
        const n_elements = shape.get_nelem();

        for (const index of this.shape.get_axis_iterable(n)) {
            yield new Tensor(shape, this.data.subarray(index, index + n_elements))
        }
    }

    public get(...loc: number[]): Tensor | number {
        if (loc.length > this.rank)
            throw new Error(`Location [${loc}] is too specific for shape [${this}]`);

        const [index, shape] = this.shape.get_index(...loc);

        // return element, if location describes a scalar, return subtensor if not
        if (loc.length === this.rank) return this.data[index];
        return new Tensor(shape, this.data.subarray(index, index + shape.get_nelem()));
    }

    private binary_op(name: string, core_fn_prw: Function, core_fn_scl: Function, b: Tensor | number) {
        if (b instanceof Tensor) {
            // perform fast pairwise addition without broadcasting
            if (this.shape.equals(b.shape)) {
                core_fn_prw(this.get_ptr(), b.get_ptr(), this.data.length);
                return this;
            }

            // check if broadcasting is possible
            if (!this.shape.broadcastable(b.shape))
                throw new Error(`Shape mismatch: Cannot broadcast tensor of shape [${this.shape}] with [${b.shape}].`);

            const max_rank = Math.max(this.rank, b.rank);
            const result_shape = this.shape.broadcast(b.shape);
            const result = tensor(result_shape);

            const exp_a = this.shape.expand_left(max_rank);
            const exp_b =    b.shape.expand_left(max_rank);
            const strides_a = exp_a.get_strides();
            const strides_b = exp_b.get_strides();

            for (let i = 0; i < max_rank; i++) {
                if (exp_a[i] === 1) strides_a[i] = 0;
                if (exp_b[i] === 1) strides_b[i] = 0;
            }

            const strides_arr_size = max_rank * 3;
            const strides_arr_ptr = core._alloc_starr(strides_arr_size);
            const strides = new Uint32Array(
                core.memory.buffer,
                strides_arr_ptr,
                strides_arr_size);
            
            strides.set([...strides_a, ...strides_b, ...result_shape]);

            core._prw_op_broadcast(
                this.get_ptr(), b.get_ptr(), result.get_ptr(),
                strides_arr_ptr, max_rank);

            return result;
        }
        else if (typeof b === 'number') core_fn_scl(this.get_ptr(), b, this.data.length);
        else throw new Error(`Type mismatch: Tensor.${name}() expects a Tensor or number.`);
        return this;
    }

    private unary_op(core_fn: Function) {
        core_fn(this.get_ptr(), this.data.length);
        return this;
    }

    // binary operations (pairwise/scalar)
    public add = (other: Tensor | number) => this.binary_op('add', core._add_prw, core._add_scl, other);
    public sub = (other: Tensor | number) => this.binary_op('sub', core._sub_prw, core._sub_scl, other);
    public div = (other: Tensor | number) => this.binary_op('div', core._div_prw, core._div_scl, other);
    public mul = (other: Tensor | number) => this.binary_op('mul', core._mul_prw, core._mul_scl, other);

    // unary operations
    public relu = () => this.unary_op(core._act_relu);
    public tanh = () => this.unary_op(core._act_tanh);

    // checks, if this tensor is matmul/dot compatible with tensor b regarding columns/rows
    private check_row_col_compat(b: Tensor) {
        if (this.shape.get_cols() !== b.shape.get_rows())
            throw new Error(`Cannot multiply tensors of shape [${this.shape}] and [${b.shape}]`);
    }
    
    // standard matmul on tensors 
    public matmul(b: Tensor): Tensor {
        if (!(b instanceof Tensor)) throw new Error('Tensor.matmul() expects a tensor.');
        this.check_row_col_compat(b);

        // flatten tensors to a "list of matrices" and get the size of that list
        const nmat_a = this.shape.flatten(3)[0];
        const nmat_b =    b.shape.flatten(3)[0];
        
        // check hidim matmul compatibility
        if (nmat_a > 1 && nmat_b > 1 && nmat_a != nmat_b)
            throw new Error(`Cannot multiply matrices of shape [${this.shape}] and [${b.shape}]`);

        // get shape of resulting tensor without the last two axes
        const high_level_shape = this.rank > b.rank
            ? this.shape.slice(0, this.rank - 2)
            : b.shape.slice(0, b.rank - 2);

        // create result tensor
        const result_shape = new Shape(...high_level_shape, this.nrows, b.ncols); // todo kinda want to do this with .get_axis_shape
        const result = tensor(result_shape);

        // perform computation using core
        core._mul_tns( /*
            data pointer    n rows,     n cols,     number mats (=i) when flattened to [i, rows, cols] */
            this.get_ptr(), this.nrows, this.ncols, nmat_a,
            b.get_ptr(),    b.nrows,     b.ncols,   nmat_b,
            result.get_ptr()
        );

        return result;
    }

    // numpy-style dot-product
    public dot(b: Tensor): Tensor {
        if (!(b instanceof Tensor)) throw new Error('Tensor.dot() expects a tensor.');
        this.check_row_col_compat(b);

        const result_shape = new Shape(
            ...this.shape.slice(0, this.rank - 1),                  // shape of tensor a without the last axis
               ...b.shape.slice(0,    b.rank - 2), b.shape[b.rank - 1]);  // shape of tensor b without the second-to-last axis
        const result = tensor(result_shape);

        // flatten tensors to a list of vectors/matrices respectively
        const nvec_a = this.shape.flatten(2)[0];
        const nmat_b =    b.shape.flatten(3)[0];

        core._dot_tns(
            this.get_ptr(), this.ncols,    nvec_a,
            b.get_ptr(), b.nrows, b.ncols, nmat_b,
            result.get_ptr()
        );

        return result;
    }

    // usability methods
    public toString = () => this.to_str();
    public to_str(num_width = 10, space_before = 0) {
        switch(this.rank) {
            case 0: return '[]';
            case 1: return `[ ${this.data.join(', ')} ]`;
            case 2: return mat_to_string(this, num_width, space_before);
        }

        // hidim tensors
        let strings: string[] = [];
        for (const element of this.get_axis_iterable(0)) {
            strings.push(element.to_str(num_width, space_before + 2)!);
        }

        return `[ ${strings.join(',\n\n' + ' '.repeat(space_before + 2))} ]`
    }
}

/**
 * create a new tensor
 * NOTE: Only safe to call after core_ready is resolved
 * @param shape shape of tensor
 * @param data tensor data
 * @returns tensor of specified shape with specified data
 */
export default function tensor(shape: number[] | Shape, data?: number[]) {
    const _shape = new Shape(...shape);         // clone shape (shapes are mutable, prevents issues later)
    const n_elem = _shape.get_nelem();          // compute number of elements based on shape
    const pointer = core._alloc_farr(n_elem);   // allocate space in wasm memory

    const _data = new Float32Array(             // create js f32arr binding to that space
        core.memory.buffer,                     // ref to wasm memory
        pointer,                                // ref to previously allocated space
        n_elem);                                // nr of elements in array

    if (data !== undefined) _data.set(data);    // fill array with data

    return new Tensor(_shape, _data);
}
