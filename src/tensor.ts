import core from './core/build';
import { Shape } from './shape';
import { mat_to_string } from './util';

export class Tensor {
    shape: Shape; // [outermost axis, ..., rows, cols]
    data: Float32Array;

    constructor(shape: Shape, data: Float32Array) {
        this.shape = shape;
        this.data = data;

        // // this is probably a bad idea but it allows for syntax like this:
        // // my_tensor[1][3].str
        // return new Proxy(this, {
        //     get(target: Tensor, prop: any) {
        //         if (!isNaN(prop)) return target.get(prop) as Tensor;
        //         if (prop === 'str') return target.toString();
        //         return target[prop];
        //     },
        // });
    }

    clone(): Tensor {
        const new_tensor = tensor(this.shape);
        new_tensor.data.set(this.data);
        return new_tensor;
    }

    *get_axis_iterable(n: number) {
        // todo extract into a function (duplicate code in shape.get_axis_iterable)

        const shape = this.shape.get_axis_shape(n + 1);
        const n_elements = shape.get_nelem();

        for (const index of this.shape.get_axis_iterable(n)) {
            yield new Tensor(shape, this.data.subarray(index, index + n_elements))
        }
    }

    get_ptr(): number {
        return this.data.byteOffset;
    }

    // shape operations
    flatten  = (n: number): Tensor => new Tensor(this.shape.flatten(n), this.data);
    mat_flat = ():          Tensor => new Tensor(this.shape.mat_flat(), this.data);

    // data operations
    free = () => core._free_farr(this.get_ptr());

    rand(min = -1, max = 1) {
        core._rand_f(this.get_ptr(), this.data.length, min, max);
        return this;
    }

    rand_int(min = -1, max = 1) {
        core._rand_i(this.get_ptr(), this.data.length, min, max);
        return this;
    }

    // todo: consolidate api of get_axis_iterable/get into a single method

    public get(...loc: number[]): Tensor | number {
        if (loc.length > this.shape.get_ndim()) throw new Error(`Location [${loc}] is too specific for shape [${this}]`);
        const [index, shape] = this.shape.get_index(...loc);
        if (loc.length === this.shape.get_ndim()) return this.data[index];
        return new Tensor(shape, this.data.subarray(index, index + shape.get_nelem()));
    }

    private binary_op(name: string, core_fn_prw: Function, core_fn_scl: Function, other: Tensor | number) {
        if (other instanceof Tensor) {
            if (!this.shape.equals(other.shape)) // todo: should be overridable
                throw new Error(`Shape mismatch: Tensor.${name}() expects tensors of the same shape.`);
            core_fn_prw(this.get_ptr(), other.get_ptr(), this.data.length);
        }
        else if (typeof other === 'number') core_fn_scl(this.get_ptr(), other, this.data.length);
        else throw new Error(`Type mismatch: Tensor.${name}() expects a Tensor or number.`);
        return this;
    }

    private unary_op(name: string, core_fn: Function) {
        core_fn(this.get_ptr(), this.data.length);
        return this;
    }

    // binary operations (pairwise/scalar)
    public add = (other: Tensor | number) => this.binary_op('add', core._add_prw, core._add_scl, other);
    public sub = (other: Tensor | number) => this.binary_op('sub', core._sub_prw, core._sub_scl, other);
    public div = (other: Tensor | number) => this.binary_op('div', core._div_prw, core._div_scl, other);
    public mul = (other: Tensor | number) => this.binary_op('mul', core._mul_prw, core._mul_scl, other);

    // unary operations
    public relu = () => this.unary_op('relu', core._act_relu);
    public relu_simd = () => this.unary_op('relu', core._act_relu_simd); // todo: remove. no perf gains
    public tanh = () => this.unary_op('tanh', core._act_tanh);
    
    // dot product/standard matmul on tensors 
    public matmul(b: Tensor): Tensor {
        if (!(b instanceof Tensor)) throw new Error('Tensor.matmul() expects a tensor.');

        // todo this can cause problems - when the tensors are of rank <= 2 
        // const nmat_a = this.shape.mat_flat()[0];
        // const nmat_b = b.shape.mat_flat()[0];
        const nmat_a = this.shape.get_ndim() > 2 ? this.shape.mat_flat()[0] : 1;
        const nmat_b =    b.shape.get_ndim() > 2 ?    b.shape.mat_flat()[0] : 1;

        // console.log(nmat_a, nmat_b)

        if (this.shape.get_cols() !== b.shape.get_rows() ||
            nmat_a > 1 && nmat_b > 1 && nmat_a != nmat_b) {
            throw new Error(`Cannot multiply matrices of shape [${this.shape.get_mat_shape()}] and [${b.shape.get_mat_shape()}]`);
        }

        const rows_a = this.shape.get_rows();
        const cols_a = this.shape.get_cols();
        const rows_b = b.shape.get_rows();
        const cols_b = b.shape.get_cols();

        // todo validate
        const high_level_shape = this.shape.get_ndim() > b.shape.get_ndim()
            ? this.shape.slice(0, this.shape.get_ndim() - 2)
            : b.shape.slice(0, b.shape.get_ndim() - 2);

        const result_shape = new Shape(...high_level_shape, rows_a, cols_b); // todo kinda want to do this with .get_axis_shape
        const result = tensor(result_shape);

        core._mul_tns(
            this.get_ptr(), rows_a, cols_a, nmat_a,
            b.get_ptr(),    rows_b, cols_b, nmat_b,
            result.get_ptr()
        );

        return result;
    }

    public dot(b: Tensor): Tensor {
        if (!(b instanceof Tensor)) throw new Error('Tensor.dot() expects a tensor.');

        if (this.shape.get_cols() !== b.shape.get_rows()) {
            throw new Error(`Cannot compute dot product of tensors of shapes [${this.shape.get_mat_shape()}] and [${b.shape.get_mat_shape()}]`);
        }

        const ndim_a = this.shape.get_ndim();
        const ndim_b = b.shape.get_ndim();

        const result_shape = new Shape(
            ...this.shape.slice(0, ndim_a - 1),
            ...b.shape.slice(0, ndim_b - 2), b.shape[ndim_b - 1]);

        console.log(result_shape);
    }

    // usability methods
    public toString = () => this.to_str();
    public to_str(num_width = 10, space_before = 0) {
        const ndim = this.shape.get_ndim();

        // empty arr, vec, mat
        if (ndim === 0) return '[]';
        if (ndim === 1) return `[ ${this.data.join(', ')} ]`;
        if (ndim == 2)  return mat_to_string(this, num_width, space_before);

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
