import core from './core/build';
import { Shape } from './shape';

export class Tensor {
    shape: Shape; // [outermost axis, ..., rows, cols]
    data: Float32Array;

    constructor(shape: Shape, data: Float32Array) {
        this.shape = shape;
        this.data = data;
    }

    clone(): Tensor {
        const new_tensor = tensor(this.shape);
        new_tensor.data.set(this.data);
        return new_tensor;
    }

    *get_axis_iterable(n: number) {
        // todo extract into a function (duplicate code in shape.get_axis_iterable)
        const flattened: Shape = this.shape.flatten(n);
        const new_shape = flattened.slice(1) as Shape;
        const stepover = new_shape.get_nelem() || 0;

        for (let index of this.shape.get_axis_iterable(n)) {
            // creating a tensor that references the data of this tensor
            yield new Tensor(new_shape, this.data.subarray(index, index + stepover));
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

    rand_f(min = -1, max = 1) {
        core._rand_f(this.get_ptr(), this.data.length, min, max);
        return this;
    }

    rand_i(min = -1, max = 1) {
        core._rand_i(this.get_ptr(), this.data.length, min, max);
        return this;
    }

    basic_op(name: string, core_fn_prw: Function, core_fn_scl: Function, other: Tensor | number) {
        if (other instanceof Tensor) {
            // todo: should be overridable
            if (!this.shape.equals(other.shape))
                throw `Shape mismatch: Tensor.${name}() expects tensors of the same shape.`;

            core_fn_prw(this.get_ptr(), other.get_ptr(), this.data.length);
        }

        else if (typeof other === 'number') core_fn_scl(this.get_ptr(), other, this.data.length);
        else throw `Type mismatch: Tensor.${name}() expects a Tensor or number.`;
        return this;
    }

    // pairwise/scalar operations
    add = (other: Tensor | number) => this.basic_op('add', core._add_prw, core._add_scl, other);
    sub = (other: Tensor | number) => this.basic_op('sub', core._sub_prw, core._sub_scl, other);
    div = (other: Tensor | number) => this.basic_op('div', core._div_prw, core._div_scl, other);
    mul = (other: Tensor | number) => this.basic_op('mul', core._mul_prw, core._mul_scl, other);
    
    // dot product/standard matmul
    dot(other: Tensor): Tensor {
        if (!(other instanceof Tensor)) throw 'Tensor.dot() expects a tensor.';
        this.shape.check_matmul_compat(other.shape);

        const rows = this.shape.get_rows();
        const cols = this.shape.get_cols();
        const rows_other = other.shape.get_rows();
        const cols_other = other.shape.get_cols();
        const result = tensor([rows, cols_other]);

        core._mul_mat(
            this.get_ptr(),  rows,       cols,
            other.get_ptr(), rows_other, cols_other,
            result.get_ptr());

        return result;
    }

    // usability methods
    
    private mat_to_string(num_width = 10, space_before = 0) {
        if (this.shape.get_ndim() !== 2)
            throw new Error(`Cannot print tensor of shape [${this.shape}] as matrix.`);

        const rows = this.shape.get_rows();
        const cols = this.shape.get_cols();

        let s = '[ ';

        for (let r = 0; r < rows; r++) {
            if (r !== 0) s += ' '.repeat(space_before + 2);

            for (let c = 0; c < cols; c++) {
                const index = this.shape.get_index(r, c);

                // -5 because of: space, comma, sign, dot, and at least one digit
                let p = String(this.data[index].toFixed(num_width - 5));

                // commas, newlines, padding
                if (c !== cols - 1 || r !== rows - 1) p += ',';
                if (c !== cols - 1) p = p.padEnd(num_width);
                if (c === cols - 1 && r !== rows - 1) p += '\n';
                s += p;
            }
        }

        return s + ' ]';
    }

    toString(num_width = 10, space_before = 0) {
        const ndim = this.shape.get_ndim();

        if (ndim === 0) return '[]';

        // vectors
        if (ndim === 1) return `[ ${this.data.join(', ')} ]`;

        // matrices
        if (ndim == 2) {
            return this.mat_to_string(num_width, space_before);
        }

        // whatever lies beyond
        let strings: string[] = [];
        for (const element of this.get_axis_iterable(0)) {
            strings.push(element.toString(num_width, space_before + 2)!);
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
