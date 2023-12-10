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
        console.log(new_shape)

        for (let index of this.shape.get_axis_iterable(n)) {
            // creating a tensor that references the data of this tensor
            yield new Tensor(new_shape, this.data.subarray(index, index + stepover));
        }
    }

    flatten(n: number): Tensor {
        return new Tensor(this.shape.flatten(n), this.data);
    }

    mat_flat(): Tensor {
        return new Tensor(this.shape.mat_flat(), this.data);
    }

    mat_to_string(num_width = 10, space_before = 0) {
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

    mul(other: Tensor): Tensor {
        this.shape.check_matmul_compat(other.shape);

        const rows = this.shape.get_rows();
        const cols = this.shape.get_cols();
        const cols_other = other.shape.get_cols();
        const result = tensor([rows, cols_other]);

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols_other; c++) {
                const index = result.shape.get_index(r, c);
            
                 for (let i = 0; i < cols; i++) {
                    const ia = this.shape.get_index(r, i);
                    const ib = other.shape.get_index(i, c);
                    result.data[index] += this.data[ia] * other.data[ib];
                 }
            }
        }

        return result;
    }

    rand(min = -1, max = 1) {
        const range = max - min;

        this.data.forEach((_, index) =>
            this.data[index] = Math.random() * range + min);

        return this;
    }

    rand_int(min = -1, max = 1) {
        min = Math.ceil(min);
        max = Math.floor(max);

        this.data.forEach((_, index) =>
            this.data[index] = ((Math.random() * (max - min + 1)) | 0) + min);

        return this;
    }
}

export default function tensor(shape: number[] | Shape, data?: number[]) {
    const _shape = new Shape(...shape); // clone shape
    const _data = new Float32Array(_shape.get_nelem());

    if (data !== undefined)
        _data.set(data);

    return new Tensor(_shape, _data);
}
