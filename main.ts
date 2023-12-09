class Shape extends Array {

    constructor(...shape: number[]) {
        // special case: arr length == 1:
        // would create an array of size shape[0]
        if (shape.length === 1) {
            super(1);
            this[0] = shape[0];
            return;
        }

        super(...shape);
    }

    get_nelem(): number {
        if (this.length === 0) return 0;
        return this.reduce((acc, cur) => acc *= cur, 1);
    }

    get_ndim(): number {
        return this.length;
    }

    get_axis_size(axis_index: number): number {
        let axis_size = this[axis_index];
        if (axis_size === undefined) return 1;
        return axis_size;
    }

    // returns the size of the second-to-last axis, or 1 if that axis does not exits 
    get_rows(): number {
        return this.get_axis_size(this.get_ndim() - 2);
    }

    // returns the size of the last axis, or 1 if that axis does not exits 
    get_cols(): number {
        return this.get_axis_size(this.get_ndim() - 1);
    }

    get_mat_shape(): Shape {
        return new Shape(this.get_rows(), this.get_cols());
    }

    // this is a work in progress
    check_matmul_compat(other: Shape): void {
        if (this.get_cols() !== other.get_rows()) {
            throw new Error(`Cannot multiply matrices of shape [${this.get_mat_shape()}] and [${other.get_mat_shape()}]`);
        }
    }

    get_index(row: number, col: number): number {
        const cols = this.get_cols();
        return row * cols + col;
    }

    flat_shape(n?: number): Shape {
        if (n === undefined) n = this.get_ndim() - 1;
        if (this.get_ndim() <= n) throw new Error("Can't flatten this much.");
        let new_axis_size = 1;

        for (let i = 0; i < n + 1; i++) {
            new_axis_size *= this.get_axis_size(i);
        }

        return new Shape(new_axis_size, ...this.slice(n + 1));
    }

    // flatten to such an extent that we get an array of matrices
    mat_flat(): Shape {
        const amount = Math.max(this.get_ndim() - 3, 0);
        return this.flat_shape(amount);
    }

    get_axis_iterable(n: number) {
        // squish shape down to lower number of dimensions
        const flattened: Shape = this.flat_shape(n);

        // the the shape of each element of the relevant axis
        const new_shape = flattened.slice(1) as Shape;

        // calculate how many elements are in each sub-shape
        const stepover = new_shape.get_nelem() || 1;
        const n_steps = flattened[0];

        return {
            [Symbol.iterator]() {
                return {
                    last: n_steps,
                    i: 0,
        
                    next() {
                        if (this.i === this.last) {
                            return { done : true };
                        }
        
                        const value = this.i * stepover;
                        this.i++;
        
                        return { value, done: false };
                    }
                };
            }
        }
    }
}

class Tensor {
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

    mul(other: Tensor) {
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

    *get_axis_iterable(n: number) {

        // todo extract into a function (duplicate code in shape.get_axis_iterable)
        const flattened: Shape = this.shape.flat_shape(n);
        const new_shape = flattened.slice(1) as Shape;
        const stepover = new_shape.get_nelem() || 0;
        console.log(new_shape)

        for (let index of this.shape.get_axis_iterable(n)) {
            // creating a tensor that references the data of this tensor
            yield new Tensor(new_shape, this.data.subarray(index, index + stepover));
        }
    }
}

function tensor(shape: number[] | Shape, data?: number[]) {
    const _shape = new Shape(...shape); // clone shape
    const _data = new Float32Array(_shape.get_nelem());

    if (data !== undefined)
        _data.set(data);

    return new Tensor(_shape, _data);
}

// matrix with 2 rows and 3 columns
let t1 = tensor([2, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);

// // matrix with 3 rows and 2 columns
// let t2 = new Tensor([3, 2], [1, 2, 3, 4, 5, 6]);

for (let mat of t1.get_axis_iterable(0)) {
    console.log(mat);
}

for (let mat of t1.get_axis_iterable(1)) {
    console.log(mat);
    mat.data[0] = 400;
}
