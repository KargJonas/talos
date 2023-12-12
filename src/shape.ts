export class Shape extends Array {

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

    equals(other: Shape): boolean {
        for (let i = 0; i < this.length; i++) {
            if (!other[i] || this[i] !== other[i]) return false;
        }
    
        return true;
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

    flatten(n?: number): Shape {
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
        return this.flatten(amount);
    }

    *get_axis_iterable(n: number) {
        // squish shape down to lower number of dimensions
        const flattened: Shape = this.flatten(n);

        // the the shape of each element of the relevant axis
        const new_shape = flattened.slice(1) as Shape;

        // calculate how many elements are in each sub-shape
        const stepover = new_shape.get_nelem() || 1;
        const n_steps = flattened[0];

        for (let i = 0; i < n_steps; i++) {
            const value = i * stepover;
            yield value;
        }
    }
}

export default function shape(...s: number[]): Shape {
    return new Shape(...s);
}
