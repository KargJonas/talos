import { ordinal_str } from "./util";

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

    get_strides(): number[] {
        const strides = Array(this.get_ndim()).fill(1);

        for (let i = this.get_ndim() - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * this[i + 1];
        }

        return strides;
    }

    /**
     * Returns the index of an element of a tensor as well as that element's shape.
     * @param loc Location vector of the desired element/slice
     * @returns A 2-Tuple containing the index and shape
     */
    get_index(...loc: number[]): [number, Shape] {
        if (loc.length > this.get_ndim()) throw new Error(`Location [${loc}] is too specific for shape [${this}]`);

        const strides = this.get_strides();
        const index = loc.reduce((acc, l_axis, i) => {
            if (l_axis >= this[i]) throw new Error(`Location [${loc}] out of bounds of shape [${this}]`);
            return acc + l_axis * strides[i];
        }, 0);

        const new_shape = this.slice(loc.length);
        return [index, shape(...new_shape)];
    }

    flatten(n?: number): Shape {
        if (n === undefined) n = this.get_ndim() - 1;
        if (this.get_ndim() <= n) throw new Error("Can't flatten this much.");
        let new_axis_size = 1;

        for (let i = 0; i < n + 1; i++) {
            new_axis_size *= this.get_axis_size(i);
        }

        return shape(new_axis_size, ...this.slice(n + 1));
    }

    // flatten to such an extent that we get an array of matrices
    mat_flat(): Shape {
        const amount = Math.max(this.get_ndim() - 3, 0);
        return this.flatten(amount);
    }

    /**
     * Returns the shape of the elements of an axis.
     * @param depth Depth of the axis - axis nr 0 is the "topmost" or least deeply nested axis.
     * @returns the shape of the elements of an axis.
     */
    get_axis_shape(depth: number): Shape {
        if (depth >= this.get_ndim()) throw new Error(`Cannot get ${ordinal_str(depth)} axis.`);
        return new Shape(...this.slice(depth - this.get_ndim()));
    }

    *get_axis_iterable(n: number) {
        const stride = this.get_strides()[n];
        const n_elem = this.get_nelem();

        for (let i = 0; i < n_elem; i += stride) {
            yield i;
        }
    }
}

export default function shape(...s: number[]): Shape {
    return new Shape(...s);
}
