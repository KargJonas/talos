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

    get_axis_size(axis_index: number): number {
        let axis_size = this[axis_index];
        if (axis_size === undefined) return 1;
        return axis_size;
    }

    public get_ndim = () => this.length;

    // returns the size of the second-to-last axis, or 1 if that axis does not exits 
    public get_rows = () => this.get_axis_size(this.get_ndim() - 2);

    // returns the size of the last axis, or 1 if that axis does not exits 
    public get_cols = () => this.get_axis_size(this.get_ndim() - 1);

    public get_mat_shape = () => new Shape(this.get_rows(), this.get_cols());

    // returns true if two shapes are identical
    equals(other: Shape): boolean {
        for (let i = 0; i < this.length; i++) {
            if (!other[i] || this[i] !== other[i]) return false;
        }
    
        return true;
    }

    // computes number of indices to step over for each element in each axis
    get_strides(): number[] {
        const strides = Array(this.get_ndim()).fill(1);

        for (let i = this.get_ndim() - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * this[i + 1];
        }

        return strides;
    }

    // flattens the array by n levels. e.g.: n=1 -> [2, 4, 3] turns to [8, 3]
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

    // get shape of elements in an axis (n determines the level of nesting)
    get_axis_shape(n: number): Shape {
        if (n >= this.get_ndim()) throw new Error(`Cannot get ${ordinal_str(n)} axis.`);
        return new Shape(...this.slice(n - this.get_ndim()));
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
