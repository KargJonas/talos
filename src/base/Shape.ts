import { ordinal_str } from "./util";
import core from "./core/build";

export default class Shape extends Int32Array {
    constructor(shape: Int32Array | number[] | Shape, attached = false) {
        if (attached) {
            if (!(shape instanceof Int32Array)) throw new Error("Shape must be Int32Array!");
            super(core.memory.buffer, shape.byteOffset, shape.length);
        } else {
            // create a detatched shape (not bound to a tensor)
            super(shape.length);
            this.set(shape);
        }
    }
    
    // public get_ndim = () => this.length;
    // public get_rows = () => this.get_axis_size(this.ndim - 2);
    // public get_cols = () => this.get_axis_size(this.ndim - 1);
    // public get_mat_shape = () => new Shape([this.rows, this.cols]);
    // public get_axis_size = (axis_index: number) => this[axis_index] === undefined ? 1 : this[axis_index];
    // public detach = () => new Shape(this);
    // public is_scalar = () => this.length === 1 && this[0] === 1;

    public get ndim() { return this.length; }
    public get rows() { return this.get_axis_size(this.ndim - 2); }
    public get cols() { return this.get_axis_size(this.ndim - 1); }
    public get mat_shape() { return new Shape([this.rows, this.cols]); }
    public get is_scalar() { return this.length === 1 && this[0] === 1; }

    public detach = () => new Shape(this);
    public get_axis_size = (axis_index: number) => this[axis_index] === undefined ? 1 : this[axis_index];

    public get nelem(): number {
        if (this.length === 0) return 0;
        return this.reduce((acc, cur) => acc *= cur, 1);
    }

    // returns true if two shapes are identical
    equals(other: Shape): boolean {
        if (this.length !== other.length) return false;

        for (let i = 0; i < this.length; i++) {
            if (this[i] !== other[i] || other[i] === undefined) {
                return false;
            }
        }
    
        return true;
    }

    broadcastable(other: Shape): boolean {
        const min_rank = Math.min(this.ndim, other.ndim);
        const a = this.detach().reverse();
        const b = other.detach().reverse();

        for (let i = 0; i < min_rank; i++) {
            if (a[i] !== b[i] && a[i] !== 1 && b[i] !== 1) return false;
        }

        return true;
    }

    broadcast(other: Shape): Shape {
        // check if broadcasting is possible
        if (!this.broadcastable(other))
            throw new Error(`Shape mismatch: Cannot broadcast tensor of shape [${this}] with [${other}].`);

        const max_rank = Math.max(this.ndim, other.ndim);
        const new_shape: number[] = [];
        const a = new Shape(this.detach().reverse());
        const b = new Shape(other.detach().reverse());

        for (let i = 0; i < max_rank; i++) {
            new_shape.unshift(Math.max(a.get_axis_size(i), b.get_axis_size(i)));
        }

        return new Shape(new_shape);
    }

    // computes number of indices to step over for each element in each axis
    get_strides(): number[] {
        const strides = Array(this.ndim).fill(1);

        for (let i = this.ndim - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * this[i + 1];
        }

        return strides;
    }

    /**
     * Appends n 1s to the left to reach the desired rank.
     * If the rank of the shape is already >= rank, nothing will be done.
     * @param rank Desired rank
     * @returns A new shape with 1s appended to the left.
     */
    expand_left(rank: number): Shape {
        const new_shape = [...this];
        const n = Math.max(0, rank - this.ndim);
        for (let i = 0; i < n; i++) new_shape.unshift(1); 
        return new Shape(new_shape);
    }

    /**
     * flattens (or unflattens) a shape to a specific rank
     * while preserving the innermost shape and overall number of elements
     * [2,2,3] --flatten(2)--> [4,3]
     * if the desired rank is higher than the current rank, the shape will be extended
     * with new size-1 axes to the left:
     * [2,3] --flatten(3}--> [1,2,3]
     * @param rank Desired rank
     * @returns A new shape that is based on this shape with the desired rank
     */
    flatten(rank: number) {
        const current_rank = this.ndim;
        const amount = Math.abs(current_rank - rank);

        // flatten
        // combines n axes from the left into a single axes through mult
        if (rank < current_rank) {
            let new_axis_size = 1;
            for (let i = 0; i < amount + 1; i++) new_axis_size *= this.get_axis_size(i);
            return new Shape([new_axis_size, ...Array.from(this).slice(amount + 1)]);
        }

        // unflatten
        if (rank > current_rank) {
            return this.expand_left(rank);
        }
        
        return this.detach();
    }

    /**
     * Returns the index of an element of a tensor as well as that element"s shape.
     * @param loc Location vector of the desired element/slice
     * @returns A 2-Tuple containing the index and shape
     */
    get_index(...loc: number[]): [number, Shape] {
        if (loc.length > this.ndim) throw new Error(`Location [${loc}] is too specific for shape [${this}]`);

        const strides = this.get_strides();
        const index = loc.reduce((acc, l_axis, i) => {
            if (l_axis >= this[i]) throw new Error(`Location [${loc}] out of bounds of shape [${this}]`);
            return acc + l_axis * strides[i];
        }, 0);

        // const new_shape = new Shape(this.detach().slice(loc.length));
        const new_shape = new Shape(this.get_axis_shape(this.length - 1));
        return [index, new_shape];
    }

    // get shape of elements in an axis (n determines the level of nesting)
    get_axis_shape(n: number): number[] {
        if (n >= this.ndim) throw new Error(`Cannot get ${ordinal_str(n)} axis.`);
        return [...this].slice(n - this.ndim); // todo fix. should work with .subarry
    }
}
