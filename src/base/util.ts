import {RawTensor} from "./RawTensor.ts";
import core from "./core/build";

export type NDArray = (NDArray | number)[];

export const set_rand_seed = (n: number) => core._rand_seed(n);

export function check_row_col_compat(a: RawTensor, b: RawTensor) {
    if (a.cols !== b.rows)
        throw new Error(`Cannot multiply tensors of shape [${a.shape}] and [${b.shape}]`);
}

/**
 * Finds the row-major strides of a tensor with the specified shape
 * @param shape Shape of tensor
 */
export function get_strides_row_major(shape: number[]): number[] {
    let stride = 1;
    const strides = new Array(shape.length);
    strides.fill(1);

    for (let i = shape.length - 2; i >= 0; i--) {
        strides[i] = stride *= shape[i + 1];
    }

    return strides;
}

/**
 * Flattens an n-dimensional array while validating that the sizes of all axes are regular.
 * @param item A nested n-dimensional array
 */
export function flatten(item: NDArray) {
    if (item.length === 0) throw new Error("Axes with size 0 not allowed.");

    let last_type = typeof item[0];
    let last_size = typeof item[0] === "number" ? -1 : item[0].length;
    let first = true;
    const flat: number[] = [];
    const shape: number[] = [item.length];

    for (const sub_element of item) {
        const current_type = typeof sub_element;
        const current_size = typeof sub_element === "number" ? -1 : sub_element.length;
        if (last_type !== current_type) throw new Error("Found type irregularities in the provided array.");
        if (last_size !== current_size) throw new Error("Found shape irregularities in the provided array.");

        if (current_type === "number") {
            flat.push(sub_element as number);
            continue;
        }

        const [_shape, _flat] = flatten(sub_element as NDArray);
        flat.push(..._flat);
        if (first) shape.push(..._shape);

        last_type = current_type;
        last_size = current_size;
        first = false;
    }

    return [shape, flat];
}
