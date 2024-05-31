import { Tensor } from "./Tensor";
import core from "./core/build";

export const set_rand_seed = (n: number) => core._rand_seed(n);

export function create_farr(nelem: number): Float32Array {
    const ptr = core._alloc_farr(nelem);
    return new Float32Array(core.memory.buffer, ptr, nelem);
}

export function create_starr(nelem: number): Int32Array {
    const ptr = core._alloc_starr(nelem);
    return new Int32Array(core.memory.buffer, ptr, nelem);
}

export function check_row_col_compat(a: Tensor, b: Tensor) {
    if (a.get_cols() !== b.get_rows())
        throw new Error(`Cannot multiply tensors of shape [${a.shape}] and [${b.shape}]`);
}

export function ordinal_str(n: number): string {
    const last_digit = n % 10;
    const last_two_digits = n % 100;

    const suffix = (last_digit == 1 && last_two_digits != 11 ? "st" :
        last_digit == 2 && last_two_digits != 12 ? "nd" :
            last_digit == 3 && last_two_digits != 13 ? "rd" : "th");

    return `${n}${suffix}`;
}

export function get_row_major(shape: number[]): number[] {
    let stride = 1;
    const strides = new Array(shape.length);
    strides.fill(1);

    for (let i = shape.length - 2; i >= 0; i--) {
        strides[i] = stride *= shape[i + 1];
    }

    return strides;
}
