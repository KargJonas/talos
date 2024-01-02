import { Tensor } from "./Tensor";
import core from "./core/build";

export const core_ready = new Promise<null>((resolve) => {
    core.onRuntimeInitialized = () => {
        core.memory = new Uint8Array(core.HEAPU8.buffer);
        resolve(null);
    }
});

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
    const last_digit = n % 10,
          last_two_digits = n % 100;

    const suffix = (last_digit == 1 && last_two_digits != 11 ? "st" :
                    last_digit == 2 && last_two_digits != 12 ? "nd" :
                    last_digit == 3 && last_two_digits != 13 ? "rd" : "th");

    return `${n}${suffix}`;
}
