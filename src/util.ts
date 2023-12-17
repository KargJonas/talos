import core from './core/build';
import { Tensor } from './tensor';

export const core_ready = new Promise<null>((resolve) => {
    core.onRuntimeInitialized = () => {
        core.memory = new Uint8Array(core.HEAPU8.buffer);
        resolve(null);
    }
});

export function ordinal_str(n: number): string {
    const last_digit = n % 10,
          last_two_digits = n % 100;

    const suffix = (last_digit == 1 && last_two_digits != 11 ? "st" :
                    last_digit == 2 && last_two_digits != 12 ? "nd" :
                    last_digit == 3 && last_two_digits != 13 ? "rd" : "th");

    return `${n}${suffix}`;
}

export function mat_to_string(mat: Tensor, num_width = 10, space_before = 0) {
    if (mat.shape.get_ndim() !== 2) {
        throw new Error(`Cannot print tensor of shape [${mat.shape}] as matrix.`);
    }

    const rows = mat.shape.get_rows();
    const cols = mat.shape.get_cols();

    let only_ints = true;
    let maxlen = 1;
    for (let i = 0; i < mat.data.length; i++) {
        if (mat.data[i] !== (mat.data[i] | 0)) {
            only_ints = false;
        }

        maxlen = Math.max(maxlen, String(mat.data[i] | 0).length);
    }

    // todo

    let s = '[';

    for (let r = 0; r < rows; r++) {
        if (r !== 0) s += ' '.repeat(space_before + 1);
        let row_string = '';

        for (let c = 0; c < cols; c++) {
            const [index] = mat.shape.get_index(r, c);
            const value = mat.data[index];

            // -5 because of: space, comma, sign, dot, and at least one digit
            let p = String(only_ints ? value : value.toFixed(num_width - 5));
            if (value > 0) p = ` ${p}`;

            // commas, newlines, padding
            if (c !== cols - 1) p += ',';
            if (c !== cols - 1) p = p.padEnd(num_width);
            row_string += p;
        }

        s += `[${row_string}]`;
        if (r !== rows - 1) s += '\n';
    }

    return s + ']';
}
