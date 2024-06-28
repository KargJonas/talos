import { Tensor } from "./Tensor";

// usability methods
export default function tensor_to_string(a: Tensor, num_width = 5, space_before = 0) {   
    switch (a.rank) {
        case 0: return "[]";
        case 1: return vec_to_string(a, num_width);
        case 2: return mat_to_string(a, num_width, space_before);
    }

    const strings: string[] = [];

    for (const element of a.get_axis_iterable(0)) {
        strings.push(tensor_to_string(element, num_width, space_before + 2)!);
    }

    return `[ ${strings.join(",\n\n" + " ".repeat(space_before + 2))} ]`;
}

function vec_to_string(vec: Tensor, n_decimals: number) {
    if (vec.shape[0] === 1) return `[ ${vec.data[vec.offset]} ]`;

    const n_integer = Math.floor(vec.max()).toString().length;
    const cols = vec.cols;
    const col_stride = vec.strides[0];
    const offset = vec.offset;
    const vals: string[] = [];

    for (let c = 0; c < cols; c++) {
        const index = offset + c * col_stride;
        const val = vec.data[index];
        const val_floor = Math.floor(val);
        const str = val === val_floor ? val.toString() : val.toFixed(n_decimals);
        const padding_amount = Math.max(n_integer - val_floor.toString().length, 0);

        vals.push(" ".repeat(padding_amount) + str);
    }

    return `[ ${vals.join(", ")} ]`;
}

function mat_to_string(mat: Tensor, n_decimals: number, space_before: number) {
    // amount of digits in the integer part of the largest number
    const n_integer = Math.floor(mat.max()).toString().length;
    const lines: string[] = [];
    const cols = mat.cols;
    const rows = mat.rows;
    const col_stride = mat.strides[1];
    const row_stride = mat.strides[0];
    const offset = mat.offset;
    const exp = Math.pow(10, n_decimals);

    const m = mat.clone();
    let only_integers = true;
    for (let i = 0; i < m.data.length; i++) {
        if (m.data[i] !== Math.floor(m.data[i])) {
            only_integers = false;
            break;
        }
    }

    const max_length = n_integer + 1 + (only_integers ? 0 : n_decimals);

    for (let r = 0; r < rows; r++) {
        const vals: string[] = [];

        for (let c = 0; c < cols; c++) {
            const index = offset + r * row_stride + c * col_stride;
            const val = Math.floor(mat.data[index] * exp) / exp;
            const str = val.toString();
            const separator = c < cols - 1 ? ", " : "";
            const padding_right = " ".repeat(Math.max(0, max_length - str.length));

            vals.push(`${str}${separator}${padding_right}`);
        }

        const padding_left = r !== 0 ? " ".repeat(space_before) : "";
        // lines.push(`${padding_left}[ ${vals.join(", ")} ]`);
        lines.push(`${padding_left}[ ${vals.join("")}]`);
    }

    return `[${lines.join("\n ")}]`;
}
