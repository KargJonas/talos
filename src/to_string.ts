import { Tensor } from "./Tensor";

// usability methods
export default function tensor_to_string(a: Tensor, num_width = 5, space_before = 0) {   
    switch (a.get_rank()) {
        case 0: return "[]";
        case 1: return `[ ${a.data.join(", ")} ]`;
        case 2: return mat_to_string(a, num_width, space_before);
    }

    // hidim tensors
    const strings: string[] = [];

    for (const element of a.get_axis_iterable(0)) {
        strings.push(tensor_to_string(element, num_width, space_before + 2)!);
    }

    return `[ ${strings.join(",\n\n" + " ".repeat(space_before + 2))} ]`;
}

function mat_to_string(mat: Tensor, n_decimals: number, space_before: number) {
    if (mat.shape.get_ndim() !== 2)
        throw new Error(`Cannot print tensor of shape [${_mat.shape}] as matrix.`);

    // largest amount of digits in the integer part of the number
    const n_integer = Math.floor(mat.max()).toString().length;

    const lines: string[] = [];
    const cols = mat.get_cols();
    const rows = mat.get_rows();
    const col_stride = mat.strides[1];
    const row_stride = mat.strides[0];
    const offset = mat.get_offset();

    for (let r = 0; r < rows; r++) {
        const vals: string[] = [];

        for (let c = 0; c < cols; c++) {
            const index = offset + r * row_stride + c * col_stride;
            const val = mat.data[index];
            const val_floor = Math.floor(val);

            // if value is integer, omit trailing zeros, otherwise use fixed nr of digits
            const str = val === val_floor ? val.toString() : val.toFixed(n_decimals);

            // compute amount of left padding
            const padding_amount = Math.max(n_integer - val_floor.toString().length, 0);

            vals.push(" ".repeat(padding_amount) + str);
        }

        const padding_left = r !== 0 ? " ".repeat(space_before) : "";
        lines.push(`${padding_left}[ ${vals.join(", ")} ]`);
    }

    return lines.join("\n");
}
