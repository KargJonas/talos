import { Tensor } from "./Tensor";

// usability methods
export default function tensor_to_string(a: Tensor, num_width = 10, space_before = 0) {   
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

function mat_to_string(mat: Tensor, num_width: number, space_before: number) {
    if (mat.shape.get_ndim() !== 2)
        throw new Error(`Cannot print tensor of shape [${_mat.shape}] as matrix.`);

    // capping the length of the numbers to the numbers of decimal places
    const decimal_places = num_width - 5;
    // const exp = Math.pow(10, decimal_places);
    // const mat = _mat.clone().mul(exp, true).floor(true).div(exp, true);

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
            const val = mat.data[index].toFixed(5);
            vals.push(val);
        }

        const padding_left = r !== 0 ? " ".repeat(space_before) : "";
        lines.push(`${padding_left}[ ${vals.join(",\t")} ]`);
    }

    return lines.join("\n");
}

function limit_decimals(n: number, n_decimals: number) {
    const exp = Math.pow(10, n_decimals);
    return Math.floor(n * exp) / exp;
}

// function mat_to_string(mat: Tensor, num_width = 10, space_before = 0) {

//     if (mat.shape.get_ndim() !== 2)
//         throw new Error(`Cannot print tensor of shape [${mat.shape}] as matrix.`);

//     const rows = mat.shape.get_rows();
//     const cols = mat.shape.get_cols();

//     const decimal_places = num_width - 5;
//     const exp = Math.pow(10, decimal_places);

//     // cap number of decimal places 
//     const _mat = mat.clone().mul(exp, true).floor(true).div(exp, true);

//     let only_ints = true;
//     let maxlen = 1;
//     for (let i = 0; i < _mat.data.length; i++) {
//         if (_mat.data[i] !== (_mat.data[i] | 0)) only_ints = false;
//         maxlen = Math.max(maxlen, String(_mat.data[i] | 0).length);
//     }

//     let s = "[";

//     for (let r = 0; r < rows; r++) {
//         if (r !== 0) s += " ".repeat(space_before + 1);
//         let row_string = "";

//         for (let c = 0; c < cols; c++) {
//             const [index] = _mat.shape.get_index(r, c);
//             const value = _mat.data[index];

//             // -5 because of: space, comma, sign, dot, and at least one digit
//             // let p = String(only_ints ? value : value.toFixed(num_width - 5));
//             let p = String(value);
//             if (value > 0) p = ` ${p}`;

//             // commas, newlines, padding
//             if (c !== cols - 1) p += ",";
//             else p += " ";
//             if (c !== cols - 1) p = p.padEnd(num_width);
//             row_string += p;
//         }

//         s += `[${row_string}]`;
//         if (r !== rows - 1) s += "\n";
//     }

//     return s + "]";
// }
