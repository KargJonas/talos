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

function mat_to_string(_mat: Tensor, num_width = 10, space_before = 0) {
    if (_mat.shape.get_ndim() !== 2)
        throw new Error(`Cannot print tensor of shape [${_mat.shape}] as matrix.`);

    // capping the length of the numbers to the numbers of decimal places
    const decimal_places = num_width - 5;
    const exp = Math.pow(10, decimal_places);

    // todo: this seems to apply the operations to the shape instead of the data !???
    //   seems like an issue with the clone() operation
    const mat = _mat.clone().mul(exp, true).floor(true).div(exp, true);

    _mat.clone().mul(5, true).print_info();

    const lines: string[] = [];
    const cols = mat.get_cols();
    const rows = mat.get_rows();
    const col_stride = mat.strides[1];
    const row_stride = mat.strides[0];

    for (let r = 0; r < rows; r++) {
        const vals: number[] = [];

        for (let c = 0; c < cols; c++) {
            const index = r * row_stride + c * col_stride;
            vals.push(mat.data[index]);
        }

        const padding_left = r !== 0 ? " ".repeat(space_before) : "";
        lines.push(`${padding_left}[ ${vals.join(", ")} ]`);
    }

    return lines.join("\n");
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
