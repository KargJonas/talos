import { Parameter } from "../autograd/node_operations.ts";
import Tensor from "../tensor.ts";
import {RawTensor} from "./raw_tensor.ts";
import Shape from "./shape.ts";

const bold = "\x1b[1m";
const reset = "\x1b[0m";
const purple = "\x1b[35m";
const grey = "\x1b[1;30m";
const green = "\x1b[32m";
const blue = "\x1b[1;34m";
const underline = "\x1b[4m";

// usability methods
export function tensor_to_string(a: RawTensor, num_width = 5, space_before = 0) {
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

function vec_to_string(vec: RawTensor, n_decimals: number) {
    // todo: make this configurable
    // if (vec.is_scalar) return `[ ${(vec.item | 0) === vec.item ? vec.item.toString() : vec.item.toFixed(n_decimals)} ]`;
    if (vec.is_scalar) {
        if ((vec.item | 0) === vec.item) return `[ ${vec.item} ]`;
        const exp = Math.log10(vec.item) | 0;
        return `[ ${((exp < 4 - n_decimals || exp > 21) && n_decimals !== 0) ? vec.item.toExponential(n_decimals) : vec.item.toFixed(n_decimals)} ]`;
    }
 
    const n_integer = Math.floor(Math.max(...vec.data)).toString().length;
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

function mat_to_string(mat: RawTensor, n_decimals: number, space_before: number) {
    // amount of digits in the integer part of the largest number
    const n_integer = Math.floor(Math.max(...mat.data)).toString().length;
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
    const has_negative_vals = Math.min(...mat.data) < 0;

    for (let r = 0; r < rows; r++) {
        const vals: string[] = [];

        for (let c = 0; c < cols; c++) {
            const index = offset + r * row_stride + c * col_stride;
            const val = Math.floor(mat.data[index] * exp) / exp;
            const str = (val >= 0 && has_negative_vals ? " " : "") + val.toString();
            const separator = c < cols - 1 ? ", " : "";
            const padding_right = " ".repeat(Math.max(0, max_length - str.length));

            vals.push(`${str}${separator}${padding_right}`);
        }

        const padding_left = r !== 0 ? " ".repeat(space_before) : "";
        // lines.push(`${padding_left}[ ${vals.join(", ")} ]`);
        lines.push(`${padding_left}[ ${vals.join("")}]`); // todo fix missing space before closing brace
    }

    return `[${lines.join("\n ")}]`;
}

export function tensor_info_to_string(a: RawTensor) {
    const max_entries = 16;
    const precision = 3;
    const exp = 10 ** precision;
    const data = [...a.data.slice(0, max_entries)].map(v => Math.floor(v * exp) / exp);

    return (
        "TENSOR INFO\n" +
        `  address: 0x${a.ptr.toString(16)}\n` +
        `  is view: ${a.isview ? "true" : "false"} [src: 0x${a.viewsrc.toString(16)}]\n` +
        `  shape:   [${a.shape.join(", ")}]\n` +
        `  strides: [${a.strides.join(", ")}]\n` +
        `  rank:    ${a.rank}\n` +
        `  nelem:   ${a.nelem}\n` +
        `  ndata:   ${a.ndata}\n` +
        `  size:    ${a.size} bytes\n` +
        `  offset:  ${a.offset}\n` +
        `  data:    [${data.join(", ")}${a.data.length > max_entries ? ", ..." : ""}]\n`);
}

// takes a tensor object and returns a readable string to represent it in the cli
function get_tensor_name(tensor: Tensor, options: gts_options): string {
    const is_param = tensor instanceof Parameter;
    return (options.show_id ? `{${tensor.id}} ` : "")
        + (is_param ? green : "") + `${tensor.constructor.name}` + reset
        + (tensor.name ? ` ("${tensor.name}")` : "")
        + (options.show_shape ? ` ${blue}[${tensor.shape}]${reset}` : "");
}

// "gts" = "graph to string"
export type gts_options = {
    show_id: boolean,
    show_shape: boolean,
};

const default_gts_options: gts_options = {
    show_id: false,
    show_shape: false,
};

export function graph_to_string(
    current_tensor: Tensor,
    options: Partial<gts_options> = {},
    visited: Set<Tensor> = new Set(),
    prefix: string = "",
    is_last: boolean = true,
    is_root: boolean = true,
): string {
    const _options: gts_options =  {...default_gts_options, ...options};
    const identifier = get_tensor_name(current_tensor, _options);

    let result = is_root
        ? ` ${bold}${purple}${identifier}${reset}\n` 
        : ` ${prefix}${bold}${is_last ? "└─ " : "├─ "}${identifier}${reset}\n`;

    visited.add(current_tensor);

    const new_prefix = prefix + `${bold}${is_last ? "   " : "│  "}${reset}`;
    current_tensor.parents.forEach((parent, index) => {
        const parent_is_last = index === current_tensor.parents.length - 1;
        const parent_identifier = get_tensor_name(parent, _options.show_id);
        result += visited.has(parent)
            ? `${new_prefix}${bold}${parent_is_last ? "└─ " : "├─ "}${parent_identifier} (already visited)${reset}\n`
            : graph_to_string(parent, options, visited, new_prefix, parent_is_last, false);
    });

    return result;
}

export function ordinal_str(n: number): string {
    const last_digit = n % 10;
    const last_two_digits = n % 100;

    const suffix = (last_digit == 1 && last_two_digits != 11 ? "st" :
        last_digit == 2 && last_two_digits != 12 ? "nd" :
            last_digit == 3 && last_two_digits != 13 ? "rd" : "th");

    return `${n}${suffix}`;
}

export function unary_error(op: string, a: RawTensor, dest: RawTensor, result_shape: Shape) {
    return `\n  Failed to execute operation: ${underline}${op}${reset}\n` +
           `  Input shape: [${a.shape}]\n` +
           `  Would produce output of shape: [${result_shape}]\n` +
           `  Incompatible destination shape: [${dest.shape}].\n\n`;
}

export function binary_error(op: string, a: RawTensor, b: RawTensor, dest: RawTensor, result_shape: Shape) {
    return `\n  Failed to execute operation: ${underline}${op}${reset}\n` +
           `  Input shapes: [${a.shape}], [${b.shape}]\n` + // todo remove
           `  Would produce output of shape: [${result_shape}]\n` +
           `  Incompatible destination shape: [${dest.shape}].\n\n`;
}
