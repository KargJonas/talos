import core from "./core/build";
import { check_row_col_compat } from "./util";
import tensor, { Tensor, derive_tensor } from "./Tensor";
import Shape from "./Shape";

// types for high level operations
export type UnaryOp = (a: Tensor, in_place: boolean) => Tensor;
export type BinaryOp<OtherType> = (a: Tensor, b: OtherType, in_place?: boolean) => Tensor;

// types of core functions
type CoreUnaryOp =  (a_ptr: number, res_ptr: number) => void;
type CoreBinaryOp = (a_ptr: number, b_ptr_or_val: number, res_ptr: number) => void;

// binary operations                       scalar,        pairwise,      broadcasting
export const add        = create_binary_op(core._add_scl, core._add_prw, core._add_prw_brc);
export const sub        = create_binary_op(core._sub_scl, core._sub_prw, core._sub_prw_brc);
export const mul        = create_binary_op(core._mul_scl, core._mul_prw, core._mul_prw_brc);
export const div        = create_binary_op(core._div_scl, core._div_prw, core._div_prw_brc);

// unary operations
export const relu       = create_unary_op(core._relu_tns);
export const tanh       = create_unary_op(core._tanh_tns);
export const binstep    = create_unary_op(core._binstep_tns);
export const logistic   = create_unary_op(core._logistic_tns);
export const negate     = create_unary_op(core._negate_tns);
export const sin        = create_unary_op(core._sin_tns);
export const cos        = create_unary_op(core._cos_tns);
export const tan        = create_unary_op(core._tan_tns);
export const asin       = create_unary_op(core._asin_tns);
export const acos       = create_unary_op(core._acos_tns);
export const atan       = create_unary_op(core._atan_tns);
export const sinh       = create_unary_op(core._sinh_tns);
export const cosh       = create_unary_op(core._cosh_tns);
export const exp        = create_unary_op(core._exp_tns);
export const log        = create_unary_op(core._log_tns);
export const log10      = create_unary_op(core._log10_tns);
export const log2       = create_unary_op(core._log2_tns);
export const invsqrt    = create_unary_op(core._invsqrt_tns);
export const sqrt       = create_unary_op(core._sqrt_tns);
export const ceil       = create_unary_op(core._ceil_tns);
export const floor      = create_unary_op(core._floor_tns);
export const abs        = create_unary_op(core._abs_tns);
export const reciprocal = create_unary_op(core._reciprocal_tns);

// be aware of tensor data dependencies when deallocating tensors !!
export const free  = (a: Tensor) => core._free_tensor(a.get_view_ptr());

/**
 * Creates a deep copy of a tensor.
 * This means all data and metadata is copied to a new tensor without 
 * referencing the original.
 * If the original tensor is a view of another tensor,
 * we will only copy the elements in the tensor that actually
 * occur in the view. This prevents allocating more memory than needed
 * @param a Original tensor to copy
 * @returns A copy of the original tensor.
 */
export const clone = (a: Tensor) => {
    const new_tensor = tensor(a.shape);
    core._copy_tensor(a.get_view_ptr(), new_tensor.get_view_ptr());
    return new_tensor;
};

function get_shape_matmul(a: Tensor, b: Tensor): Shape {
    if (!(a instanceof Tensor && b instanceof Tensor)) throw new Error("Tensor.matmul() expects a tensor.");
    check_row_col_compat(a, b);

    // flatten tensors to a "list of matrices" and get the size of that list
    const nmat_a = a.shape.flatten(3)[0];
    const nmat_b = b.shape.flatten(3)[0];
    
    // check hidim matmul compatibility
    if (nmat_a > 1 && nmat_b > 1 && nmat_a != nmat_b)
        throw new Error(`Cannot multiply matrices of shape [${a.shape}] and [${b.shape}]`);

    // get shape of resulting tensor without the last two axes
    const high_level_shape = a.get_rank() > b.get_rank()
        ? [...a.shape].slice(0, a.get_rank() - 2)   // todo: ...a.shape should probably be replaced by something like slice
        : [...b.shape].slice(0, b.get_rank() - 2);

    return new Shape([...high_level_shape, a.get_rows(), b.get_cols()]);
}

function check_in_place_compat(a: Tensor, result: Tensor, in_place: boolean) {
    if (in_place && !a.shape.equals(result.shape))
        throw new Error(`Cannot perform in-place operation. Result tensor [${result.shape}] has different shape than tensor a [${a.shape}].`);
}

// standard matmul on tensors 
export const matmul: BinaryOp<Tensor> = (a: Tensor, b: Tensor, in_place = false): Tensor => {
    const result_shape = get_shape_matmul(a, b);
    const result = tensor(result_shape);
    check_in_place_compat(a, result, in_place);

    // todo: decide if data should be passed into op from CompGraphNode.forward()
    //   or if we can just pass in the data like here

    // perform computation using core
    core._mul_tns(
        a.get_view_ptr(),
        b.get_view_ptr(),
        result.get_view_ptr()
    );

    if (in_place) return in_place_cpy(result, a);
    return result;
};

function get_shape_dot(a: Tensor, b: Tensor): Shape {
    if (!(a instanceof Tensor && b instanceof Tensor)) throw new Error("Tensor.dot() expects a tensor.");
    check_row_col_compat(a, b);

    const result_shape = new Shape([
        ...[...a.shape].slice(0, a.get_rank() - 1),
        ...[...b.shape].slice(0, b.get_rank() - 2), b.shape[b.get_rank() - 1]]); // shape of tensor b without the second-to-last axis

    return result_shape;
}

// numpy-style dot-product
export const dot = (a: Tensor, b: Tensor, in_place = false): Tensor => {
    const result_shape = get_shape_dot(a, b);
    const result = tensor(result_shape);
    check_in_place_compat(a, result, in_place);

    core._dot_tns(
        a.get_view_ptr(),
        b.get_view_ptr(),
        result.get_view_ptr());

    if (in_place) return in_place_cpy(result, a);
    return result;
};

function create_unary_op(core_fn: CoreUnaryOp): UnaryOp {
    return (a: Tensor, in_place = false) => {
        const result: Tensor = in_place ? a : clone(a);
        core_fn(a.get_view_ptr(), result.get_view_ptr());
        return result;
    };
}

// computes a tensor-tensor/tensor-scalar operation and returns a result tensor
function binary_op(
    core_fn_scl: CoreBinaryOp, core_fn_prw: CoreBinaryOp, core_fn_brc: CoreBinaryOp,
    a: Tensor, b: Tensor | number,
    in_place: boolean
): Tensor {
    // perform scalar operation
    if (typeof b === "number") {
        const result = in_place ? a : clone(a);
        core_fn_scl(a.get_view_ptr(), b, result.get_view_ptr());
        return result;
    }

    if (!(b instanceof Tensor))
        throw new Error("Type mismatch: Binary operations expect tensors or numbers.");

    // perform fast pairwise operation without broadcasting if possible
    if (a.shape.equals(b.shape)) {
        const result = in_place ? a : clone(a);
        core_fn_prw(a.get_view_ptr(), b.get_view_ptr(), result.get_view_ptr());
        return result;
    }

    // check if broadcasting is possible
    if (!a.shape.broadcastable(b.shape))
        throw new Error(`Shape mismatch: Cannot broadcast tensor of shape [${a.shape}] with [${b.shape}].`);

    const result_shape = a.shape.broadcast(b.shape);

    if (in_place && !a.shape.equals(result_shape))
        throw new Error(`Cannot perform in-place operation. Result tensor [${result_shape}] has different shape than tensor a [${a.shape}].`);

    const result = in_place ? a : tensor(result_shape);
    core_fn_brc(a.get_view_ptr(), b.get_view_ptr(), result.get_view_ptr());

    return result;
}

function create_binary_op(core_fn_scl: CoreBinaryOp, core_fn_prw: CoreBinaryOp, core_fn_brc: CoreBinaryOp): BinaryOp<Tensor | number> {
    return (a: Tensor, b: Tensor | number, in_place = false) => binary_op(core_fn_scl, core_fn_prw, core_fn_brc, a, b, in_place);
}

// copies data from one tensor to another
function in_place_cpy(source: Tensor, dest: Tensor) {
    core._copy_farr(source.get_data_ptr(), dest.get_data_ptr(), source.get_nelem());
    source.free();
    return dest;
}

function validate_permutation(permutation: number[], rank: number): void {
    if (permutation.length !== rank)
        throw new Error(`The provided permutation [${permutation}] does not match the rank of the tensor (rank = ${rank}).`);

    const _permutation = [...permutation].sort();

    for (let i = 0; i < permutation.length; i++) {
        if (_permutation[i] !== i)
            throw new Error(`The provided permutation [${permutation}] is not valid.`);
    }
}

/**
 * This function generates a permutation that swaps the
 * last (rightmost) two axes of a rank-n tensor.
 * @param rank Rank of tensor
 * @returns a permutation that swaps the last two axes.
 */
function get_matrix_transpose_permutation(rank: number): number[] {
    // // swap last two axes
    // const permutation: number[] = [];
    // for (let i = 0; i < rank - 2; i++) permutation.push(i);
    // permutation.push(rank - 1);
    // permutation.push(rank - 2);

    // NumPy-style default permutation (inverse)
    const permutation: number[] = [];
    for (let i = 0; i < rank; i++) permutation.push(i);
    permutation.reverse();
    return permutation;
}

export function transpose(a: Tensor, permutation?: number[]): Tensor {
    let _permutation: number[];

    // todo: handle rank=1: shape should be 1-extended to the right

    if (!permutation || permutation.length === 0) {
        _permutation = get_matrix_transpose_permutation(a.get_rank());
    }
    else {
        _permutation = [...permutation];
        validate_permutation(permutation, a.get_rank());
    }

    const new_shape   = _permutation.map(i => a.shape[i]);
    const new_strides = _permutation.map(i => a.strides[i]);

    return derive_tensor(a, new_shape, new_strides, a.get_offset());
}

// todo: add pairwise functionality (tensor-valued functions)
export const max = (a: Tensor) => core._max_red(a.get_view_ptr());
export const min = (a: Tensor) => core._min_red(a.get_view_ptr());
export const sum = (a: Tensor) => core._sum_red(a.get_view_ptr());
