import core from "../core/build";
import { Shape } from "../Shape";
import { check_row_col_compat, tensor_like, TensorOp } from "../util";
import { tensor } from "../util";
import Tensor from './Tensor';

// binary operations                       scalar,      pairwise,        broadcasting
export const add        = create_binary_op(core._add_scl, core._add_prw, core._add_prw_brc);
export const sub        = create_binary_op(core._sub_scl, core._sub_prw, core._sub_prw_brc);
export const mul        = create_binary_op(core._mul_scl, core._mul_prw, core._mul_prw_brc);
export const div        = create_binary_op(core._div_scl, core._div_prw, core._div_prw_brc);

// unary operations
export const relu       = create_unary_op(core._relu_tns);
export const tanh       = create_unary_op(core._tanh_tns);
export const binstep    = create_unary_op(core._binstep_tns);
export const logistic   = create_unary_op(core._logistic_tns);
export const sigmoid    = create_unary_op(core._sigmoid_tns);
export const negate     = create_unary_op(core._negate_tns);
export const identity   = create_unary_op(core._copy);
export const copy       = create_unary_op(core._copy);
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

function get_shape_matmul(a: Tensor, b: Tensor) {
    if (!(a instanceof Tensor && b instanceof Tensor)) throw new Error('Tensor.matmul() expects a tensor.');
    check_row_col_compat(a, b);

    // flatten tensors to a "list of matrices" and get the size of that list
    const nmat_a = a.shape.flatten(3)[0];
    const nmat_b = b.shape.flatten(3)[0];

    // check hidim matmul compatibility
    if (nmat_a > 1 && nmat_b > 1 && nmat_a != nmat_b)
        throw new Error(`Cannot multiply matrices of shape [${a.shape}] and [${b.shape}]`);

    // get shape of resulting tensor without the last two axes
    const high_level_shape = a.rank > b.rank
        ? a.shape.slice(0, a.rank - 2)
        : b.shape.slice(0, b.rank - 2);

    // create result tensor
    const result_shape = new Shape(...high_level_shape, a.nrows, b.ncols); // todo kinda want to do this with .get_axis_shape
    return [result_shape, nmat_a, nmat_b];
}

function check_in_place_compat(a: Tensor, result: Tensor, in_place: boolean) {
    if (in_place && a.shape.equals(result.shape))
        throw new Error(`Cannot perform in-place operation. Result tensor [${result.shape}] has different shape than tensor a [${a.shape}].`);
}

// standard matmul on tensors 
export const matmul = (a: Tensor, b: Tensor, in_place = false): Tensor => {
    const [result_shape, nmat_a, nmat_b] = get_shape_matmul(a, b);
    const result = tensor(result_shape);
    check_in_place_compat(a, result, in_place);

    // perform computation using core
    core._mul_tns(
        a.get_ptr(), a.nrows, a.ncols, nmat_a,
        b.get_ptr(), b.nrows, b.ncols, nmat_b,
        result.get_ptr()
    );

    if (in_place) return in_place_cpy(a, result);
    return result;
}

function get_shape_dot(a: Tensor, b: Tensor) {
    if (!(a instanceof Tensor && b instanceof Tensor)) throw new Error('Tensor.dot() expects a tensor.');
    check_row_col_compat(a, b);

    const result_shape = new Shape(
        ...a.shape.slice(0, a.rank - 1),
        ...b.shape.slice(0, b.rank - 2), b.shape[b.rank - 1]); // shape of tensor b without the second-to-last axis

    // flatten tensors to a list of vectors/matrices respectively
    const nvec_a = a.shape.flatten(2)[0];
    const nmat_b = b.shape.flatten(3)[0];

    return [result_shape, nvec_a, nmat_b];
}

// numpy-style dot-product
export const dot = (a: Tensor, b: Tensor, in_place = false): Tensor => {
    const [result_shape, nvec_a, nmat_b] = get_shape_dot(a, b);
    const result = tensor(result_shape);
    check_in_place_compat(a, result, in_place);

    core._dot_tns(
        a.get_ptr(), a.ncols, nvec_a,
        b.get_ptr(), b.nrows, b.ncols, nmat_b,
        result.get_ptr()
    );

    if (in_place) return in_place_cpy(a, result);
    return result;
}

// computes a tensor-tensor/tensor-scalar operation and returns a result tensor
function binary_op(
    core_fn_scl: Function, core_fn_prw: Function, core_fn_brc: Function,
    a: Tensor, b: Tensor | number,
    in_place: boolean
) {
    // perform scalar operation
    if (typeof b === 'number') {
        const result = in_place ? a : tensor_like(a);
        core_fn_scl(a.get_ptr(), b, result.get_ptr(), a.data.length);
        return result;
    }

    if (!(b instanceof Tensor))
        throw new Error(`Type mismatch: Binary operations expect tensors or numbers.`);

    // perform fast pairwise operation without broadcasting if possible
    if (a.shape.equals(b.shape)) {
        const result = in_place ? a : tensor_like(a);
        core_fn_prw(a.get_ptr(), b.get_ptr(), result.get_ptr(), a.data.length);
        return result;
    }

    // check if broadcasting is possible
    if (!a.shape.broadcastable(b.shape))
        throw new Error(`Shape mismatch: Cannot broadcast tensor of shape [${a.shape}] with [${b.shape}].`);

    const max_rank = Math.max(a.rank, b.rank);
    const result_shape = a.shape.broadcast(b.shape);

    if (in_place && !a.shape.equals(result_shape))
        throw new Error(`Cannot perform in-place operation. Result tensor [${result_shape}] has different shape than tensor a [${a.shape}].`);

    const result = in_place ? a : tensor(result_shape);
    const shape_a_exp = a.shape.expand_left(max_rank);
    const shape_b_exp = b.shape.expand_left(max_rank);
    const strides_a = shape_a_exp.get_strides();
    const strides_b = shape_b_exp.get_strides();

    for (let i = 0; i < max_rank; i++) {
        if (shape_a_exp[i] === 1) strides_a[i] = 0;
        if (shape_b_exp[i] === 1) strides_b[i] = 0;
    }

    // create array for passing in metadata
    const metadata_size = max_rank * 3;
    const metadata_ptr = core._alloc_starr(metadata_size);
    const metadata = new Uint32Array(core.memory.buffer, metadata_ptr, metadata_size);
    metadata.set([...strides_a, ...strides_b, ...result_shape]);

    // perform operation with broadcast
    core_fn_brc(a.get_ptr(), b.get_ptr(), result.get_ptr(), metadata_ptr, max_rank);
    core._free_starr(metadata_ptr);

    return result;
}

// applies a unary operation to a tensor
function unary_op(core_fn: Function, a: Tensor, in_place: boolean) {
    const result = in_place ? a : tensor_like(a);
    core_fn(a.get_ptr(), result.get_ptr(), a.data.length);
    return result;
}

function create_unary_op(core_fn: Function) {
    return (a: Tensor, in_place = false) => unary_op(core_fn, a, in_place);
}

function create_binary_op(core_fn_scl: Function, core_fn_prw: Function, core_fn_brc: Function) {
    return (a: Tensor, b: Tensor | number, in_place = false) => binary_op(core_fn_scl, core_fn_prw, core_fn_brc, a, b, in_place);
}

// copies data from result tensor back into tensor a
function in_place_cpy(a: Tensor, result: Tensor) {
    core._copy(result.get_ptr(), a.get_ptr(), a.shape.get_nelem());
    result.free();
    return a;
}
