import core from "./core/build";
import { check_row_col_compat } from "./util";
import tensor, {Tensor, create_view, tensor_like} from "./Tensor";
import Shape from "./Shape";

// types for high level operations
export type UnaryOp = (src: Tensor, dest?: Tensor) => Tensor;
export type BinaryOp<OtherType> = (src_a: Tensor, src_b: OtherType, dest?: Tensor) => Tensor;

// types of core functions
type CoreUnaryOp =  (src_ptr: number, dest_ptr: number) => void;
type CoreBinaryOp = (src_a_ptr: number, src_b_ptr_or_imm: number, dest_ptr: number) => void;

// binary operations                       broadcasting
export const add        = create_binary_op(core._add_prw_brc);
export const sub        = create_binary_op(core._sub_prw_brc);
export const mul        = create_binary_op(core._mul_prw_brc);
export const div        = create_binary_op(core._div_prw_brc);
export const pow        = create_binary_op(core._pow_prw_brc);

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
export const free = (a: Tensor) => core._free_tensor(a.get_view_ptr());

/**
 * Creates a deep copy of a tensor.
 * This means all data and metadata is copied to a destination tensor without
 * referencing the original.
 * If the original tensor is a view of another tensor,
 * we will only copy the elements in the tensor that actually
 * occur in the view. This prevents allocating more memory than needed
 * @param src Original tensor to copy
 * @param dest Destination tensor where data will be copied to
 * @returns A copy of the original tensor.
 */
export const clone = (src: Tensor, dest?: Tensor) => {
    const result = dest || tensor_like(src);
    core._clone_tensor(src.get_view_ptr(), result.get_view_ptr());
    return result;
};

function get_shape_matmul(a: Tensor, b: Tensor): Shape {
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

// standard matmul on tensors 
export const matmul: BinaryOp<Tensor> = (src_a: Tensor, src_b: Tensor, dest?: Tensor): Tensor => {
    const result_shape = get_shape_matmul(src_a, src_b);
    const result = dest || tensor(result_shape);

    if (dest && !dest.shape.equals(result_shape))
        throw new Error(`Cannot perform matmul. Result tensor [${result_shape}] has different shape than destination tensor [${dest.shape}].`);

    // todo: decide if data should be passed into op from CompGraphNode.forward()
    //   or if we can just pass in the data like here

    // perform computation using core
    core._mul_tns(
        src_a.get_view_ptr(),
        src_b.get_view_ptr(),
        result.get_view_ptr()
    );

    return result;
};

function get_shape_dot(a: Tensor, b: Tensor): Shape {
    check_row_col_compat(a, b);

    const result_shape = new Shape([
        ...[...a.shape].slice(0, a.get_rank() - 1),
        ...[...b.shape].slice(0, b.get_rank() - 2), b.shape[b.get_rank() - 1]]); // shape of tensor b without the second-to-last axis

    return result_shape;
}

// numpy-style dot-product
export const dot = (a: Tensor, b: Tensor, dest?: Tensor): Tensor => {
    const result_shape = get_shape_dot(a, b);
    const result = dest || tensor(result_shape);

    if (dest && !dest.shape.equals(result_shape))
        throw new Error(`Cannot compute dot product. Result tensor [${result_shape}] has different shape than destination tensor [${dest.shape}].`);

    core._dot_tns(
        a.get_view_ptr(),
        b.get_view_ptr(),
        result.get_view_ptr());

    return result;
};

function create_unary_op(core_fn: CoreUnaryOp): UnaryOp {
    return (src: Tensor, dest?: Tensor) => {
        if (dest && !dest.shape.equals(src.shape))
            throw new Error(`Cannot perform unary op. Result tensor [${src.shape}] has different shape than destination tensor [${dest.shape}].`);

        const result: Tensor = dest || tensor_like(src);
        core_fn(src.get_view_ptr(), result.get_view_ptr());
        return result;
    };
}

// computes a tensor-tensor/tensor-scalar operation and returns a result tensor
function binary_op(core_fn_brc: CoreBinaryOp, src_a: Tensor, src_b: Tensor, dest?: Tensor): Tensor {
    // check if broadcasting is possible
    if (!src_a.shape.broadcastable(src_b.shape))
        throw new Error(`Shape mismatch: Cannot broadcast tensor of shape [${src_a.shape}] with [${src_b.shape}].`);

    const result_shape = src_a.shape.broadcast(src_b.shape);

    if (dest) {
        if (!dest.shape.equals(result_shape))
            throw new Error(`Cannot perform broadcasting binary operation. Result tensor [${result_shape}] has different shape than tensor a [${src_a.shape}].`);

        // TODO: Check if this can somehow be done safely
        //       If not it might be worth it to create a tensor that holds
        //       the interim before the result is written back to the source
        if ((dest == src_a && !dest.shape.equals(src_a.shape)) || (dest == src_b && !dest.shape.equals(src_b.shape)))
            throw new Error("In-place broadcasting is not supported.");
    }

    const result = dest || tensor(result_shape);
    core_fn_brc(src_a.get_view_ptr(), src_b.get_view_ptr(), result.get_view_ptr());

    return result;
}

function create_binary_op(core_fn_brc: CoreBinaryOp): BinaryOp<Tensor> {
    return (a: Tensor, b: Tensor, dest?: Tensor) => binary_op(core_fn_brc, a, b, dest);
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

    const new_tensor = create_view(a);
    new_tensor.shape.set(new_shape);
    new_tensor.strides.set(new_strides);

    return new_tensor;
}

// todo: add pairwise functionality (tensor-valued functions)
export const max  = (a: Tensor) => core._max_red(a.get_view_ptr());
export const min  = (a: Tensor) => core._min_red(a.get_view_ptr());
export const sum  = (a: Tensor) => core._sum_red(a.get_view_ptr());
export const mean = (a: Tensor) => core._mean_red(a.get_view_ptr());
