import { RawTensor } from "./RawTensor.ts";
import core from "./core/build";
import Shape from "./Shape";


// types for high level operations
export type UnaryOp = (src: RawTensor, dest?: RawTensor, param?: number) => RawTensor;
export type BinaryOp<OtherType> = (src_a: RawTensor, src_b: OtherType, dest?: RawTensor) => RawTensor;
export type DropoutOp = (src: RawTensor, dest?: RawTensor, p?: number, seed?: number) => RawTensor;

// types of core functions
type CoreUnaryOp   =  (src_ptr: number, dest_ptr: number, param?: number) => void;
type CoreBinaryOp  = (src_a_ptr: number, src_b_ptr_or_imm: number, dest_ptr: number) => void;
type CoreDropoutOp =  (src_ptr: number, dest_ptr: number, p: number, seed: number) => void;

// binary operations (dest = a <OP> b)
export const add    = create_binary_op("add");
export const sub    = create_binary_op("sub");
export const mul    = create_binary_op("mul");
export const div    = create_binary_op("div");
export const pow    = create_binary_op("pow");
export const dot    = create_dot_op("dot");
export const matmul = create_matmul_op("matmul");

export const add_acc    = create_binary_op("add", true);
export const sub_acc    = create_binary_op("sub", true);
export const mul_acc    = create_binary_op("mul", true);
export const div_acc    = create_binary_op("div", true);
export const pow_acc    = create_binary_op("pow", true);
export const dot_acc    = create_dot_op("dot", true);
export const matmul_acc = create_matmul_op("matmul", true);

// misc operations
export const dropout     = create_dropout_op("dropout");
export const dropout_acc = create_dropout_op("dropout", true);

// unary operations
export const relu       = create_unary_op("relu");
export const leaky_relu = create_unary_op("leaky_relu");
export const binstep    = create_unary_op("binstep");
export const logistic   = create_unary_op("logistic");
export const negate     = create_unary_op("negate");
export const sin        = create_unary_op("sin");
export const cos        = create_unary_op("cos");
export const tan        = create_unary_op("tan");
export const asin       = create_unary_op("asin");
export const acos       = create_unary_op("acos");
export const atan       = create_unary_op("atan");
export const sinh       = create_unary_op("sinh");
export const cosh       = create_unary_op("cosh");
export const tanh       = create_unary_op("tanh");
export const exp        = create_unary_op("exp");
export const log        = create_unary_op("log");
export const log10      = create_unary_op("log10");
export const log2       = create_unary_op("log2");
export const invsqrt    = create_unary_op("invsqrt");
export const sqrt       = create_unary_op("sqrt");
export const ceil       = create_unary_op("ceil");
export const floor      = create_unary_op("floor");
export const abs        = create_unary_op("abs");
export const sign       = create_unary_op("sign");
export const reciprocal = create_unary_op("reciprocal");

export const relu_acc       = create_unary_op("relu", true);
export const leaky_relu_acc = create_unary_op("leaky_relu", true);
export const binstep_acc    = create_unary_op("binstep", true);
export const logistic_acc   = create_unary_op("logistic", true);
export const negate_acc     = create_unary_op("negate", true);
export const sin_acc        = create_unary_op("sin", true);
export const cos_acc        = create_unary_op("cos", true);
export const tan_acc        = create_unary_op("tan", true);
export const asin_acc       = create_unary_op("asin", true);
export const acos_acc       = create_unary_op("acos", true);
export const atan_acc       = create_unary_op("atan", true);
export const sinh_acc       = create_unary_op("sinh", true);
export const cosh_acc       = create_unary_op("cosh", true);
export const tanh_acc       = create_unary_op("tanh", true);
export const exp_acc        = create_unary_op("exp", true);
export const log_acc        = create_unary_op("log", true);
export const log10_acc      = create_unary_op("log10", true);
export const log2_acc       = create_unary_op("log2", true);
export const invsqrt_acc    = create_unary_op("invsqrt", true);
export const sqrt_acc       = create_unary_op("sqrt", true);
export const ceil_acc       = create_unary_op("ceil", true);
export const floor_acc      = create_unary_op("floor", true);
export const abs_acc        = create_unary_op("abs", true);
export const sign_acc       = create_unary_op("sign", true);
export const reciprocal_acc = create_unary_op("reciprocal", true);

// derivatives of (some of the) unary operations
export const df_relu       = create_unary_op("df_relu");
export const df_leaky_relu = create_unary_op("df_leaky_relu");
export const df_negate     = create_unary_op("df_negate");
export const df_sin        = create_unary_op("df_sin");
export const df_cos        = create_unary_op("df_cos");
export const df_tan        = create_unary_op("df_tan");
export const df_asin       = create_unary_op("df_asin");
export const df_acos       = create_unary_op("df_acos");
export const df_atan       = create_unary_op("df_atan");
export const df_sinh       = create_unary_op("df_sinh");
export const df_cosh       = create_unary_op("df_cosh");
export const df_tanh       = create_unary_op("df_tanh");
export const df_exp        = create_unary_op("df_exp");
export const df_log        = create_unary_op("df_log");
export const df_log10      = create_unary_op("df_log10");
export const df_log2       = create_unary_op("df_log2");
export const df_invsqrt    = create_unary_op("df_invsqrt");
export const df_sqrt       = create_unary_op("df_sqrt");
export const df_abs        = create_unary_op("df_abs");
export const df_reciprocal = create_unary_op("df_reciprocal");

// reduce operations
// todo: add pairwise functionality (tensor-valued functions)
export const max  = (a: RawTensor) => core._max_red_scl(a.ptr);
export const min  = (a: RawTensor) => core._min_red_scl(a.ptr);
export const sum  = (a: RawTensor) => core._sum_red_scl(a.ptr);
export const mean = (a: RawTensor) => core._mean_red_scl(a.ptr);
export const max_tns = create_select_op(core._max_red_tns);
export const min_tns = create_select_op(core._min_red_tns);
export const sum_tns = create_reduce_op(core._sum_red_tns);
export const mean_tns = create_reduce_op(core._mean_red_tns);

// be aware of tensor data dependencies when deallocating tensors !!
export const free = (a: RawTensor) => core._free_tensor(a.ptr);

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
export const clone = (src: RawTensor, dest?: RawTensor) => {
    const result = dest || RawTensor.like(src);
    core._clone_tensor(src.ptr, result.ptr);
    return result;
};

export function get_shape_matmul(a: RawTensor, b: RawTensor): Shape {
    if (a.cols !== b.rows)
        throw new Error(`Cannot perform matmul on tensors of shape [${a.shape}] and [${b.shape}]`);

    // flatten tensors to a "list of matrices" and get the size of that list
    const nmat_a = a.shape.flatten(3)[0];
    const nmat_b = b.shape.flatten(3)[0];
    
    // check hidim matmul compatibility
    if (nmat_a > 1 && nmat_b > 1 && nmat_a != nmat_b)
        throw new Error(`Cannot multiply matrices of shape [${a.shape}] and [${b.shape}]`);

    // get shape of resulting tensor without the last two axes
    const high_level_shape = a.rank > b.rank
        ? [...a.shape].slice(0, a.rank - 2)   // todo: ...a.shape should probably be replaced by something like slice
        : [...b.shape].slice(0, b.rank - 2);

    return new Shape([...high_level_shape, a.rows, b.cols]);
}

export function get_shape_dot(a: RawTensor, b: RawTensor): Shape {
    if (a.cols !== b.rows)
        throw new Error(`Cannot perform dot on tensors of shape [${a.shape}] and [${b.shape}]`);

    const result_shape = new Shape([
        ...[...a.shape].slice(0, a.rank - 1),
        ...[...b.shape].slice(0, b.rank - 2), b.shape[b.rank - 1]]); // shape of tensor b without the second-to-last axis

    return result_shape;
}

function create_matmul_op(opcode: string, accumulative = false):  BinaryOp<RawTensor> {
    const postfix = accumulative ? "_acc" : "";
    const core_fn: CoreBinaryOp = core[`_${opcode}${postfix}`];

    return (src_a: RawTensor, src_b: RawTensor, dest?: RawTensor): RawTensor => {
        const result_shape = get_shape_matmul(src_a, src_b);
        const result = dest || RawTensor.create(result_shape);
    
        if (dest && !dest.shape.equals(result_shape))
            throw new Error(`Cannot perform matmul. Result tensor [${result_shape}] has different shape than destination tensor [${dest.shape}].`);
    
        // todo: decide if data should be passed into op from Tensor.forward()
        //   or if we can just pass in the data like here
    
        // perform computation using core
        core_fn(
            src_a.ptr,
            src_b.ptr,
            result.ptr
        );
    
        return result;
    };
}

function create_dot_op(opcode: string, accumulative = false):  BinaryOp<RawTensor> {
    const postfix = accumulative ? "_acc" : "";
    const core_fn: CoreBinaryOp = core[`_${opcode}${postfix}`];

    return (a: RawTensor, b: RawTensor, dest?: RawTensor): RawTensor => {
        const result_shape = get_shape_dot(a, b);
        const result = dest || RawTensor.create(result_shape);
    
        if (dest && !dest.shape.equals(result_shape))
            throw new Error(`Cannot compute dot product. Result tensor [${result_shape}] has different shape than destination tensor [${dest.shape}].`);
    
        core_fn(
            a.ptr,
            b.ptr,
            result.ptr
        );
    
        return result;
    };
}

function create_dropout_op(opcode: string, accumulative = false): DropoutOp {
    const postfix = accumulative ? "_acc" : "";
    const core_fn: CoreDropoutOp = core[`_${opcode}${postfix}`];

    return (src: RawTensor, dest?: RawTensor, p = .5, seed = 0): RawTensor => {
        const result = dest || RawTensor.create(src.shape);
    
        if (dest && !dest.shape.equals(src.shape))
            throw new Error(`Cannot compute dropout. Destination tensor [${dest.shape}] has different shape than source tensor [${src.shape}].`);
    
        core_fn(src.ptr, result.ptr, p, seed);
        return result;
    };
}

function create_binary_op(opcode: string, accumulative = false): BinaryOp<RawTensor | number> {
    const postfix = accumulative ? "_acc" : "";
    const core_fn_brc: CoreBinaryOp = core[`_${opcode}_brc${postfix}`];   //   broadcasting operation
    const core_fn_dbrc: CoreBinaryOp = core[`_${opcode}_dbrc${postfix}`]; // debroadcasting operations

    return (src_a: RawTensor, _src_b: RawTensor | number, _dest?: RawTensor): RawTensor => {
        const scalar_op = typeof _src_b === "number";
        const src_b = scalar_op ? RawTensor.scalar(_src_b) : _src_b;
        const brc_result_shape = src_a.shape.broadcast(src_b.shape);
        const dest = _dest || RawTensor.create(brc_result_shape);

        // todo come up with a better error message
        // todo compatibility with brc/dbrc would be nice
        // check if in-place op is possible
        if ((src_a.ptr === dest.ptr && !src_a.shape.equals(brc_result_shape)) || (src_b.ptr === dest.ptr && !src_b.shape.equals(dest.shape)))
            throw new Error("Could not perform in-place operation in this case.");

        // case: broadcasting / pairwise
        if (dest.shape.nelem >= brc_result_shape.nelem) {
            if (!brc_result_shape.broadcastable(dest.shape))
                throw new Error(`Cant perform broadcasting because result shape [${brc_result_shape}] is incompatible with shape of destination [${dest.shape}].`);

            core_fn_brc(src_a.ptr, src_b.ptr, dest.ptr);
        }

        // case: debroadcasting
        else if (dest.shape.nelem < brc_result_shape.nelem) {
            if (!brc_result_shape.broadcastable(dest.shape))
                throw new Error(`Cant perform debroadcasting because result shape [${brc_result_shape}] is incompatible with shape of destination [${dest.shape}].`);

            core_fn_dbrc(src_a.ptr, src_b.ptr, dest.ptr);
        }

        // deallocate temporary scalar tensor
        if (scalar_op) src_b.free();

        return dest;
    };
}

function create_unary_op(opcode: string, accumulative = false): UnaryOp {
    const postfix = accumulative ? "_acc" : "";
    const core_fn_prw: CoreUnaryOp = core[`_${opcode}_prw${postfix}`];   // pairwise
    const core_fn_brc: CoreUnaryOp = core[`_${opcode}_brc${postfix}`];   // broadcasting
    const core_fn_dbrc: CoreUnaryOp = core[`_${opcode}_dbrc${postfix}`]; // debroadcasting

    return (src: RawTensor, _dest?: RawTensor, param?: number) => {
        if (_dest && !src.shape.broadcastable(_dest.shape))
            throw new Error(`Cannot perform unary operation because broadcasting is not possible between source tensor [${src.shape}] and destination tensor [${_dest.shape}].`);

        const dest = _dest || RawTensor.like(src);

        // pairwise
        if (src.nelem === dest.nelem) {
            core_fn_prw(src.ptr, dest.ptr, param);
            return dest;
        }

        // todo add support for (de)broadcasting in-place ops
        if (src.ptr === dest.ptr)
            throw new Error("Could not perform in-place operation in this case.");

        if (src.nelem < dest.nelem) core_fn_brc(src.ptr, dest.ptr);       // broadcasting
        else if (src.nelem > dest.nelem) core_fn_dbrc(src.ptr, dest.ptr); // debroadcasting
    
        return dest;
    };
}

// these operations select one element from a source tensor
// the result is a scalar view of the source
// example: min finds the smallest element of the source
function create_select_op(core_fn: CoreUnaryOp) {
    return (src: RawTensor, dest?: RawTensor) => {
        if (dest && !dest.shape.is_scalar)
            throw new Error(`Cannot perform reduce operation. Provided destination tensor is not scalar. Destination shape: [${dest.shape}]`);

        if (dest && dest.data_ptr !== src.data_ptr)
            throw new Error("Cannot perform reduce operation. Destination is a view that does not reference the source pointer.");

        const result: RawTensor = dest || RawTensor.view_of(src, src.rank);
        core_fn(src.ptr, result.ptr);
        return result;
    };
}

function create_reduce_op(core_fn: CoreUnaryOp) {
    return (src: RawTensor, dest?: RawTensor) => {
        if (dest && !dest.shape.is_scalar)
            throw new Error(`Cannot perform reduce operation. Provided destination tensor is not scalar. Destination shape: [${dest.shape}]`);

        const result: RawTensor = dest || RawTensor.scalar();
        core_fn(src.ptr, result.ptr);
        return result;
    };
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

export function transpose(a: RawTensor, permutation?: number[]): RawTensor {
    let _permutation: number[];

    // todo: handle rank=1: shape should be 1-extended to the right

    let new_shape, new_strides;

    // // vectors need to be 1-extended to the right
    if (a.rank === 1) {
        if (!permutation || permutation.length !== 0)
            throw new Error("Can't transpose tensors of rank 1 with a specific permutation because there is only one axis.");

        new_shape = [...a.shape, 1];
        new_strides = [1, ...a.shape];
    } else {
        if (!permutation || permutation.length === 0) {
            _permutation = get_matrix_transpose_permutation(a.rank);
        } else {
            _permutation = [...permutation];
            validate_permutation(permutation, a.rank);
        }

        new_shape   = _permutation.map(i => a.shape[i]);
        new_strides = _permutation.map(i => a.strides[i]);
    }

    return a.reshape(new_shape, new_strides);
}

// size_t get_index(struct tensor_t* a, size_t linear_index) {
//     size_t ia = a->offset;
//     size_t remainder = linear_index;
//     size_t iaxis;

//     for (size_t dim = a->rank; dim-- > 0;) {
//         iaxis = remainder % a->shape[dim];
//         ia += iaxis * a->strides[dim];
//         remainder /= a->shape[dim];
//     }

//     return ia;
// }

// function for making two scalar views of independent but equally shaped tensors
// point to the same element of said tensors.
// example:
//   let A, B be two identically shaped tensors
//   let a, b be two scalar views of these tensors (shape = [1])
//   where a points to some element A_ijk... of A and b points to some element of B
//   then sync_scl_views(a, b) will make b point to the element B_ijk... in tensor B.
// export function sync_scl_views(a: RawTensor, b: RawTensor, linear_index: number) {
//     if (a.shape !== b.shape)
//         throw new Error("Can't synchronize differently shaped views.");

//     if (!a.shape.is_scalar)
//         throw new Error("Synchronization of non-scalar views is not supported.");

//     let ib = b.offset;
//     let remainder = linear_index;
//     let iaxis;

//     for (let dim = b.rank; dim-- > 0;) {
//         iaxis = remainder % b.shape[dim];
//         ib += iaxis * b.strides[dim];
//         remainder /= b.shape[dim];
//     }

//     b.set_offset(ib);
// }
