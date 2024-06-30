import { check_row_col_compat } from "./util";
import { RawTensor } from "./RawTensor.ts";
import core from "./core/build";
import Shape from "./Shape";


// types for high level operations
export type UnaryOp = (src: RawTensor, dest?: RawTensor) => RawTensor;
export type BinaryOp<OtherType> = (src_a: RawTensor, src_b: OtherType, dest?: RawTensor) => RawTensor;

// types of core functions
type CoreUnaryOp =  (src_ptr: number, dest_ptr: number) => void;
type CoreBinaryOp = (src_a_ptr: number, src_b_ptr_or_imm: number, dest_ptr: number) => void;

// binary operations (dest = a <OP> b)
export const add = create_binary_op("add");
export const sub = create_binary_op("sub");
export const mul = create_binary_op("mul");
export const div = create_binary_op("div");
export const pow = create_binary_op("pow");

export const add_acc = create_binary_op("add", true);
export const sub_acc = create_binary_op("sub", true);
export const mul_acc = create_binary_op("mul", true);
export const div_acc = create_binary_op("div", true);
export const pow_acc = create_binary_op("pow", true);

// unary operations
export const relu       = create_unary_op("relu");
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
export const reciprocal = create_unary_op("reciprocal");

export const relu_acc       = create_unary_op("relu", true);
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
export const reciprocal_acc = create_unary_op("reciprocal", true);

// reduce operations
// todo: add pairwise functionality (tensor-valued functions)
export const max  = (a: RawTensor) => core._max_red_scl(a.ptr);
export const min  = (a: RawTensor) => core._min_red_scl(a.ptr);
export const sum  = (a: RawTensor) => core._sum_red_scl(a.ptr);
export const mean = (a: RawTensor) => core._mean_red_scl(a.ptr);
export const max_tns = create_reduce_op(core._max_red_tns);
export const min_tns = create_reduce_op(core._min_red_tns);
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

function get_shape_matmul(a: RawTensor, b: RawTensor): Shape {
    check_row_col_compat(a, b);

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

// standard matmul on tensors 
export const matmul: BinaryOp<RawTensor> = (src_a: RawTensor, src_b: RawTensor, dest?: RawTensor): RawTensor => {
    const result_shape = get_shape_matmul(src_a, src_b);
    const result = dest || RawTensor.create(result_shape);

    if (dest && !dest.shape.equals(result_shape))
        throw new Error(`Cannot perform matmul. Result tensor [${result_shape}] has different shape than destination tensor [${dest.shape}].`);

    // todo: decide if data should be passed into op from CompGraphNode.forward()
    //   or if we can just pass in the data like here

    // perform computation using core
    core._mul_tns(
        src_a.ptr,
        src_b.ptr,
        result.ptr
    );

    return result;
};

function get_shape_dot(a: RawTensor, b: RawTensor): Shape {
    check_row_col_compat(a, b);

    const result_shape = new Shape([
        ...[...a.shape].slice(0, a.rank - 1),
        ...[...b.shape].slice(0, b.rank - 2), b.shape[b.rank - 1]]); // shape of tensor b without the second-to-last axis

    return result_shape;
}

// numpy-style dot-product
export const dot = (a: RawTensor, b: RawTensor, dest?: RawTensor): RawTensor => {
    const result_shape = get_shape_dot(a, b);
    const result = dest || RawTensor.create(result_shape);

    if (dest && !dest.shape.equals(result_shape))
        throw new Error(`Cannot compute dot product. Result tensor [${result_shape}] has different shape than destination tensor [${dest.shape}].`);

    core._dot_tns(
        a.ptr,
        b.ptr,
        result.ptr);

    return result;
};

export function grad_acc(src: RawTensor, dest: RawTensor) {
    core._sum_red_brc(src.ptr, dest.ptr);
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

    return (src: RawTensor, _dest?: RawTensor) => {
        if (_dest && !src.shape.broadcastable(_dest.shape))
            throw new Error(`Cannot perform unary operation because broadcasting is not possible between source tensor [${src.shape}] and destination tensor [${_dest.shape}].`);

        const dest = _dest || RawTensor.like(src);

        // pairwise
        if (src.nelem === dest.nelem) {
            core_fn_prw(src.ptr, dest.ptr);
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

    if (!permutation || permutation.length === 0) {
        _permutation = get_matrix_transpose_permutation(a.rank);
    }
    else {
        _permutation = [...permutation];
        validate_permutation(permutation, a.rank);
    }

    const new_shape   = _permutation.map(i => a.shape[i]);
    const new_strides = _permutation.map(i => a.strides[i]);

    const new_tensor = RawTensor.view_from(a);
    new_tensor.shape.set(new_shape);
    new_tensor.strides.set(new_strides);

    return new_tensor;
}
