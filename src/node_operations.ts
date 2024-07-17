import * as ops from "./base/raw_tensor_operations.ts";
import { get_shape_dot, get_shape_matmul } from "./base/raw_tensor_operations.ts";
import {RawTensor} from "./base/raw_tensor.ts";
import Shape from "./base/shape.ts";
import { get_global_seed } from "./base/util.ts";
import Tensor from "./tensor.ts";

// This file contains all operations of the graph-node abstraction-level
// These are essentially all operations of the tensor level plus their derivatives

// FwOps only have primals (no grad, no backprop)
export class FwOp extends Tensor {
    value: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.like(parents[0].value);
    }
}

// FwBwOps have primals and gradients
export class FwBwOp extends FwOp {
    grad: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.grad = RawTensor.like(this.value);
    }
}

// FwBwInterimOps have primals, grads and an interim to hold intermediate results
export class FwBwInterimOp extends FwBwOp {
    interim: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.interim = RawTensor.like(this.value);
    }
}

export class Parameter extends Tensor {
    value: RawTensor;

    constructor(value: RawTensor | number, requires_grad: boolean) {
        super([]);
        this.value = typeof value === "number" ? RawTensor.scalar(value) : value;
        if (requires_grad) this.grad = RawTensor.like(this.value);
    }
}

export class Source extends Tensor {
    value: RawTensor;
    producer: () => RawTensor;

    constructor(shape: Shape | number[], producer: () => RawTensor) {
        super([]);

        this.value = RawTensor.create(shape);
        this.producer = producer;
    }

    fw = () => this.value = this.producer();
}

export class Add extends Tensor {
    value: RawTensor;
    grad: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.create(this.parents[0].value.shape.broadcast(this.parents[1].value.shape));
        this.grad = RawTensor.like(this.value);
    }

    fw = () => ops.add(this.parents[0].value, this.parents[1].value, this.value);

    bw() {
        // d/da (a+b) = 1
        if (this.parents[0].grad) ops.add(this.grad, this.parents[0].grad, this.parents[0].grad); // parents[0].grad = 1 * this.grad

        // d/db (a+b) = 1
        if (this.parents[1].grad) ops.add(this.grad, this.parents[1].grad, this.parents[1].grad); // parents[0].grad = 1 * this.grad
    }
}

export class Sub extends Tensor {
    value: RawTensor;
    grad: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.create(this.parents[0].value.shape.broadcast(this.parents[1].value.shape));
        this.grad = RawTensor.like(this.value);
    }

    fw = () => ops.sub(this.parents[0].value, this.parents[1].value, this.value);

    bw() {
        // d/da (a-b) = 1
        if (this.parents[0].grad) ops.add(this.parents[0].grad, this.grad, this.parents[0].grad); // parents[0].grad = 1 * this.grad

        // d/db (a-b) = -1
        if (this.parents[1].grad) ops.sub(this.parents[1].grad, this.grad, this.parents[1].grad); // parents[1].grad parents[0].grad = -1 * this.grad
    }
}

export class Mul extends Tensor {
    value: RawTensor;
    grad: RawTensor;
    interim: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.create(this.parents[0].value.shape.broadcast(this.parents[1].value.shape));
        this.grad = RawTensor.like(this.value);
        this.interim = RawTensor.like(this.value);
    }

    fw = () => ops.mul(this.parents[0].value, this.parents[1].value, this.value);

    bw() {
        const a = this.parents[0];
        const b = this.parents[1];

        // d/da (a*b) = b
        if (a.grad) ops.mul_acc(this.grad, b.value, a.grad);

        // d/db (a*b) = a
        if (b.grad) ops.mul_acc(this.grad, a.value, b.grad);
    }
}

export class Div extends Tensor {
    value: RawTensor;
    grad: RawTensor;
    interim: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.create(this.parents[0].value.shape.broadcast(this.parents[1].value.shape));
        this.grad = RawTensor.like(this.value);
        this.interim = RawTensor.like(this.value);
    }

    fw = () => ops.div(this.parents[0].value, this.parents[1].value, this.value);

    bw() {
        const a = this.parents[0];
        const b = this.parents[1];

        // d/da (a/b) = 1/b
        if (a.grad) {
            ops.div(this.grad, b.value, this.interim); // interim = this.grad * 1/b = this.grad / b
            ops.add(a.grad, this.interim, a.grad); // a.grad += interim
        }

        // d/db (a/b) = -a/(b^2)
        if (b.grad) {       
            ops.pow(b.value, 2, this.interim);              // interim = b^2
            ops.div(a.value, this.interim, this.interim);   // interim = a / b^2
            ops.negate(this.interim, this.interim);         // interim = -a / b^2
            ops.mul_acc(this.grad, this.interim, b.grad);   // b.grad += this.grad * (-a / b^2)
        }
    }
}

export class Pow extends Tensor {
    value: RawTensor;
    grad: RawTensor;
    interim_0: RawTensor;
    interim_1: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.create(this.parents[0].value.shape.broadcast(this.parents[1].value.shape));
        this.grad = RawTensor.like(this.value);
        this.interim_0 = RawTensor.like(this.parents[0].value);
        this.interim_1 = RawTensor.like(this.parents[1].value);
    }

    fw = () => ops.pow(this.parents[0].value, this.parents[1].value, this.value);

    bw() {
        const a = this.parents[0].value; // base
        const b = this.parents[1].value; // exponent

        if (this.parents[0].grad) {
            // d/da (a^b) = b * a^(b-1)
            ops.sub(b, 1, this.interim_1); // interim = b - 1
            ops.pow(a, this.interim_1, this.interim_0); // interim = a^(b-1)
            ops.mul(b, this.interim_0, this.interim_0); // interim = b * a^(b-1)
            ops.mul(this.grad, this.interim_0, this.interim_0); // interim = grad * b * a^(b-1)
            ops.add_acc(this.interim_0, this.parents[0].grad, this.parents[0].grad); // parents[0].grad += interim
        }

        if (this.parents[1].grad) {
            // todo validate
            // d/db (a^b) = a^b * ln(a)
            ops.log(a, this.interim_0); // interim = ln(a)
            ops.mul(this.value, this.interim_0, this.interim_0); // interim = a^b * ln(a)
            ops.mul(this.grad, this.interim_0, this.interim_0); // interim = grad * a^b * ln(a)
            ops.add_acc(this.interim_0, this.parents[1].grad, this.parents[1].grad); // parents[1].grad += interim
        }
    }
}

export class Matmul extends Tensor {
    value: RawTensor;
    grad: RawTensor;

    // these are views and therefore don't need a lot of memory
    A: RawTensor;
    B: RawTensor;
    A_T: RawTensor;
    B_T: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        
        const A = this.parents[0].value;
        const B = this.parents[1].value;

        // extend vectors such that they can be multiplied
        this.A = A.rank === 1 ? A.left_extend() : A;
        this.B = B.rank === 1 ? B.right_extend() : B;

        this.A_T = this.A.T;
        this.B_T = this.B.T;

        this.value = RawTensor.create(get_shape_matmul(this.A, this.B));
        this.grad = RawTensor.like(this.value);

        // todo: YOU LEFT OFF HERE
        // this is nice and all but i think the real solution would be to
        // just support vec-mat, mat-vec and vec-vec ops in matmul and dot.
        // i think we might only need to support the cases where the tensor
        // rank is 1 or 2, because it might be impossible to tell if the
        // user wants to perform vec-wise or mat-wise operation when when
        // input tensor rank is higher than that...
    }

    fw = () => ops.matmul(this.A, this.B, this.value);

    bw() {
        const A = this.parents[0];
        const B = this.parents[1];

        if (A.grad) ops.matmul_acc(this.grad, this.B_T, A.grad);
        if (B.grad) ops.matmul_acc(this.A_T, this.grad, B.grad);
    }
}

export class Dot extends Tensor {
    value: RawTensor;
    grad: RawTensor;

    A_T: RawTensor;
    B_T: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.create(get_shape_dot(this.parents[0].value, this.parents[1].value));
        this.grad = RawTensor.like(this.value);
        this.A_T = this.parents[0].value.T;
        this.B_T = this.parents[1].value.T;
    }

    fw = () => ops.dot(this.parents[0].value, this.parents[1].value, this.value);

    bw() {
        // todo: validate - it is likely that we need to handle dot differently than matmul

        const A = this.parents[0];
        const B = this.parents[1];

        if (A.grad) ops.dot_acc(this.grad, this.B_T, A.grad);
        if (B.grad) ops.dot_acc(this.A_T, this.grad, B.grad);
    }
}

export class Transpose extends Tensor {
    value: RawTensor;
    grad?: RawTensor;

    constructor(parents: Tensor[], ...permutation: number[]) {
        super(parents);
        this.value = parents[0].value.transpose(...permutation);
        this.grad = parents[0].grad?.transpose(...permutation);
    }
}

export class Min extends Tensor {
    value: RawTensor;
    grad: RawTensor;
    grad_view?: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = parents[0].value.create_view(parents[0].value.rank);
        this.grad = RawTensor.scalar(0);
        if (this.parents[0].grad) this.grad_view = this.parents[0].grad.create_view(this.parents[0].grad.rank);
    }

    fw() {
        const linear_idx = ops.min_idx(this.parents[0].value);
        ops.shift_view(this.value, linear_idx);
        if (this.grad_view) ops.shift_view(this.grad_view, linear_idx);
    }

    bw() {
        if (!this.parents[0].grad) return;
        ops.add(this.grad_view!, this.grad, this.grad_view);
    }
}

export class Max extends Min {
    fw() {
        const linear_idx = ops.max_idx(this.parents[0].value);
        ops.shift_view(this.value, linear_idx);
        if (this.grad_view) ops.shift_view(this.grad_view, linear_idx);
    }
}

export class Sum extends Tensor {
    value: RawTensor;
    grad: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.scalar();
        this.grad = RawTensor.scalar();
    }

    fw = () => ops.sum_tns(this.parents[0].value, this.value);

    bw() {
        if (!this.parents[0].grad) return;
        ops.add(this.parents[0].grad, this.grad, this.parents[0].grad);
    }
}

// OLD MEAN IMPL. APPARENTLY INCORRECT
export class Mean extends Tensor {
    value: RawTensor;
    grad: RawTensor;
    interim: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.scalar(); // Scalar tensor to hold the mean value
        this.grad = RawTensor.like(this.value); // Gradient tensor with the same shape as input
        this.interim = RawTensor.like(this.parents[0].value);
    }

    fw = () => ops.mean_tns(this.parents[0].value, this.value);

    bw() {
        if (!this.parents[0].grad) return;
        ops.div(this.grad, this.parents[0].value.nelem, this.interim);
        ops.add(this.parents[0].grad, this.interim, this.parents[0].grad);
    }
}

export class MseLoss extends Tensor {
    value: RawTensor;
    grad: RawTensor;

    // intermediate values
    interim: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);

        // todo: add shape compat check
        this.value = RawTensor.scalar(0);
        this.grad = RawTensor.like(parents[0].value).ones(); // todo: this should be set to 1 after at some point (maybe reintroduce init()?)
        this.interim = RawTensor.like(parents[0].value);
    }

    fw() {
        const prediction = this.parents[0].value;
        const target = this.parents[1].value;

        // todo: for perf optimizations, this could be moved to the core as a single operation
        ops.sub(prediction, target, this.interim);
        ops.pow(this.interim, 2, this.interim);
        this.value.fill(ops.mean(this.interim));
    }

    bw() {
        const prediction = this.parents[0];
        const target = this.parents[1];

        if (!prediction.grad && !target.grad) return;
        ops.sub(prediction.value, target.value, this.interim);

        if (prediction.grad) {
            // gradient of MSE loss w.r.t. prediction: 2 * (prediction - target) / N       
            ops.mul_acc(this.interim, 2 / prediction.value.nelem, prediction.grad);
        }

        // todo fix: no need to to this twice
        if (target.grad) {
            ops.mul_acc(this.interim, -2 / prediction.value.nelem, target.grad);
        }
    }
}

export class Dropout extends FwBwOp {
    p: number;
    seed: number = 0;

    constructor(parents: Tensor[], p: number) {
        super(parents);
        this.p = p;
    }

    fw() {
        // todo: figure out when a new dropout mask should be used
        //       during mini-batch gradient descent
        //       (currently, we create a new mask on every fw)

        // changing the seed is equal to using a new dropout mask
        this.seed = get_global_seed();
        ops.dropout(this.parents[0].value, this.value, this.p, this.seed);
    }

    bw() {
        // todo: figure out which mask to use for backprop after a mini batch
        ops.dropout_acc(this.grad, this.parents[0].grad, this.p, this.seed);
    }
}

export class Relu extends FwBwInterimOp {
    fw = () => ops.relu(this.parents[0].value, this.value);
    bw() {
        // d/dx relu(x) = binstep(x)
        if (!this.parents[0].grad) return;
        ops.df_relu(this.parents[0].value, this.interim);
        ops.mul_acc(this.interim, this.grad, this.parents[0].grad);
    }
}

export class LeakyRelu extends FwBwInterimOp {
    neg_slope: number;

    constructor(parents: Tensor[], neg_slope: number) {
        super(parents);
        this.neg_slope = neg_slope;
    }

    fw = () => ops.leaky_relu(this.parents[0].value, this.value, this.neg_slope);
    bw() {
        // d/dx leaky_relu(x) = x < 0 ? neg_slope : 1
        if (!this.parents[0].grad) return;
        ops.df_leaky_relu(this.parents[0].value, this.interim, this.neg_slope);
        ops.mul_acc(this.grad, this.interim, this.parents[0].grad);
    }
}

export class Logistic extends FwBwInterimOp {
    one = RawTensor.scalar(1);

    fw = () => ops.logistic(this.parents[0].value, this.value);
    bw() {
        // df/dx = f(x) * (1 - f(x))
        if (!this.parents[0].grad) return;
        ops.sub(this.one, this.value, this.interim);  // interim = 1 - value
        ops.mul(this.value, this.interim, this.interim); // this.grad = value * (1 - value)
        ops.mul_acc(this.grad, this.interim, this.parents[0].grad);
    }
}

export class Negate extends FwBwOp {
    fw = () => ops.negate(this.parents[0].value, this.value);
    bw() {
        // d/dx -x = -1
        if (!this.parents[0].grad) return;
        ops.negate_acc(this.grad, this.parents[0].grad);
    }
}

export class Sin extends FwBwInterimOp {
    fw = () => ops.sin(this.parents[0].value, this.value);
    bw() {
        // d/dx sin(x) = cos(x)
        if (!this.parents[0].grad) return;
        ops.df_sin(this.parents[0].value, this.interim);
        ops.mul_acc(this.grad, this.interim, this.parents[0].grad);
    }
}

export class Cos extends FwBwInterimOp {
    fw = () => ops.cos(this.parents[0].value, this.value);
    bw() {
        // d/dx cos(x) = -sin(x)
        if (!this.parents[0].grad) return;
        ops.df_cos(this.parents[0].value, this.interim);
        ops.mul_acc(this.grad, this.interim, this.parents[0].grad);
    }
}

export class Tan extends FwBwInterimOp {
    fw = () => ops.tan(this.parents[0].value, this.value);
    bw() {
        // d/dx tan(x) = sec^2(x) = 1 / cos^2(x)
        if (!this.parents[0].grad) return;
        ops.df_tan(this.parents[0].value, this.interim);
        ops.mul_acc(this.grad, this.interim, this.parents[0].grad);
    }
}

export class Asin extends FwBwInterimOp {
    fw = () => ops.asin(this.parents[0].value, this.value);
    bw() {
        // d/dx asin(x) = 1 / sqrt(1 - x^2)
        if (!this.parents[0].grad) return;
        ops.df_asin(this.parents[0].value, this.interim);
        ops.mul_acc(this.grad, this.interim, this.parents[0].grad);
    }
}

export class Acos extends FwBwInterimOp {
    fw = () => ops.acos(this.parents[0].value, this.value);
    bw() {
        // d/dx asin(x) = -1 / sqrt(1 - x^2)
        if (!this.parents[0].grad) return;
        ops.df_acos(this.parents[0].value, this.interim);
        ops.mul_acc(this.grad, this.interim, this.parents[0].grad);
    }
}

export class Atan extends FwBwInterimOp {
    fw = () => ops.atan(this.parents[0].value, this.value);
    bw() {
        // d/dx atan(x) = asec^2(x) = 1 / (x^2 + 1)
        if (!this.parents[0].grad) return;
        ops.df_atan(this.parents[0].value, this.interim);
        ops.mul_acc(this.grad, this.interim, this.parents[0].grad);
    }
}

export class Sinh extends FwBwInterimOp {
    fw = () => ops.sinh(this.parents[0].value, this.value);
    bw() {
        // d/dx sinh(x) = cosh(x)
        if (!this.parents[0].grad) return;
        ops.df_sinh(this.parents[0].value, this.interim);
        ops.mul_acc(this.grad, this.interim, this.parents[0].grad);
    }
}

export class Cosh extends FwBwInterimOp {
    fw = () => ops.cosh(this.parents[0].value, this.value);
    bw() {
        // d/dx cosh(x) = sinh(x)
        if (!this.parents[0].grad) return;
        ops.df_cosh(this.parents[0].value, this.interim);
        ops.mul_acc(this.grad, this.interim, this.parents[0].grad);
    }
}

export class Tanh extends FwBwInterimOp {
    fw = () => ops.tanh(this.parents[0].value, this.value);
    bw() {
        // d/dx tanh(x) = 1 - tanh^2(x)
        if (!this.parents[0].grad) return;
        ops.df_tanh(this.parents[0].value, this.interim);
        ops.mul_acc(this.grad, this.interim, this.parents[0].grad);
    }
}

export class Exp extends FwBwOp {
    fw = () => ops.exp(this.parents[0].value, this.value);
    bw() {
        // d/dx e^x = e^x
        if (!this.parents[0].grad) return;
        ops.add(this.value, this.parents[0].grad, this.parents[0].grad);
    }
}

export class Log extends FwBwInterimOp {
    fw = () => ops.log(this.parents[0].value, this.value);
    bw() {
        // d/dx ln(x) = 1/x
        if (!this.parents[0].grad) return;
        ops.df_log(this.parents[0].value, this.interim);
        ops.mul_acc(this.grad, this.interim, this.parents[0].value);
    }
}

export class Log10 extends FwBwInterimOp {
    ln10 = Math.log(10);
    fw = () => ops.log10(this.parents[0].value, this.value);
    bw() {
        // d/dx log10(x) = 1 / (x * ln(10))
        if (!this.parents[0].grad) return;
        ops.df_log10(this.parents[0].value, this.interim);
        ops.mul_acc(this.grad, this.interim, this.parents[0].grad);
    }
}

export class Log2 extends FwBwInterimOp {
    ln2 = Math.log(2);

    fw = () => ops.log2(this.parents[0].value, this.value);
    bw() {
        // d/dx log2(x) = 1 / (x * ln(2))
        if (!this.parents[0].grad) return;
        ops.df_log2(this.parents[0].value, this.interim);
        ops.mul_acc(this.grad, this.interim, this.parents[0].grad);
    }
}

export class Invsqrt extends FwBwInterimOp {
    fw = () => ops.invsqrt(this.parents[0].value, this.value);
    bw() {
        // d/dx 1 / sqrt(x) = - 1 / (2x^(3/2))
        if (!this.parents[0].grad) return;
        ops.df_invsqrt(this.parents[0].value, this.interim);
        ops.mul_acc(this.grad, this.interim, this.parents[0].grad);
    }
}

export class Sqrt extends FwBwInterimOp {
    fw = () => ops.sqrt(this.parents[0].value, this.value);
    bw() {
        // d/dx sqrt(x) = 1 / (2 * sqrt(x))
        if (!this.parents[0].grad) return;
        ops.df_sqrt(this.parents[0].value, this.interim);
        ops.mul_acc(this.grad, this.interim, this.parents[0].grad);
    }
}

export class Abs extends FwBwInterimOp {
    fw = () => ops.abs(this.parents[0].value, this.value);
    bw() {
        // d/dx |x| = sign(x)
        if (!this.parents[0].grad) return;
        ops.df_abs(this.parents[0].value, this.interim);
        ops.mul_acc(this.grad, this.interim, this.parents[0].grad);
    }
}

export class Reciprocal extends FwBwInterimOp {
    fw = () => ops.reciprocal(this.parents[0].value, this.value);
    bw() {
        // d/dx 1 / x = -1 / (x^2)
        if (!this.parents[0].grad) return;
        ops.df_reciprocal(this.parents[0].value, this.interim);
        ops.mul_acc(this.grad, this.interim, this.parents[0].grad);
    }
}

/**
 * The following operations are special in the sense that they
 * do not propagate gradients backward.
 * 
 * todo:
 * maybe we should notify the user that these can prevent
 * branches of the graph from receiving gradients.
 */
export class Ceil extends FwOp {
    fw = () => ops.ceil(this.parents[0].value, this.value);
}

export class Floor extends FwOp {
    fw = () => ops.floor(this.parents[0].value, this.value);
}

export class Binstep extends FwOp {
    fw = () => ops.binstep(this.parents[0].value, this.value);
}
