import * as ops from "../raw_tensor/raw_tensor_operations.ts";
import { get_shape_dot, get_shape_matmul } from "../raw_tensor/raw_tensor_operations.ts";
import {RawTensor} from "../raw_tensor/raw_tensor.ts";
import Shape from "../raw_tensor/shape.ts";
import { get_global_seed } from "../raw_tensor/util.ts";
import Tensor from "../tensor.ts";
import { tensor } from "../tensor_factory.ts";
import { binary_error } from "../raw_tensor/to_string.ts";

// This file contains all operations of the graph-node abstraction-level
// These are essentially all operations of the tensor level plus their derivatives

// FwOps only have primals (no grad, no backprop)
export class FwOp extends Tensor {
    value: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.create(this.a.value.shape);
    }
}

// FwBwOps have primals and gradients
export class FwBwOp extends FwOp {
    grad: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.grad = RawTensor.create(this.value.shape);
    }
}

// FwBwInterimOps have primals, grads and an interim to hold intermediate results
export class FwBwInterimOp extends FwBwOp {
    interim: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.interim = RawTensor.create(this.value.shape);
    }
}

// Constants don't have parents, never change and don't need gradients
export class Constant extends Tensor {
    value: RawTensor;

    constructor(value: RawTensor | number) {
        super([]);
        this.value = typeof value === "number" ? RawTensor.scalar(value) : value;
    }
}

// Parameters don't have parents, can change and do require gradients
export class Parameter extends Tensor {
    value: RawTensor;
    grad: RawTensor;

    constructor(value: RawTensor | number) {
        super([]);
        this.value = typeof value === "number" ? RawTensor.scalar(value) : value;
        this.grad = RawTensor.create(this.value.shape);
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

    fw() {
        const item = this.producer();
        ops.identity(item, this.value);
    }
}

// This is just an identity operation with dynamically adjustable parents.
// note: not thread safe
export class Input extends Tensor {
    value: RawTensor;
    grad: RawTensor;
    connected: boolean = false;

    constructor(shape: Shape | number[], grad_shape: Shape | number[] = shape) {
        super([]);
        this.value = RawTensor.create(shape);
        this.grad = RawTensor.create(grad_shape);
    }

    // connects (the graph of) some input tensor with the graph that the input node is part of
    connect(input: Tensor) {
        if (!input.shape.equals(this.value.shape)) throw new Error(`Input node of shape [${this.value.shape}] cannot handle input tensor of shape [${input.shape}].`);
        if (input.grad && !input.grad.shape.equals(this.grad.shape)) throw new Error(`Input node with grad shape [${this.grad.shape}] cannot handle input tensor with grad shape [${input.grad.shape}].`);
        
        this.parents = [input];
        this.a.add_child(this);
        this.connected = true;
    }

    // disconnects the graphs
    disconnect() {
        this.a.remove_child(this);
        this.parents = [];
        this.connected = false;
    }

    // identity operation. this.value is replaced by the parent value
    fw () {
        if (!this.connected) throw new Error("Cannot perform forward pass. Input node is disconnected.");
        ops.identity(this.a.value, this.value);
    }

    // directly accumulate gradient from this node into the parent grad
    bw() {
        if (!this.connected) throw new Error("Cannot perform backward pass. Input node is disconnected.");
        if (this.a.grad) ops.identity_acc(this.grad, this.a.grad);
    }
}

export class Add extends Tensor {
    value: RawTensor;
    grad: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.create(this.a.value.shape.broadcast(this.b.value.shape));
        this.grad = RawTensor.create(this.value.shape);
    }

    fw = () => ops.add(this.a.value, this.b.value, this.value);

    bw() {
        // d/da (a+b) = 1
        if (this.a.grad) ops.add(this.grad, this.a.grad, this.a.grad); // parents[0].grad = 1 * this.grad

        // d/db (a+b) = 1
        if (this.b.grad) ops.add(this.grad, this.b.grad, this.b.grad); // parents[0].grad = 1 * this.grad
    }
}

export class Sub extends Tensor {
    value: RawTensor;
    grad: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.create(this.a.value.shape.broadcast(this.b.value.shape));
        this.grad = RawTensor.create(this.value.shape);
    }

    fw = () => ops.sub(this.a.value, this.b.value, this.value);

    bw() {
        // d/da (a-b) = 1
        if (this.a.grad) ops.add(this.a.grad, this.grad, this.a.grad); // parents[0].grad = 1 * this.grad

        // d/db (a-b) = -1
        if (this.b.grad) ops.sub(this.b.grad, this.grad, this.b.grad); // parents[1].grad parents[0].grad = -1 * this.grad
    }
}

export class Mul extends Tensor {
    value: RawTensor;
    grad: RawTensor;
    interim: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.create(this.a.value.shape.broadcast(this.b.value.shape));
        this.grad = RawTensor.create(this.value.shape);
        this.interim = RawTensor.create(this.value.shape);
    }

    fw = () => ops.mul(this.a.value, this.b.value, this.value);

    bw() {
        const a = this.a;
        const b = this.b;

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
        this.value = RawTensor.create(this.a.value.shape.broadcast(this.b.value.shape));
        this.grad = RawTensor.create(this.value.shape);
        this.interim = RawTensor.create(this.value.shape);
    }

    fw = () => ops.div(this.a.value, this.b.value, this.value);

    bw() {
        const a = this.a;
        const b = this.b;

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
        this.value = RawTensor.create(this.a.value.shape.broadcast(this.b.value.shape));
        this.grad = RawTensor.create(this.value.shape);
        this.interim_0 = RawTensor.create(this.a.shape);
        this.interim_1 = RawTensor.create(this.b.shape);
    }

    fw = () => ops.pow(this.a.value, this.b.value, this.value);

    bw() {
        const a = this.a.value; // base
        const b = this.b.value; // exponent

        if (this.a.grad) {
            // d/da (a^b) = b * a^(b-1)
            ops.sub(b, 1, this.interim_1); // interim = b - 1
            ops.pow(a, this.interim_1, this.interim_0); // interim = a^(b-1)
            ops.mul(b, this.interim_0, this.interim_0); // interim = b * a^(b-1)
            ops.mul(this.grad, this.interim_0, this.interim_0); // interim = grad * b * a^(b-1)
            ops.add_acc(this.interim_0, this.a.grad, this.a.grad); // parents[0].grad += interim
        }

        if (this.b.grad) {
            // todo validate
            // d/db (a^b) = a^b * ln(a)
            ops.log(a, this.interim_0); // interim = ln(a)
            ops.mul(this.value, this.interim_0, this.interim_0); // interim = a^b * ln(a)
            ops.mul(this.grad, this.interim_0, this.interim_0); // interim = grad * a^b * ln(a)
            ops.add_acc(this.interim_0, this.b.grad, this.b.grad); // parents[1].grad += interim
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
    grad_A?: RawTensor;
    grad_B?: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        
        // extend vectors such that they can be multiplied
        this.A = this.a.rank === 1 ? this.a.value.left_extend() : this.a.value;
        this.B = this.b.rank === 1 ? this.b.value.right_extend() : this.b.value;

        // we also have to extend the gradients if vectors are involved
        if (this.a.grad) this.grad_A = this.a.rank === 1 ? this.a.grad.left_extend() : this.a.grad;
        if (this.b.grad) this.grad_B = this.b.rank === 1 ? this.b.grad.right_extend() : this.b.grad;

        this.A_T = this.A.T;
        this.B_T = this.B.T;

        // todo: this can throw an error during graph acquisition, which means we cant display the
        //       graph. makes things difficult to debug
        this.value = RawTensor.create(get_shape_matmul(this.A, this.B));
        this.grad = RawTensor.create(this.value.shape);
    }

    fw = () => ops.matmul(this.A, this.B, this.value);

    bw() {
        // if (this.b.grad) throw new Error(binary_error("matmul_acc", this.A_T, this.grad, this.b.grad, get_shape_matmul(this.A_T, this.grad)));

        if (this.grad_A) ops.matmul_acc(this.grad, this.B_T, this.grad_A);
        if (this.grad_B) ops.matmul_acc(this.A_T, this.grad, this.grad_B);
    }
}

// Todo: validate
export class Dot extends Tensor {
    value: RawTensor;
    grad: RawTensor;

    // these are views and therefore don't need a lot of memory
    A: RawTensor;
    B: RawTensor;
    A_T: RawTensor;
    B_T: RawTensor;
    grad_A?: RawTensor;
    grad_B?: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        
        // extend vectors such that they can be multiplied
        this.A = this.a.value;
        this.B = this.b.value;

        // we also have to extend the gradients if vectors are involved
        if (this.a.grad) this.grad_A = this.a.grad;
        if (this.b.grad) this.grad_B = this.b.grad;

        this.A_T = this.A.T;
        this.B_T = this.B.T;

        // todo: this can throw an error during graph acquisition, which means we cant display the
        //       graph. makes things difficult to debug
        this.value = RawTensor.create(get_shape_dot(this.A, this.B));
        this.grad = RawTensor.create(this.value.shape);
    }

    fw = () => ops.dot(this.A, this.B, this.value);

    bw() {
        // if (this.b.grad) throw new Error(binary_error("matmul_acc", this.A_T, this.grad, this.b.grad, get_shape_matmul(this.A_T, this.grad)));

        if (this.grad_A) ops.dot_acc(this.grad, this.B_T, this.grad_A);
        if (this.grad_B) ops.dot_acc(this.A_T, this.grad, this.grad_B);
    }
}

// export class Dot extends Tensor {
//     value: RawTensor;
//     grad: RawTensor;

//     A_T: RawTensor;
//     B_T: RawTensor;

//     constructor(parents: Tensor[]) {
//         super(parents);
//         this.value = RawTensor.create(get_shape_dot(this.a.value, this.b.value));
//         this.grad = RawTensor.create(this.value.shape);
//         this.A_T = this.a.value.T;
//         this.B_T = this.b.value.T;
//     }

//     fw = () => ops.dot(this.a.value, this.b.value, this.value);

//     bw() {
//         // todo: validate - it is likely that we need to handle dot differently than matmul

//         if (this.a.grad) ops.dot_acc(this.grad, this.B_T, A.grad);
//         if (this.b.grad) ops.dot_acc(this.A_T, this.grad, B.grad);
//     }
// }

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
        if (this.a.grad) this.grad_view = this.a.grad.create_view(this.a.grad.rank);
    }

    fw() {
        const linear_idx = ops.min_idx(this.a.value);
        ops.shift_view(this.value, linear_idx);
        if (this.grad_view) ops.shift_view(this.grad_view, linear_idx);
    }

    bw() {
        if (!this.a.grad) return;
        ops.add(this.grad_view!, this.grad, this.grad_view);
    }
}

export class Max extends Min {
    fw() {
        const linear_idx = ops.max_idx(this.a.value);
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

    fw = () => ops.sum_tns(this.a.value, this.value);
    bw() {
        if (!this.a.grad) return;
        ops.add(this.a.grad, this.grad, this.a.grad);
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
        this.grad = RawTensor.create(this.value.shape); // Gradient tensor with the same shape as input
        this.interim = RawTensor.create(this.a.value.shape);
    }

    fw = () => ops.mean_tns(this.a.value, this.value);
    bw() {
        if (!this.a.grad) return;
        ops.div(this.grad, this.a.value.nelem, this.interim);
        ops.add(this.a.grad, this.interim, this.a.grad);
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
        this.grad = RawTensor.create(parents[0].value.shape).ones(); // todo: this should be reset to 1 at some point (maybe reintroduce init()?)
        this.interim = RawTensor.create(parents[0].value.shape);
    }

    fw() {
        const prediction = this.a.value;
        const target = this.b.value;

        // todo: for perf optimizations, this could be moved to the core as a single operation
        ops.sub(prediction, target, this.interim);
        ops.pow(this.interim, 2, this.interim);
        this.value.fill(ops.mean(this.interim));
    }

    bw() {
        const prediction = this.a;
        const target = this.b;

        if (!prediction.grad && !target.grad) return;
        ops.sub(prediction.value, target.value, this.interim);

        // gradient of MSE loss w.r.t. prediction: 2 * (prediction - target) / N       
        if (prediction.grad) ops.mul_acc(this.interim, 2 / prediction.value.nelem, prediction.grad);
        if (target.grad) ops.mul_acc(this.interim, -2 / prediction.value.nelem, target.grad);
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
        ops.dropout(this.a.value, this.value, this.p, this.seed);
    }

    bw() {
        // todo: figure out which mask to use for backprop after a mini batch
        ops.dropout_acc(this.grad, this.a.grad, this.p, this.seed);
    }
}

export class Relu extends FwBwInterimOp {
    fw = () => ops.relu(this.a.value, this.value);
    bw() {
        // d/dx relu(x) = binstep(x)
        if (!this.a.grad) return;
        ops.df_relu(this.a.value, this.interim);
        ops.mul_acc(this.interim, this.grad, this.a.grad);
    }
}

export class LeakyRelu extends FwBwInterimOp {
    neg_slope: number;

    constructor(parents: Tensor[], neg_slope: number) {
        super(parents);
        this.neg_slope = neg_slope;
    }

    fw = () => ops.leaky_relu(this.a.value, this.value, this.neg_slope);
    bw() {
        // d/dx leaky_relu(x) = x < 0 ? neg_slope : 1
        if (!this.a.grad) return;
        ops.df_leaky_relu(this.a.value, this.interim, this.neg_slope);
        ops.mul_acc(this.grad, this.interim, this.a.grad);
    }
}

export class Logistic extends FwBwInterimOp {
    one = RawTensor.scalar(1);

    fw = () => ops.logistic(this.a.value, this.value);
    bw() {
        // df/dx = f(x) * (1 - f(x))
        if (!this.a.grad) return;
        ops.sub(this.one, this.value, this.interim);  // interim = 1 - value
        ops.mul(this.value, this.interim, this.interim); // this.grad = value * (1 - value)
        ops.mul_acc(this.grad, this.interim, this.a.grad);
    }
}

export class Negate extends FwBwOp {
    fw = () => ops.negate(this.a.value, this.value);
    bw() {
        // d/dx -x = -1
        if (!this.a.grad) return;
        ops.negate_acc(this.grad, this.a.grad);
    }
}

export class Sin extends FwBwInterimOp {
    fw = () => ops.sin(this.a.value, this.value);
    bw() {
        // d/dx sin(x) = cos(x)
        if (!this.a.grad) return;
        ops.df_sin(this.a.value, this.interim);
        ops.mul_acc(this.grad, this.interim, this.a.grad);
    }
}

export class Cos extends FwBwInterimOp {
    fw = () => ops.cos(this.a.value, this.value);
    bw() {
        // d/dx cos(x) = -sin(x)
        if (!this.a.grad) return;
        ops.df_cos(this.a.value, this.interim);
        ops.mul_acc(this.grad, this.interim, this.a.grad);
    }
}

export class Tan extends FwBwInterimOp {
    fw = () => ops.tan(this.a.value, this.value);
    bw() {
        // d/dx tan(x) = sec^2(x) = 1 / cos^2(x)
        if (!this.a.grad) return;
        ops.df_tan(this.a.value, this.interim);
        ops.mul_acc(this.grad, this.interim, this.a.grad);
    }
}

export class Asin extends FwBwInterimOp {
    fw = () => ops.asin(this.a.value, this.value);
    bw() {
        // d/dx asin(x) = 1 / sqrt(1 - x^2)
        if (!this.a.grad) return;
        ops.df_asin(this.a.value, this.interim);
        ops.mul_acc(this.grad, this.interim, this.a.grad);
    }
}

export class Acos extends FwBwInterimOp {
    fw = () => ops.acos(this.a.value, this.value);
    bw() {
        // d/dx asin(x) = -1 / sqrt(1 - x^2)
        if (!this.a.grad) return;
        ops.df_acos(this.a.value, this.interim);
        ops.mul_acc(this.grad, this.interim, this.a.grad);
    }
}

export class Atan extends FwBwInterimOp {
    fw = () => ops.atan(this.a.value, this.value);
    bw() {
        // d/dx atan(x) = asec^2(x) = 1 / (x^2 + 1)
        if (!this.a.grad) return;
        ops.df_atan(this.a.value, this.interim);
        ops.mul_acc(this.grad, this.interim, this.a.grad);
    }
}

export class Sinh extends FwBwInterimOp {
    fw = () => ops.sinh(this.a.value, this.value);
    bw() {
        // d/dx sinh(x) = cosh(x)
        if (!this.a.grad) return;
        ops.df_sinh(this.a.value, this.interim);
        ops.mul_acc(this.grad, this.interim, this.a.grad);
    }
}

export class Cosh extends FwBwInterimOp {
    fw = () => ops.cosh(this.a.value, this.value);
    bw() {
        // d/dx cosh(x) = sinh(x)
        if (!this.a.grad) return;
        ops.df_cosh(this.a.value, this.interim);
        ops.mul_acc(this.grad, this.interim, this.a.grad);
    }
}

export class Tanh extends FwBwInterimOp {
    fw = () => ops.tanh(this.a.value, this.value);
    bw() {
        // d/dx tanh(x) = 1 - tanh^2(x)
        if (!this.a.grad) return;
        ops.df_tanh(this.a.value, this.interim);
        ops.mul_acc(this.grad, this.interim, this.a.grad);
    }
}

export class Exp extends FwBwOp {
    fw = () => ops.exp(this.a.value, this.value);
    bw() {
        // d/dx e^x = e^x
        if (!this.a.grad) return;
        ops.add(this.value, this.a.grad, this.a.grad);
    }
}

export class Log extends FwBwInterimOp {
    fw = () => ops.log(this.a.value, this.value);
    bw() {
        // d/dx ln(x) = 1/x
        if (!this.a.grad) return;
        ops.df_log(this.a.value, this.interim);
        ops.mul_acc(this.grad, this.interim, this.a.value);
    }
}

export class Log10 extends FwBwInterimOp {
    ln10 = Math.log(10);
    fw = () => ops.log10(this.a.value, this.value);
    bw() {
        // d/dx log10(x) = 1 / (x * ln(10))
        if (!this.a.grad) return;
        ops.df_log10(this.a.value, this.interim);
        ops.mul_acc(this.grad, this.interim, this.a.grad);
    }
}

export class Log2 extends FwBwInterimOp {
    ln2 = Math.log(2);

    fw = () => ops.log2(this.a.value, this.value);
    bw() {
        // d/dx log2(x) = 1 / (x * ln(2))
        if (!this.a.grad) return;
        ops.df_log2(this.a.value, this.interim);
        ops.mul_acc(this.grad, this.interim, this.a.grad);
    }
}

export class Invsqrt extends FwBwInterimOp {
    fw = () => ops.invsqrt(this.a.value, this.value);
    bw() {
        // d/dx 1 / sqrt(x) = - 1 / (2x^(3/2))
        if (!this.a.grad) return;
        ops.df_invsqrt(this.a.value, this.interim);
        ops.mul_acc(this.grad, this.interim, this.a.grad);
    }
}

export class Sqrt extends FwBwInterimOp {
    fw = () => ops.sqrt(this.a.value, this.value);
    bw() {
        // d/dx sqrt(x) = 1 / (2 * sqrt(x))
        if (!this.a.grad) return;
        ops.df_sqrt(this.a.value, this.interim);
        ops.mul_acc(this.grad, this.interim, this.a.grad);
    }
}

export class Abs extends FwBwInterimOp {
    fw = () => ops.abs(this.a.value, this.value);
    bw() {
        // d/dx |x| = sign(x)
        if (!this.a.grad) return;
        ops.df_abs(this.a.value, this.interim);
        ops.mul_acc(this.grad, this.interim, this.a.grad);
    }
}

export class Reciprocal extends FwBwInterimOp {
    fw = () => ops.reciprocal(this.a.value, this.value);
    bw() {
        // d/dx 1 / x = -1 / (x^2)
        if (!this.a.grad) return;
        ops.df_reciprocal(this.a.value, this.interim);
        ops.mul_acc(this.grad, this.interim, this.a.grad);
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
    fw = () => ops.ceil(this.a.value, this.value);
}

export class Floor extends FwOp {
    fw = () => ops.floor(this.a.value, this.value);
}

export class Binstep extends FwOp {
    fw = () => ops.binstep(this.a.value, this.value);
}
