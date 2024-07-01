import * as ops from "./base/raw_tensor_operations.ts";
import { get_shape_dot, get_shape_matmul } from "./base/raw_tensor_operations.ts";
import {RawTensor} from "./base/RawTensor.ts";
import Shape from "./base/Shape.ts";
import Tensor from "./Tensor.ts";

// This file contains all operations of the graph-node abstraction-level
// These are essentially all operations of the tensor level plus their derivatives

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

    fw() {
        this.value = this.producer();
    }
}

export class Add extends Tensor {
    value: RawTensor;
    grad: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.create(this.parents[0].value.shape.broadcast(this.parents[1].value.shape));
        this.grad = RawTensor.like(this.value);
    }

    fw() {
        ops.add(this.parents[0].value, this.parents[1].value, this.value);
    }

    bw() {
        // d/da (a+b) = 1
        if (this.parents[0].grad) ops.add(this.grad, this.parents[0].grad, this.parents[0].grad); // parents[0].grad = 1 * this.grad

        // d/da (a+b) = 1
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

    fw() {
        ops.sub(this.parents[0].value, this.parents[1].value, this.value);
    }

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

    fw() {
        ops.mul(this.parents[0].value, this.parents[1].value, this.value);
    }

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

    fw() {
        ops.div(this.parents[0].value, this.parents[1].value, this.value);
    }

    bw() {
        const a = this.parents[0];
        const b = this.parents[1];

        // d/da (a/b) = 1/b
        if (a.grad) {
            ops.div(this.grad, b.value, this.interim);
            ops.add_acc(this.interim, a.grad, a.grad);
        }

        // d/db (a/b) = -a/(b^2)
        if (b.grad) {
            ops.pow(b.value, 2, this.interim);
            ops.div(a.value, this.interim, this.interim);
            ops.sub_acc(this.interim, b.grad, b.grad);
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

    fw() {
        ops.pow(this.parents[0].value, this.parents[1].value, this.value);
    }

    bw() {
        const base = this.parents[0].value;
        const exponent = this.parents[1].value;

        if (this.parents[0].grad) {
            // d/da (a^b) = b * a^(b-1)
            ops.sub(exponent, 1, this.interim_1); // interim = b - 1
            ops.pow(base, this.interim_1, this.interim_0); // interim = a^(b-1)
            ops.mul(exponent, this.interim_0, this.interim_0); // interim = b * a^(b-1)
            ops.mul(this.grad, this.interim_0, this.interim_0); // interim = grad * b * a^(b-1)
            ops.add_acc(this.interim_0, this.parents[0].grad, this.parents[0].grad); // parents[0].grad += interim
        }

        if (this.parents[1].grad) {
            // todo validate
            // d/db (a^b) = a^b * ln(a)
            ops.log(base, this.interim_0); // interim = ln(a)
            ops.mul(this.value, this.interim_0, this.interim_0); // interim = a^b * ln(a)
            ops.mul(this.grad, this.interim_0, this.interim_0); // interim = grad * a^b * ln(a)
            ops.add_acc(this.interim_0, this.parents[1].grad, this.parents[1].grad); // parents[1].grad += interim
        }
    }
}

export class Matmul extends Tensor {
    value: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.create(get_shape_matmul(this.parents[0].value, this.parents[1].value));
    }

    fw() {
        ops.matmul(this.parents[0].value, this.parents[1].value, this.value);
    }

    bw() {
        // todo
    }
}

export class Dot extends Tensor {
    value: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.create(get_shape_dot(this.parents[0].value, this.parents[1].value));
    }

    fw() {
        ops.dot(this.parents[0].value, this.parents[1].value, this.value);
    }

    bw() {
        // todo
    }
}

export class Min extends Tensor {
    value: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.scalar();
    }

    fw() {
        ops.min_tns(this.parents[0].value, this.value);
    }
}

export class Max extends Tensor {
    value: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.scalar();
    }

    fw() {
        ops.max_tns(this.parents[0].value, this.value);
    }

    bw() {
        // todo: for bw, we need to propagate the gradient only to the location of the largest
        //       element. currently we dont have information about what element it was.
        //       solution: max_tns and min_tns as should return scalar views of the source tensor
        //                 we can then get the exact element by using the offset.
        //       problem:  there might be issues with all ops that involve scalars
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

    fw() {
        ops.sum_tns(this.parents[0].value, this.value);
    }

    bw() {
        const input = this.parents[0];
        if (!input.grad) return;

        ops.add(input.grad, this.grad, input.grad);
    }
}

export class Mean extends Tensor {
    value: RawTensor;
    grad: RawTensor;
    interim: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.scalar(); // Scalar tensor to hold the mean value
        this.grad = RawTensor.like(this.parents[0].value); // Gradient tensor with the same shape as input
        this.interim = RawTensor.like(this.parents[0].value);
    }

    fw() {
        ops.mean_tns(this.parents[0].value, this.value);
    }

    bw() {
        const input = this.parents[0];
        if (input.grad) {
            ops.div(this.grad, input.value.size, this.interim);
            ops.add(input.grad, this.interim, input.grad);
        }
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
            ops.mul_acc(this.interim, 2 / prediction.value.size, prediction.grad);
        }

        // todo fix: no need to to this twice
        if (target.grad) {
            ops.mul_acc(this.interim, -2 / prediction.value.size, target.grad);
        }
    }
}

export class Relu extends Tensor {
    value: RawTensor;
    grad: RawTensor;
    interim: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.like(parents[0].value);
        this.grad = RawTensor.like(parents[0].value);
        this.interim = RawTensor.like(parents[0].value);
    }

    fw() {
        ops.relu(this.parents[0].value, this.value);
    }

    bw() {
        if (this.parents[0].grad) {
            ops.binstep(this.parents[0].value, this.interim);
            ops.mul_acc(this.parents[0].grad, this.grad, this.parents[0].grad);
        }
    }
}

/********* THE CLASSES HERE ONLY HAVE FW IMPLEMENTED FOR TESTING *********/

export class Binstep extends Tensor {
    value: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.like(parents[0].value);
    }

    fw() {
        ops.binstep(this.parents[0].value, this.value);
    }
}

export class Logistic extends Tensor {
    value: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.like(parents[0].value);
    }

    fw() {
        ops.logistic(this.parents[0].value, this.value);
    }
}

export class Negate extends Tensor {
    value: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.like(parents[0].value);
    }

    fw() {
        ops.negate(this.parents[0].value, this.value);
    }
}

export class Sin extends Tensor {
    value: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.like(parents[0].value);
    }

    fw() {
        ops.sin(this.parents[0].value, this.value);
    }
}

export class Cos extends Tensor {
    value: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.like(parents[0].value);
    }

    fw() {
        ops.cos(this.parents[0].value, this.value);
    }
}

export class Tan extends Tensor {
    value: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.like(parents[0].value);
    }

    fw() {
        ops.tan(this.parents[0].value, this.value);
    }
}

export class Asin extends Tensor {
    value: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.like(parents[0].value);
    }

    fw() {
        ops.asin(this.parents[0].value, this.value);
    }
}

export class Acos extends Tensor {
    value: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.like(parents[0].value);
    }

    fw() {
        ops.acos(this.parents[0].value, this.value);
    }
}

export class Atan extends Tensor {
    value: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.like(parents[0].value);
    }

    fw() {
        ops.atan(this.parents[0].value, this.value);
    }
}

export class Sinh extends Tensor {
    value: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.like(parents[0].value);
    }

    fw() {
        ops.sinh(this.parents[0].value, this.value);
    }
}

export class Cosh extends Tensor {
    value: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.like(parents[0].value);
    }

    fw() {
        ops.cosh(this.parents[0].value, this.value);
    }
}

export class Tanh extends Tensor {
    value: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.like(parents[0].value);
    }

    fw() {
        ops.tanh(this.parents[0].value, this.value);
    }
}

export class Exp extends Tensor {
    value: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.like(parents[0].value);
    }

    fw() {
        ops.exp(this.parents[0].value, this.value);
    }
}

export class Log extends Tensor {
    value: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.like(parents[0].value);
    }

    fw() {
        ops.log(this.parents[0].value, this.value);
    }
}

export class Log10 extends Tensor {
    value: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.like(parents[0].value);
    }

    fw() {
        ops.log10(this.parents[0].value, this.value);
    }
}

export class Log2 extends Tensor {
    value: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.like(parents[0].value);
    }

    fw() {
        ops.log2(this.parents[0].value, this.value);
    }
}

export class Invsqrt extends Tensor {
    value: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.like(parents[0].value);
    }

    fw() {
        ops.invsqrt(this.parents[0].value, this.value);
    }
}

export class Sqrt extends Tensor {
    value: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.like(parents[0].value);
    }

    fw() {
        ops.sqrt(this.parents[0].value, this.value);
    }
}

export class Ceil extends Tensor {
    value: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.like(parents[0].value);
    }

    fw() {
        ops.ceil(this.parents[0].value, this.value);
    }
}

export class Floor extends Tensor {
    value: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.like(parents[0].value);
    }

    fw() {
        ops.floor(this.parents[0].value, this.value);
    }
}

export class Abs extends Tensor {
    value: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.like(parents[0].value);
    }

    fw() {
        ops.abs(this.parents[0].value, this.value);
    }
}

export class Reciprocal extends Tensor {
    value: RawTensor;

    constructor(parents: Tensor[]) {
        super(parents);
        this.value = RawTensor.like(parents[0].value);
    }

    fw() {
        ops.reciprocal(this.parents[0].value, this.value);
    }
}
