import * as ops from "./base/tensor_operations.ts";
import {Tensor, tensor_like, tensor_scalar} from "./base/Tensor.ts";
import {tensor} from "../index.ts";
import Shape from "./base/Shape.ts";
import CompGraphNode from "./CompGraphNode.ts";

// This file contains all operations of the graph-node abstraction-level
// These are essentially all operations of the tensor level plus their derivatives

export class Parameter extends CompGraphNode {
    value: Tensor;

    constructor(value: Tensor | number, requires_grad: boolean) {
        super([]);
        this.value = typeof value === "number" ? tensor_scalar(value) : value;
        if (requires_grad) this.grad = tensor_like(this.value);
    }
}

export class Source extends CompGraphNode {
    value: Tensor;
    producer: () => Tensor;

    constructor(shape: Shape | number[], producer: () => Tensor) {
        super([]);

        this.value = tensor(shape);
        this.producer = producer;
    }

    fw() {
        this.value = this.producer();
    }
}

export class Add extends CompGraphNode {
    value: Tensor;
    grad: Tensor;

    constructor(parents: CompGraphNode[]) {
        super(parents);
        this.value = tensor(this.parents[0].value.shape.broadcast(this.parents[1].value.shape));
        this.grad = tensor_like(this.value);
    }

    fw() {
        ops.add(this.parents[0].value, this.parents[1].value, this.value);
    }

    bw() {
        // d/da (a+b) = 1
        if (this.parents[0].grad) {
            ops.add(this.parents[0].grad, this.grad, this.parents[0].grad); // parents[0].grad = 1 * this.grad

            //           data       first parent          second parent         destination
            ops.add_dbcr(this.grad, this.parents[0].grad, this.parents[1].grad, this.parents[0].grad);
        }

        // d/da (a+b) = 1
        if (this.parents[1].grad) {
            ops.add(this.parents[1].grad, this.grad, this.parents[1].grad); // parents[1].grad = 1 * this.grad
        }
    }
}

export class Sub extends CompGraphNode {
    value: Tensor;
    grad: Tensor;

    constructor(parents: CompGraphNode[]) {
        super(parents);
        this.value = tensor(this.parents[0].value.shape.broadcast(this.parents[1].value.shape));
        this.grad = tensor_like(this.value);
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

export class Mul extends CompGraphNode {
    value: Tensor;
    grad: Tensor;
    interim: Tensor;

    constructor(parents: CompGraphNode[]) {
        super(parents);
        this.value = tensor(this.parents[0].value.shape.broadcast(this.parents[1].value.shape));
        this.grad = tensor_like(this.value);
        this.interim = tensor_like(this.value);
    }

    fw() {
        ops.mul(this.parents[0].value, this.parents[1].value, this.value);
    }

    bw() {
        const a = this.parents[0];
        const b = this.parents[1];

        // d/da (a*b) = a
        if (a.grad) {
            ops.mul(b.value, this.grad, this.interim);
            ops.add(a.grad, this.interim, a.grad);
        }

        // d/db (a*b) = b
        if (b.grad) {
            ops.mul(a.value, this.grad, this.interim);
            ops.add(b.grad, this.interim, b.grad);
        }
    }
}

export class Div extends CompGraphNode {
    value: Tensor;
    grad: Tensor;
    interim: Tensor;

    constructor(parents: CompGraphNode[]) {
        super(parents);
        this.value = tensor(this.parents[0].value.shape.broadcast(this.parents[1].value.shape));
        this.grad = tensor_like(this.value);
        this.interim = tensor_like(this.value);
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
            ops.add(a.grad, this.interim, a.grad);
        }

        // d/db (a/b) = -a/(b^2)
        if (b.grad) {
            ops.pow(b.value, 2, this.interim);
            ops.div(a.value, this.interim, this.interim);
            b.grad.sub(this.interim);
        }
    }
}

export class Pow extends CompGraphNode {
    value: Tensor;
    grad: Tensor;
    interim: Tensor;

    constructor(parents: CompGraphNode[]) {
        super(parents);
        this.value = tensor(this.parents[0].value.shape.broadcast(this.parents[1].value.shape));
        this.grad = tensor_like(this.value);
        this.interim = tensor_like(this.value);
    }

    fw() {
        ops.pow(this.parents[0].value, this.parents[1].value, this.value);
    }

    bw() {
        const base = this.parents[0].value;
        const exponent = this.parents[1].value;

        // TODO: i think this is all wrong
        //       here, i am trying to store tensors derived from both exponent and
        //       base in the same interim, even though they may have different shapes

        if (this.parents[0].grad) {
            // d/da (a^b) = b * a^(b-1)
            ops.sub(exponent, 1, this.interim); // interim = b - 1
            ops.pow(base, this.interim, this.interim); // interim = a^(b-1)
            ops.mul(exponent, this.interim, this.interim); // interim = b * a^(b-1)
            ops.mul(this.grad, this.interim, this.interim); // interim = grad * b * a^(b-1)
            // ops.add(this.parents[0].grad, this.interim, this.parents[0].grad); // parents[0].grad += interim

            // undoes broadcasting by summing the gradient along the appropriate axes if necessary,
            // then adds the de-broadcasted tensor to the destination tensor
            ops.grad_acc(this.interim, this.parents[0].grad);
        }

        if (this.parents[1].grad) {
            // d/db (a^b) = a^b * ln(a)
            ops.log(base, this.interim); // interim = ln(a)
            ops.mul(this.value, this.interim, this.interim); // interim = a^b * ln(a)
            ops.mul(this.grad, this.interim, this.interim); // interim = grad * a^b * ln(a)
            // ops.add(this.parents[1].grad, this.interim, this.parents[1].grad); // parents[1].grad += interim

            ops.grad_acc(this.interim, this.parents[1].grad);
        }
    }
}

export class Sum extends CompGraphNode {
    value: Tensor;
    grad: Tensor;

    constructor(parents: CompGraphNode[]) {
        super(parents);
        this.value = tensor_scalar();
        this.grad = tensor_scalar();
    }

    fw() {
        // TODO: Refactor ops.sum to support scalar tensor?
        this.value.data[this.value.get_offset()] = ops.sum(this.parents[0].value);
    }

    bw() {
        const input = this.parents[0];
        if (!input.grad) return;

        ops.add(input.grad, this.grad, input.grad);
    }
}

export class Mean extends CompGraphNode {
    value: Tensor;
    grad: Tensor;
    interim: Tensor;

    constructor(parents: CompGraphNode[]) {
        super(parents);
        this.value = tensor_scalar(); // Scalar tensor to hold the mean value
        this.grad = tensor_like(this.parents[0].value); // Gradient tensor with the same shape as input
        this.interim = tensor_like(this.parents[0].value);
    }

    fw() {
        // todo: fix
        this.value.data[this.value.get_offset()] = ops.mean(this.parents[0].value); // Compute the mean
    }

    bw() {
        const input = this.parents[0];
        if (!input.grad) return;

        ops.div(this.grad, input.value.size, this.interim);
        ops.add(input.grad, this.interim, input.grad);
    }
}


export class MseLoss extends CompGraphNode {
    value: Tensor;
    grad: Tensor;

    // intermediate values
    interim: Tensor;

    constructor(parents: CompGraphNode[]) {
        super(parents);

        // todo: add shape compat check
        this.value = tensor_scalar(0);
        this.grad = tensor_like(parents[0].value).ones(); // todo: this should be set to 1 after at some point (maybe reintroduce init()?)
        this.interim = tensor_like(parents[0].value);
    }

    fw() {
        const prediction = this.parents[0].value;
        const target = this.parents[1].value;

        // todo: for perf optimizations, this could be moved to the core as a single operation
        // compute MSE loss: (prediction - target)^2 / N and store result this.difference
        ops.sub(prediction, target, this.interim);
        ops.pow(this.interim, 2, this.interim);
        this.value.fill(ops.mean(this.interim));
    }

    bw() {
        const prediction = this.parents[0];
        const target = this.parents[1];

        if (prediction.grad) {
            // gradient of MSE loss w.r.t. prediction: 2 * (prediction - target) / N
            ops.sub(prediction.value, target.value, this.grad);
            ops.mul(this.grad, 2 / prediction.value.size, this.grad);
            ops.add(prediction.grad, this.grad, prediction.grad);
        }
    }
}
