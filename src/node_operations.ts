import * as ops from "./base/tensor_operations.ts";
import {Tensor, tensor_like, tensor_scalar} from "./base/Tensor.ts";
import {tensor} from "../index.ts";
import Shape from "./base/Shape.ts";
import CompGraphNode from "./CompGraphNode.ts";

// This file contains all operations of the graph-node abstraction-level
// These are essentially all operations of the tensor level plus their derivatives

export class ConstScalar extends CompGraphNode {
    value: Tensor;

    constructor(scalar: number) {
        super([]);
        this.value = tensor_scalar(scalar);
    }
}

export class Input extends CompGraphNode {
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

// Pairwise tensor addition
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
        // todo: ensure that the if's are actually what we need here
        if (this.parents[0].grad) ops.add(this.parents[0].grad, this.grad, this.parents[0].grad); // in-place op
        if (this.parents[1].grad) ops.add(this.parents[1].grad, this.grad, this.parents[1].grad); // in-place op
    }
}

export class MseLoss extends CompGraphNode {
    value: Tensor;

    // intermediate values
    interim: Tensor;

    constructor(parents: CompGraphNode[]) {
        // todo: add shape compat check

        super(parents);
        this.value = tensor_scalar(0);
        this.interim = tensor_like(parents[0].value);
    }

    fw() {
        const prediction = this.parents[0].value;
        const target = this.parents[1].value;

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
            ops.sub(prediction.value, target.value, prediction.grad);
            ops.mul(prediction.grad, 2 / prediction.value.size, prediction.grad);
            ops.add(prediction.grad, this.grad!, prediction.grad);
        }
    }
}
