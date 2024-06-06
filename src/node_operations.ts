import Node from "./Node.ts";
import * as ops from "./base/tensor_operations.ts";
import {Tensor, tensor_scalar} from "./base/Tensor.ts";
import {tensor} from "../index.ts";

// This file contains all operations of the graph-node abstraction-level
// These are essentially all operations of the tensor level plus their derivatives

export type NodeInitFunc = (parents: Node[], self: Node) => Tensor;
export type ForwardFunc = (parents: Node[], self: Node) => void;
export type BackwardFunc = (parents: Node[], self: Node) => void;

export interface BidirectionalOperation {
    name: string,
    init: NodeInitFunc;
    fw: ForwardFunc;
    bw: BackwardFunc;
}

// Pairwise tensor addition
export const add: BidirectionalOperation = {
    name: "add",

    init(parents: Node[], self: Node): Tensor {
        return tensor(parents[0].primal.shape.broadcast(parents[1].primal.shape));
    },

    fw(parents: Node[], self: Node) {
        ops.add(parents[0].primal, parents[1].primal, self.primal);
    },

    bw(parents: Node[], self: Node) {
        // todo: ensure that the if's are actually what we need here
        if (parents[0].grad) ops.add(parents[0].grad, self.grad, parents[0].grad); // in-place op
        if (parents[1].grad) ops.add(parents[1].grad, self.grad, parents[1].grad); // in-place op
    }
};

export const mse_loss: BidirectionalOperation = {
    name: "mse_loss",

    init(parents: Node[], self: Node): Tensor {
        return tensor_scalar(0);
    },

    fw(parents: Node[], self: Node) {
        const prediction = parents[0].primal;
        const target = parents[1].primal; // other is always in parents[1]

        // compute MSE loss: (prediction - target)^2 / N
        ops.sub(prediction, target, self.primal);
        ops.pow(self.primal, 2, self.primal);
        const mean_val = ops.mean(self.primal);
        self.primal.fill(mean_val);
    },

    bw(parents: Node[], self: Node) {
        const prediction = parents[0];
        const target = parents[1];

        // gradient of MSE loss w.r.t. prediction: 2 * (prediction - target) / N

        // if (prediction.requires_grad) {
        if (prediction.grad) {
            ops.sub(prediction.primal, target.primal, prediction.grad!);
            ops.mul(prediction.grad, 2 / prediction.primal.size, prediction.grad!);
            ops.add(prediction.grad, self.grad!, prediction.grad!);
        }
    }
};
