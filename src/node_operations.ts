import {Tensor} from "./base/Tensor.ts";
import Node from "./Node.ts";
import * as ops from "./base/tensor_operations.ts";

// This file contains all operations of the graph-node abstraction-level
// These are essentially all operations of the tensor level plus their derivatives

export type ForwardFunc = (parents: Node[], self: Node) => void;
export type BackwardFunc = (parents: Node[], self: Node) => void;

export interface BidirectionalOperation {
    name: string,
    fw: ForwardFunc;
    bw: BackwardFunc;
}

export const add: BidirectionalOperation = {
    name: "add",

    fw(parents: Node[], self: Node) {
        ops.add(parents[0].primal, parents[1].primal, self.primal);
    },

    bw(parents: Node[], self: Node) {
        // todo: validate if this works as expected (because of the changes to the tensor_ops + Tensor class)
        ops.add(parents[0].grad, self.grad, parents[0].grad); // in-place op
        ops.add(parents[1].grad, self.grad, parents[1].grad); // in-place op
        // parents[0].grad.add(self.grad, true);
        // parents[1].grad.add(self.grad, true);
    }
};

export const nop: BidirectionalOperation = {
    fw(parents: Node[], primal: Node) {},
    bw(parents: Node[], primal: Node) {}
};
