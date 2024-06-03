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

// TODO: what about scalar addition?     THIS IS ON TOP OF THE IMPORTANCE LIST PROBABLY
//       one solution is to add support for scalar tensors      !!!!!!!!!!!!!!!!!
//       scalar tensors can be broadcast to any shape           !!!!!!!!!!!!!!!!!
//       this way, the separate scalar binary-ops logic could be removed

// Pairwise tensor addition
export const add: BidirectionalOperation = {
    name: "add",

    fw(parents: Node[], self: Node) {
        //      src 1               src 2             dest
        ops.add(parents[0].primal, parents[1].primal, self.primal);
    },

    bw(parents: Node[], self: Node) {
        // todo: ensure that the if's are actually what we need here
        //      src 1            src 2      dest
        if (parents[0]?.requires_grad) ops.add(parents[0].grad, self.grad, parents[0].grad); // in-place op
        if (parents[1]?.requires_grad) ops.add(parents[1].grad, self.grad, parents[1].grad); // in-place op
    }
};

// Does nothing.
export const nop: BidirectionalOperation = {
    name: "nop",

    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    fw(parents: Node[], primal: Node) {},

    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    bw(parents: Node[], primal: Node) {}
};
