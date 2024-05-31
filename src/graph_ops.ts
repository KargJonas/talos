import {Tensor} from "./base/Tensor.ts";
import Node from "./GradTensor.ts";
import * as ops from "./base/tensor_operations.ts";

// This file contains all operations of the graph-node abstraction-level
// These are essentially all operations of the tensor level plus their derivatives

export type ForwardFunc = (parents: Node[], primal: Tensor) => void;
export type BackwardFunc = (parents: Node[], incoming_grad: Tensor) => void;

interface BidirectionalOperation {
    fw: ForwardFunc;
    bw: BackwardFunc;
}

// Note: FW and BW are procedures
//   They update the values of the primals and gradients in-place as a side effect

export const add: BidirectionalOperation = {
    fw(parents: Node[], primal: Tensor) {
        //  src0        src1        dest
        ops.add(parents[0].primal, parents[1].primal, primal);
    },

    bw(parents: Node[], incoming_grad: Tensor) {
        // todo: validate if this works as expected (because of the changes to the tensor_ops + Tensor class)
        parents[0].grad.add(incoming_grad, true);
        parents[1].grad.add(incoming_grad, true);
    }
};

export const nop: BidirectionalOperation = {
    fw(parents: Node[], primal: Tensor) {},
    bw(parents: Node[], primal: Tensor) {}
}
