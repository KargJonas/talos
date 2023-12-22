import * as op from '../tensor/tensor_operations';
import Tensor from '../tensor/Tensor';
import { TensorOp } from '../util';

// const tnf = (fw: Function, bw: Function[]) => new CompGraphNode(fw, bw);

// class CompGraphNode {
//     fw: Function;
//     bw: Function[];

//     constructor(fw: Function, bw: Function[]) {
//         this.fw = fw;
//         this.bw = bw;
//     }
// }

// const add = tnf(ops.add, [ops.identity, ops.identity]);
// const sin = tnf(ops.sin, [ops.cos]);


export default class CompGraphNode {
    operation: TensorOp;        // The operation performed in this node (e.g., add, multiply, sin)
    inputs: Tensor[];           // The input tensors to this operation
    output: Tensor | undefined; // The output tensor of this operation
    grad_fn: Function[];        // Array of functions to compute gradients w.r.t each input tensor

    constructor(operation: TensorOp, grad_fn: TensorOp[], inputs: Tensor[]) {
        this.operation = operation;
        this.grad_fn = grad_fn;
        this.inputs = inputs;
        this.output = undefined;
    }

    // Performs the forward pass and saves the output tensor
    forward() {
        this.output = this.operation(...this.inputs);
        if (this.output !== undefined) this.output.graph_node = this; // Linking the output tensor to this graph node
    }

    // Performs the backward pass to compute gradients
    backward(grad_output: Tensor | undefined) {
        if (grad_output === undefined)
            throw new Error('Gradient is undefined.');

        // Assuming the same number of gradFn functions as input tensors
        const gradients = this.grad_fn.map((fn, idx) => fn(grad_output, ...this.inputs));
        for (let i = 0; i < this.inputs.length; i++) {
            // todo: we have to be extremely careful here!
            // tensor data continues to exist in memory until we free it.
            // we cant simply do assignments like this it is also inefficient

            // accumulate gradients
            if (this.inputs[i].grad === undefined) this.inputs[i].grad = gradients[i];
            else op.add(this.inputs[i].grad, gradients[i], true);

            // Recursive call for backpropagation
            if (this.inputs[i].graph_node) {
                this.inputs[i].graph_node?.backward(this.inputs[i].grad);
            }
        }
    }
}
