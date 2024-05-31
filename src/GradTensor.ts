import {Tensor, tensor_like} from "./base/Tensor.ts";
import * as graph_ops from "./graph_ops.ts";

export default class Node {
    // State of the node
    readonly primal: Tensor;
    readonly grad: Tensor;

    // Function will compute the current value of the primal from the values of the parents a and b
    readonly fw: graph_ops.ForwardFunc; // forward pass

    // This function will update the gradients of the parent nodes
    readonly bw: graph_ops.BackwardFunc; // backward pass

    // Metadata
    readonly parents: Node[];
    readonly children: Node[];
    readonly operation: string;

    constructor(value: Tensor, operation = "init", fw: graph_ops.ForwardFunc, bw: graph_ops.BackwardFunc, parents: Node[]) {
        this.primal = value;
        this.parents = parents;
        this.operation = operation;
        this.children = [];

        // todo: the following is only necessary when requires_grad is true

        this.fw = fw;
        this.bw = bw;

        this.grad = tensor_like(this.primal).zeros();
    }

    zero_grad = () => this.grad.zeros();
    register_child = (child: Node) => this.children.push(child);

    add(other: Node): Node {
        const new_node = new Node(
            tensor_like(this.primal),   // Where the actual tensor data will be stored (a grad tensor of the same shape will be allocated automatically)
            "add",             // Defines the origin of the tensor. This will be used in the backward() function.
            graph_ops.add.fw, graph_ops.add.bw,
            [this, other]
        );

        // Register the new node as a child of its parents. This is necessary because
        // we will need access to each node's children during the forward pass.
        this.register_child(new_node);
        other.register_child(new_node);

        return new_node;
    }

    forward() {
        // todo: if the node is an input, then then fw should either not be called at all
        //       or it could maybe load new data from the preprocessor
        //       but that should be done by the user by passing in a special fw function

        this.fw(this.parents, this.primal);
        this.children.map(child => child.forward());
    }

    backward(): void {
    // backward(grad: Tensor): void {
        // todo: something about grad accumulation here is wrong (redundant addition i think)

        // add incoming gradient to this gradient
        // this.grad.add(grad);

        // we have reached a node with no dependencies
        // this is where backprop stops for this branch
        if (!this.parents) return;

        this.bw(this.parents, this.grad);
        this.parents.forEach(parent => parent.backward());
    }
}


// sub(other: Node): Node {
//     const backwardFunc = () => {
//         this.grad += 1;
//         other.grad -= 1;
//     };
//     return new Node(this.value - other.value, [this, other], OpType.SUB, backwardFunc);
// }
//
// mul(other: Node): Node {
//     const backwardFunc = () => {
//         this.grad += other.value;
//         other.grad += this.value;
//     };
//     return new Node(this.value * other.value, [this, other], OpType.MUL, backwardFunc);
// }
//
// div(other: Node): Node {
//     const backwardFunc = () => {
//         this.grad += 1 / other.value;
//         other.grad -= this.value / (other.value * other.value);
//     };
//     return new Node(this.value / other.value, [this, other], OpType.DIV, backwardFunc);
// }
//
// pow(exponent: number): Node {
//     const backwardFunc = (incoming_grad: number) => {
//         this.grad += incoming_grad * exponent * Math.pow(this.value, exponent - 1)
//     }
//     return new Node(Math.pow(this.value, exponent), [this], OpType.POW, backwardFunc);
// }