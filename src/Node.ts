import tensor, {Tensor, tensor_like} from "./base/Tensor.ts";
import * as graph_ops from "./node_operations.ts";
import Shape from "./base/Shape.ts";
import {BidirectionalOperation} from "./node_operations.ts";
import CompGraph from "./ComputationGraph.ts";

export default class Node {
    // State of the node
    primal: Tensor;
    readonly grad: Tensor;

    // Function will compute the current value of the primal from the values of the parents a and b
    readonly fw: graph_ops.ForwardFunc; // forward pass

    // This function will update the gradients of the parent nodes
    readonly bw: graph_ops.BackwardFunc; // backward pass

    // Metadata
    readonly parents: Node[];
    readonly children: Node[];
    readonly operation: string;

    constructor(value: Tensor, operation: string, fw: graph_ops.ForwardFunc, bw: graph_ops.BackwardFunc, parents: Node[]) {
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

    create_binary_op(graph_op: BidirectionalOperation) {
        return (other: Node) => {
            const new_node = new Node(
                tensor_like(this.primal),   // Where the actual tensor data will be stored (a grad tensor of the same shape will be allocated automatically)
                graph_op.name,                       // Op name for debugging purposes
                graph_op.fw, graph_op.bw,
                [this, other]
            );

            // Register the new node as a child of its parents. This is necessary because
            // we will need access to each node's children during graph acquisition.
            this.children.push(new_node);
            other.children.push(new_node);

            return new_node;
        };
    }

    add = this.create_binary_op(graph_ops.add);

    // Find all nodes that are directly or transitively connected to this node using DFS
    // i.e. find the set of all nodes in this graph
    private get_graph_nodes(node_set = new Set<Node>()) {
        if (node_set.has(this)) return node_set; // cycle detected

        node_set.add(this);
        const neighbors = [...this.parents, ...this.children];

        for (const neighbor of neighbors) {
            if (neighbor !== this) {
                neighbor.get_graph_nodes(node_set);
            }
        }

        return node_set;
    }

    /**
     * Finds the computation graph that this node belongs to
     * @returns A computation graph
     */
    get_computation_graph(): CompGraph {
        const all_nodes: Node[] = [...this.get_graph_nodes()];
        const inputs: Node[] = [];
        const outputs: Node[] = [];

        for (const node of all_nodes) {
            if (node.children.length === 0) outputs.push(node);
            if (node.parents.length === 0) inputs.push(node);
        }

        return new CompGraph(inputs, outputs, all_nodes);
    }
}

// creates an input node
// this node has no parents (but can have children),
// it has a NOP (no operation) as forward and backward function,
// meaning that it will never alter the state of the model
// (neither forward-, nor backward passes)
export function node(shape: number[] | Shape, producer: () => Tensor) {
    return new Node(
        tensor(shape),
        "input",
        // TODO: validate. does changing the tensor reference of the primal break things?
        //       do we need to consider de-allocation? (i think no)
        (_parents: Node[], self: Node) => self.primal = producer(),
        graph_ops.nop.bw,
        []
    );
}

// creates an input node with a constant value
export function const_node(shape: number[] | Shape, data?: number[]) {
    return new Node(
        tensor(shape, data),
        "input",
        graph_ops.nop.fw,
        graph_ops.nop.bw,
        []
    );
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