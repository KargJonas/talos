import {Tensor} from "./base/Tensor.ts";
import * as graph_ops from "./node_operations.ts";
import type {ConstScalar} from "./node_operations.ts";
import CompGraph from "./ComputationGraph.ts";
import {const_scalar_node} from "./node_factory.ts";

export default abstract class CompGraphNode {
    // State of the node
    abstract value: Tensor;
    grad?: Tensor = undefined;

    // Metadata
    readonly parents: CompGraphNode[];
    readonly children: CompGraphNode[];

    fw() {}
    bw() {}

    constructor(parents: CompGraphNode[]) {
        this.parents = parents;
        this.children = [];

        // value is initialized in extending classes
    }

    zero_grad = () => this.grad?.zeros();

    print = () => this.value.print();
    print_info = () => this.value.print_info();

    print_grad = () => {
        if (this.grad) this.grad.print();
        else console.log("Tensor has no gradient to print.");
    };

    print_grad_info = () => {
        if (this.grad) this.grad.print_info();
        else console.log("Tensor has no gradient info to print.");
    };

    private create_binary_op<T extends CompGraphNode>(op_class: new (parents: CompGraphNode[], requires_grad: boolean) => T) {
        return (_other: CompGraphNode | number) => {

            // If _other is a scalar, create a tensor that holds the scalar value such that it can be referenced in the graph
            const other: CompGraphNode = typeof _other === "number" ? const_scalar_node(_other)  : _other;
            const parents: CompGraphNode[] = [this, other];
            const new_node: CompGraphNode = new op_class(parents, true);

            // Register the new node as a child of its parents. This is necessary because
            // we will need access to each node's children during graph acquisition.
            this.children.push(new_node);
            other.children.push(new_node);

            return new_node;
        };
    }

    // common node operations
    add = this.create_binary_op(graph_ops.Add);
    mse_loss = this.create_binary_op(graph_ops.MseLoss);

    // Find all nodes that are directly or transitively connected to this node using DFS
    // i.e. find the set of all nodes in this graph
    private get_graph_nodes(node_set = new Set<CompGraphNode>()) {
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
        const all_nodes: CompGraphNode[] = [...this.get_graph_nodes()];
        const inputs: CompGraphNode[] = [];
        const outputs: CompGraphNode[] = [];

        for (const node of all_nodes) {
            if (node.children.length === 0) outputs.push(node);
            if (node.parents.length === 0) inputs.push(node);
        }

        return new CompGraph(inputs, outputs, all_nodes);
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