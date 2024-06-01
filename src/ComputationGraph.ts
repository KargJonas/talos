import Node from "./Node.ts";

/**
 * This is a basic implementation of the computation graph.
 *
 * To perform forwards and backwards passes, we need to first know
 * in which order each of the graph operations need to be executed.
 * To find this order, topological sorting is used.
 *
 * For topological sorting, it is also necessary to know the inputs
 * and outputs of the graph. We find all nodes of the graph by
 * considering the graph to be non-directed and applying depth-first
 * search. We then filter by the number of children/parents.
 */
export default class CompGraph {
    inputs: Node[];
    outputs: Node[];
    all_nodes: Node[];

    topological_ordering: Node[];

    constructor(inputs: Node[], outputs: Node[], all_nodes: Node[]) {
        this.inputs = inputs;
        this.outputs = outputs;
        this.all_nodes = all_nodes;
        this.topological_ordering = this.find_topological_order();
    }

    // todo: this is where parallelization could come into play
    //       for parallelization, there are some optimization in the way we find the
    //       topological ordering. e.g. if there are three input nodes, they are all
    //       "on the same level", even though maybe it would be wise to first execute
    //       those where the result deadline is closer. (where the result will be needed
    //       earlier)
    // todo: handle cycles. topological orderings exist iff the graph is acyclic

    /**
     * uses kahn's algorithm to find a topological ordering / a correct execution sequence
     * NOTE: NOT THREAD SAFE BECAUSE OF DEGREE MODIFICATION
     * @returns An array that represents the topological ordering or the graph execution.
     *          The first op that should be performed is in index 0, and the last is in the last index.
     */
    find_topological_order(): Node[] {
        const queue: Node[] = [...this.inputs];
        const topological_order: Node[] = [];
        const in_degrees = new Map<Node, number>();

        for (const node of this.all_nodes) {
            in_degrees.set(node, node.parents.length);
        }

        while (queue.length > 0) {
            const node = queue.pop()!;
            const directed_neighbors = node!.children;
            topological_order.push(node);

            for (const neighbor of directed_neighbors) {
                const updated_degree = in_degrees.get(neighbor)! - 1;
                in_degrees.set(neighbor, updated_degree);
                if (updated_degree === 0) queue.push(neighbor);
            }
        }

        return topological_order;
    }

    forward() {
        // Step forward through node execution order and update primals using forward functions
        for (let i = 0; i < this.topological_ordering.length; i++) {
            const node = this.topological_ordering[i];
            node.fw(node.parents, node);
        }
    }

    backward(): void {
        // Step backward through node execution order and update grads using backward functions
        for (let i = this.topological_ordering.length - 1; i >= 1; i--) {
            const node = this.topological_ordering[i];
            node.bw(node.parents, node);
        }
    }
}