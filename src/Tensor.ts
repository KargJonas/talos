import { RawTensor } from "./base/RawTensor.ts";
import { tensor_scalar } from "./tensor_factory.ts";
import * as graph_ops from "./node_operations.ts";
import Graph from "./Graph.ts";
import ITensor from "./base/ITensor.ts";

type OperationClass<T> = new (parents: Tensor[], ...params: any[]) => T;

export default abstract class Tensor implements ITensor<Tensor> {
    // State of the node
    abstract value: RawTensor;
    grad?: RawTensor = undefined;

    // Metadata
    readonly parents: Tensor[];
    readonly children: Tensor[];

    private cached_graph: Graph | undefined;

    protected constructor(parents: Tensor[]) {
        this.parents = parents;
        this.children = [];

        // value is initialized in extending classes
    }

    public get rank()           { return this.value.rank; }
    public get nelem()          { return this.value.nelem; }
    public get size()           { return this.value.size; }
    public get rows()           { return this.value.get_axis_size(this.rank - 2); }
    public get cols()           { return this.value.get_axis_size(this.rank - 1); }
    public get_axis_size = (axis_index: number) => this.value.get_axis_size(axis_index);

    fw() {} // forward
    bw() {} // backward

    public zero_grad() {
        this.grad?.zeros(); // todo: should we throw?
        return this;
    }

    public rand(min = -1, max = 1) {
        this.value.rand(min, max);
        return this;
    }

    public rand_int(min = -1, max = 1) {
        this.value.rand_int(min, max);
        return this;
    }

    public fill(value: number) {
        this.value.fill(value);
        return this;
    }

    public zeros = () => this.fill(0);
    public ones = () => this.fill(1);

    print = (precision?: number) => this.value.print(precision);
    print_info = () => this.value.print_info();

    // binary operations
    add = this.create_binary_op(graph_ops.Add);
    sub = this.create_binary_op(graph_ops.Sub);
    mul = this.create_binary_op(graph_ops.Mul);
    div = this.create_binary_op(graph_ops.Div);
    pow = this.create_binary_op(graph_ops.Pow);

    matmul = this.create_binary_op(graph_ops.Matmul);
    dot = this.create_binary_op(graph_ops.Dot);

    transpose = this.create_unary_op(graph_ops.Transpose);
    public get T() { return this.transpose(); }

    // unary operations
    relu = this.create_unary_op(graph_ops.Relu);
    binstep = this.create_unary_op(graph_ops.Binstep);
    logistic = this.create_unary_op(graph_ops.Logistic);
    negate = this.create_unary_op(graph_ops.Negate);
    sin = this.create_unary_op(graph_ops.Sin);
    cos = this.create_unary_op(graph_ops.Cos);
    tan = this.create_unary_op(graph_ops.Tan);
    asin = this.create_unary_op(graph_ops.Asin);
    acos = this.create_unary_op(graph_ops.Acos);
    atan = this.create_unary_op(graph_ops.Atan);
    sinh = this.create_unary_op(graph_ops.Sinh);
    cosh = this.create_unary_op(graph_ops.Cosh);
    tanh = this.create_unary_op(graph_ops.Tanh);
    exp = this.create_unary_op(graph_ops.Exp);
    log = this.create_unary_op(graph_ops.Log);
    log10 = this.create_unary_op(graph_ops.Log10);
    log2 = this.create_unary_op(graph_ops.Log2);
    invsqrt = this.create_unary_op(graph_ops.Invsqrt);
    sqrt = this.create_unary_op(graph_ops.Sqrt);
    ceil = this.create_unary_op(graph_ops.Ceil);
    floor = this.create_unary_op(graph_ops.Floor);
    abs = this.create_unary_op(graph_ops.Abs);
    reciprocal = this.create_unary_op(graph_ops.Reciprocal);

    // reduce operations
    min = this.create_unary_op(graph_ops.Min);
    max = this.create_unary_op(graph_ops.Max);
    sum = this.create_unary_op(graph_ops.Sum);
    mean = this.create_unary_op(graph_ops.Mean);
    mse_loss = this.create_binary_op(graph_ops.MseLoss);

    // Find all nodes that are directly or transitively connected to this node using DFS
    // i.e. find the set of all nodes in this graph
    private get_graph_nodes(node_set = new Set<Tensor>()) {
        if (node_set.has(this)) return node_set; // cycle detected

        node_set.add(this);

        // note: if you wanted a complete graph that includes all paths
        //       to all outputs, then you could use [...this.parents, ...this.children]
        for (const neighbor of this.parents) {
            if (neighbor !== this) {
                neighbor.get_graph_nodes(node_set);
            }
        }

        return node_set;
    }

    realize(): Tensor {
        this.graph.forward();
        return this;
    }

    /**
     * Finds the computation graph that this node belongs to
     * @returns A computation graph
     */
    get graph(): Graph {
        if (this.cached_graph) return this.cached_graph;

        const all_nodes: Tensor[] = [...this.get_graph_nodes()];
        const inputs: Tensor[] = [];
        const outputs: Tensor[] = [];

        for (const node of all_nodes) {
            if (node.children.length === 0) outputs.push(node);
            if (node.parents.length === 0) inputs.push(node);
        }

        this.cached_graph = new Graph(inputs, outputs, all_nodes);
        return this.cached_graph;
    }

    private create_binary_op<T extends Tensor>(op_class: OperationClass<T>) {
        return (_other: Tensor | number, requires_grad = true, ...params: any[]) => {

            // If _other is a scalar, create a tensor that holds the scalar value such that it can be referenced in the graph
            const other: Tensor = _other instanceof Tensor ? _other : tensor_scalar(_other, requires_grad);
            const parents: Tensor[] = [this, other];
            const new_node: Tensor = new op_class(parents, ...params);

            // Register the new node as a child of its parents. This is necessary because
            // we will need access to each node's children during graph acquisition.
            this.children.push(new_node);
            other.children.push(new_node);

            return new_node;
        };
    }

    private create_unary_op<T extends Tensor>(op_class: OperationClass<T>) {
        return (...params: any[]) => {

            // If _other is a scalar, create a tensor that holds the scalar value such that it can be referenced in the graph
            const parents: Tensor[] = [this];
            const new_node: Tensor = new op_class(parents, ...params);

            // Register the new node as a child of its parents. This is necessary because
            // we will need access to each node's children during graph acquisition.
            this.children.push(new_node);

            return new_node;
        };
    }
}

/**
 * todo:
 *   requires_grad in its current form wont work properly..
 *   even if requires_grad=true, we will still need to compute the grads for all
 *   nodes, but we dont need to store them. this is a bit difficult to realize
 *   in the current setup...
 */
