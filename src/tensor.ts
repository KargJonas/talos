import { RawTensor } from "./raw_tensor/raw_tensor.ts";
import { tensor_scalar } from "./tensor_factory.ts";
import * as graph_ops from "./autograd/node_operations.ts";
import Graph from "./autograd/graph.ts";

// NodeOption = any additional option/parameter that can be passed into a node
// (e.g. negative slope of leaky relu)
type NodeOption = any;
type OperationClass<T> = new (parents: Tensor[], ...params: NodeOption[]) => T;

export default abstract class Tensor {
    private static id_counter: number = 0;
    public id: number = Tensor.id_counter++;

    // state of the node
    abstract value: RawTensor;
    grad?: RawTensor = undefined;

    // metadata
    readonly parents: Tensor[];
    readonly children: Tensor[];

    private cached_graph: Graph | undefined;
    name?: string;

    protected constructor(parents: Tensor[]) {
        this.parents = parents;
        this.children = [];

        // value is initialized in extending classes
    }

    get rank()  { return this.value.rank; }
    get nelem() { return this.value.nelem; }
    get size()  { return this.value.size; }
    get rows()  { return this.value.get_axis_size(this.rank - 2); }
    get cols()  { return this.value.get_axis_size(this.rank - 1); }
    get item()  { return this.value.item; }
    get T() { return this.transpose(); }
    get_axis_size = (axis_index: number) => this.value.get_axis_size(axis_index);

    fw() {} // forward
    bw() {} // backward

    private chain_op<T extends any[]>(operation: (...params: T) => void): (...params: T) => Tensor {
        return (...params: T): Tensor => {
            operation.apply(this, params);
            return this;
        };
    }

    // initialization stuff
    uniform         = this.chain_op((min = -1, max = 1, seed?: number) => this.value.rand(min, max, seed));
    normal          = this.chain_op((mean = 0, std_dev = 1, seed?: number) => this.value.normal(mean, std_dev, seed));
    kaiming_uniform = this.chain_op((n_in: number, seed?: number) => this.value.kaiming_uniform(n_in, seed));
    kaiming_normal  = this.chain_op((n_in: number, seed?: number) => this.value.kaiming_normal(n_in, seed));
    xavier_uniform  = this.chain_op((n_in: number, n_out: number, seed?: number) => this.value.xavier_uniform(n_in, n_out, seed));
    xavier_normal   = this.chain_op((n_in: number, n_out: number, seed?: number) => this.value.xavier_uniform(n_in, n_out, seed));
    fill            = this.chain_op((value: number) => this.value.fill(value));
    set_name        = this.chain_op((name: string) => this.name = name);
    zero_grad       = this.chain_op(() => this.grad?.zeros());
    realize         = this.chain_op(() => this.graph.forward());
    zeros           = () => this.fill(0);
    ones            = () => this.fill(1);

    // printing/logging stuff
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

    // unary operations
    transpose = this.create_unary_op(graph_ops.Transpose);
    dropout = this.create_unary_op(graph_ops.Dropout);
    relu = this.create_unary_op(graph_ops.Relu);
    leaky_relu = this.create_unary_op(graph_ops.LeakyRelu);
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

    // sequentially applies a list of layers by temporarily connecting the
    // input node of each layer with the output of the previous layer
    public sequential(layers: Layer[]) {
        
    }

    // Find all nodes that are directly or transitively connected to this node using DFS
    // i.e. find the set of all nodes in this graph
    private get_graph_nodes(node_set = new Set<Tensor>()) {
        if (node_set.has(this)) {
            // cycle detected
            throw new Error("Detected a cycle in the graph. Computation graphs must be acyclic.");
            return node_set;
        }

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

    /**
     * Finds the computation graph that this node belongs to
     * @returns A computation graph
     */
    get graph(): Graph {
        if (this.cached_graph) return this.cached_graph;

        const all_nodes: Tensor[] = [...this.get_graph_nodes()];
        const inputs: Tensor[] = [];
        const parameters: graph_ops.Parameter[] = [];

        for (const node of all_nodes) {
            if (node.parents.length === 0) inputs.push(node);
            if (node instanceof graph_ops.Parameter) parameters.push(node);
        }

        this.cached_graph = new Graph(inputs, this, parameters, all_nodes);
        return this.cached_graph;
    }

    private create_binary_op<T extends Tensor>(op_class: OperationClass<T>) {
        return (_other: Tensor | number, requires_grad = true, ...params: NodeOption[]) => {

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
        return (...params: NodeOption[]) => {

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
 *   nodes, but we don't need to store them. this is a bit difficult to realize
 *   in the current setup...
 */
