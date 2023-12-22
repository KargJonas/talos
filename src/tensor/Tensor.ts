import core from '../core/build';
import CompGraphNode from '../graph/graph';
import { Shape } from '../Shape';
import { mat_to_string, tensor, tensor_like, TensorOp } from '../util';
import { add } from './tensor_operations';

export default class Tensor {
    public readonly shape: Shape; // [outermost axis, ..., rows, cols]
    public readonly data: Float32Array;

    public readonly rank: number;
    public readonly nrows: number;
    public readonly ncols: number;

    // reference to the node of the computation graph that created this tensor 
    graph_node?: CompGraphNode;

    // gradient of this tensor
    public grad?: Tensor;

    constructor(shape: Shape, data: Float32Array) {
        this.shape = new Shape(...shape);
        this.data = data;

        this.rank = this.shape.get_ndim();
        this.nrows = this.shape.get_rows();
        this.ncols = this.shape.get_cols();
    }

    public add_to_graph(operation: TensorOp, inputs: Tensor[], grad_fn: TensorOp[]) {
        this.graph_node = new CompGraphNode(operation, grad_fn, inputs);
    }

    public backward(grad_output: Tensor | undefined = undefined) {
        if (!this.graph_node)
            throw new Error('No computation graph node associated with this tensor');

        // if grad_output is undefined, this is the end of the graph (e.g., loss tensor)
        if (grad_output === undefined && this.grad === undefined) this.grad = tensor(this.shape).ones();
        else if (grad_output !== undefined) this.grad = grad_output;

        // reached start of graph
        if (this.graph_node.inputs.length === 0) return;

        // propagate gradients up to the input tensors
        const inputGrads = this.graph_node.backward(this.grad);
        this.graph_node.inputs.forEach((input_tensor, idx) => {
            input_tensor.backward(inputGrads[idx]);
        });
    }

    // todo: separate allocation from computation
    //   currently, when i call something like add(Tensor, Tensor),
    //   add() computes the shape of the resulting tensor, allocates
    //   memory and computes its value.
    // how it should be:
    //   get_pairwise_op_shape() caluclates the shape and does error handling
    //   

    public add(other: Tensor) {
        // result.add_to_graph(add, );
    }

    public clone(): Tensor {
        // todo: all this could be replaced by copy(this);
        //   not sure if we should use tensor ops in the tensor class though
        const new_tensor = tensor_like(this);
        core._copy(this.get_ptr(), new_tensor.get_ptr(), this.data.length);
        return new_tensor;
    }

    // data operations
    public get_ptr = () => this.data.byteOffset;
    public free = () => core._free_farr(this.get_ptr());

    // in place ops [CAREFUL]

    public rand(min = -1, max = 1) {
        core._rand_f(this.get_ptr(), this.data.length, min, max);
        return this;
    }

    public rand_int(min = -1, max = 1) {
        core._rand_i(this.get_ptr(), this.data.length, min, max);
        return this;
    }

    public fill(value: number) {
        core._fill(this.get_ptr(), this.data.length, value);
        return this;
    }

    public zeros = () => this.fill(0);
    public ones = () => this.fill(1);

    // shape operations

    // watch out! this will create a reference to the original tensor
    // no new memory will be allocated
    flatten = (n: number): Tensor => new Tensor(this.shape.flatten(n), this.data);

    *get_axis_iterable(n: number) {
        const shape = this.shape.get_axis_shape(n + 1);
        const n_elements = shape.get_nelem();

        for (const index of this.shape.get_axis_iterable(n)) {
            yield new Tensor(shape, this.data.subarray(index, index + n_elements))
        }
    }

    public get(...loc: number[]): Tensor | number {
        if (loc.length > this.rank)
            throw new Error(`Location [${loc}] is too specific for shape [${this}]`);

        const [index, shape] = this.shape.get_index(...loc);

        // return element if location describes a scalar, return subtensor if not
        if (loc.length === this.rank) return this.data[index];
        return new Tensor(shape, this.data.subarray(index, index + shape.get_nelem()));
    }

    // usability methods
    private to_str(a: Tensor, num_width = 10, space_before = 0) {
        switch (a.rank) {
            case 0: return '[]';
            case 1: return `[ ${a.data.join(', ')} ]`;
            case 2: return mat_to_string(a, num_width, space_before);
        }

        // hidim tensors
        let strings: string[] = [];
        for (const element of a.get_axis_iterable(0)) {
            strings.push(this.to_str(element, num_width, space_before + 2)!);
        }

        return `[ ${strings.join(',\n\n' + ' '.repeat(space_before + 2))} ]`;
    }

    public toString = () => this.to_str(this);
}
