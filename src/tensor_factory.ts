import { Constant, Parameter, Source } from "./autograd/node_operations.ts";
import { RawTensor } from "./raw_tensor/raw_tensor.ts";
import Shape from "./raw_tensor/shape.ts";
import {NDArray, flatten} from "./raw_tensor/util.ts";
import Tensor from "./tensor.ts";

/**
 * Creates an input node.
 *
 * The created node has no parents, and has NOP operations as forward and backward functions,
 * meaning that it will never alter the state of the model.
 *
 * @param shape Shape of the tensors this source node will produce
 * @param producer Function that returns a tensor each time it is called.
 *  These tensors should have the same shape as specified by the shape parameter.
 */
export function tensor_producer(shape: Shape | number[], producer: () => (NDArray | RawTensor | Tensor | number)): Source {
    // todo: fix, this is not memory safe because we allocate memory
    return new Source(shape, () => {
        const item = producer();
        if (item instanceof Array) return RawTensor.from_array(item);
        if (item instanceof RawTensor) return item;
        if (item instanceof Tensor) return item.value;

        throw new Error("Producer returned an element that is not Tensor, RawTensor or Array.");
    });
}

// export function tensor_scalar(): Parameter;
// export function tensor_scalar(value?: number): Parameter;
// export function tensor_scalar(requires_grad?: boolean): Parameter;
export function tensor_scalar(arg_1?: number | boolean, arg_2?: boolean): Parameter {
    // only value provided
    if (arg_1 === undefined) {
        return new Constant(RawTensor.scalar());
    }

    if (typeof arg_1 === "boolean" && arg_2 === undefined) {
        if (arg_1) return new Parameter(RawTensor.scalar());
        return new Constant(RawTensor.scalar());
    }

    if (typeof arg_1 === "number") {
        if (arg_2) return new Parameter(RawTensor.scalar(arg_1));
        return new Constant(RawTensor.scalar(arg_1));
    }

    throw new Error("Cant create scalar tensor with the provided arguments.");
}

// Overload for shape and requires_grad
export function tensor(shape: number[], is_parameter?: boolean): Parameter;
export function tensor(shape: number[], data: number[], is_parameter?: boolean): Parameter;
export function tensor(shape: number[], arg_1?: number[] | boolean, arg_2?: boolean) {
    // only shape provided
    if (arg_1 === undefined && arg_2 === undefined) {
        return new Constant(RawTensor.create(shape));
    }
    
    // shape and requires_grad provided
    if (typeof arg_1 === "boolean" && arg_2 === undefined) {
        if (arg_1) return new Parameter(RawTensor.create(shape));
        return new Constant(RawTensor.create(shape));
    }

    // shape and data provided (and maybe requires_grad)
    if (arg_1 instanceof Array) {
        if (arg_2) return new Parameter(RawTensor.create(shape, arg_1));
        return new Constant(RawTensor.create(shape, arg_1));
    }

    throw new Error("Cant create tensor with the provided arguments.");
}

export function tensor_from_array(_data: NDArray, is_parameter: boolean = true): Parameter {
    const [shape, data] = flatten(_data);
    return tensor(shape, data, is_parameter);
}
