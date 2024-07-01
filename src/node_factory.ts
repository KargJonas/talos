import { Parameter, Source } from "./node_operations.ts";
import { RawTensor } from "./base/RawTensor.ts";
import Shape from "./base/Shape.ts";

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
export function tensor_producer(shape: Shape | number[], producer: () => RawTensor): Source {
    return new Source(shape, producer);
}

// export function tensor_scalar(): Parameter;
// export function tensor_scalar(value?: number): Parameter;
// export function tensor_scalar(requires_grad?: boolean): Parameter;
export function tensor_scalar(arg_1?: number | boolean, arg_2?: boolean): Parameter {
    // todo fix requires_grad (see comment below)

    // only value provided
    if (arg_1 === undefined) {
        return new Parameter(RawTensor.scalar(), false);
    }

    if (typeof arg_1 === "boolean" && arg_2 === undefined) {
        return new Parameter(RawTensor.scalar(), arg_1 || false);
    }

    if (typeof arg_1 === "number") {
        return new Parameter(RawTensor.scalar(arg_1), arg_2 || false);
    }

    throw new Error("Cant create scalar tensor with the provided arguments.");
}

// Overload for shape and requires_grad
export function tensor(shape: number[], requiresGrad?: boolean): Parameter;
export function tensor(shape: number[], data: number[], requiresGrad?: boolean): Parameter;
export function tensor(shape: number[], arg_1?: number[] | boolean, arg_2?: boolean) {
    // todo fix requires_grad (see comment below)

    // only shape provided
    if (arg_1 === undefined && arg_2 === undefined) {
        return new Parameter(RawTensor.create(shape), false);
    }
    
    // shape and requires_grad provided
    if (typeof arg_1 === "boolean" && arg_2 === undefined) {
        return new Parameter(RawTensor.create(shape), arg_1 || false);
    }    

    // shape and data provided (and maybe requires_grad)
    if (arg_1 instanceof Array) {
        return new Parameter(RawTensor.create(shape, arg_1), arg_2 || false);
    }

    throw new Error("Cant create tensor with the provided arguments.");
}
