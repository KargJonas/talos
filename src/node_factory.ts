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
export function source_node(shape: Shape | number[], producer: () => RawTensor): Source {
    return new Source(shape, producer);
}

export function parameter_node(value: RawTensor | number, requires_grad: boolean): Parameter {
    return new Parameter(value, requires_grad);
}
