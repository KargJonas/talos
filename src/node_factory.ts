import Shape from "./base/Shape.ts";
import {Tensor} from "./base/Tensor.ts";
import {Const, Input} from "./node_operations.ts";

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
export function source_node(shape: Shape | number[], producer: () => Tensor): Input {
    return new Input(shape, producer);
}

export function const_node(scalar: Tensor | number): Const {
    return new Const(scalar);
}
