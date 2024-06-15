/**
 * This file is used for validation and debugging during development. 
 */

import {core_ready, tensor, tensor_like} from "../index";
import {source_node} from "../src/node_factory.ts";
import {tensor_scalar} from "../src/base/Tensor.ts";
import {debroadcast, grad_acc} from "../src/base/tensor_operations.ts";

// if your runtime does not support top-level await,
// you'll have to use core_ready.then(() => { ... }) instead
await core_ready;

console.log("###########\n".repeat(2));

/**
 * TODO:
 *   inputs != inputs
 *
 *   we need to distinguish a bit.
 *   currently all tensor or scalar values passed to node operations
 *   are treated as constants. this means they do not have gradients.
 *   this is an issue because it might be nice to be able to use
 *   such values as parameters.
 *
 *
 *   YOU LEFT OFF HERE:
 *      you implemented part of the above thing.
 *      now every tensor or scalar from the outside will receive a
 *      gradient by default.
 *      the issue: when we use scalar values, the gradient will also
 *                 be scalar so when we accumulate the gradient from
 *                 the children, we might get non-scalar gradients
 *                 when we try to add these to the scalar we of course
 *                 get a broadcasting exception.
 *                 possible solution: introduce an operation that
 *                   automatically performs gradient accumulation
 *                   NOTE: for scalars, i think it the way to go
 *                         is to sum the components of the incoming
 *                         gradient and add that to the scalar
 */

// Define a weight tensor

const t0 = tensor([2, 2, 3], [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]);
const t1 = tensor([3], [0, 0, 0]);
const broadcasted_tensor = t0.add(t1); // broadcast result: tensor of shape [2, 3]

broadcasted_tensor.print();

// this should have the same shape as the corresponding parent (in this case t1)
const debroadcasted_tensor = tensor_like(t1);

debroadcast(broadcasted_tensor, debroadcasted_tensor);
debroadcasted_tensor.print();

// produces [21, 0, 0]

// const a = tensor([3], [1, 2, 3]); // this is the tensor that we take the actual data from
// const b = tensor_scalar(1);            // this is the tensor that we only take the shape from
// const res = tensor([3]);               // this is the tensor that will hold the sum
// debroadcast(a, b, res);
//
// res.print(); // produces [6, 0, 0]


// // Input and target tensors
// const weight = tensor([3], [1, 2, 3]);
// const input = tensor([3], [1, 2, 3]);
// const target = tensor([3], [1, 2, 3]);
//
// // Create a source node for input
// const a = source_node([3], () => input);
//
// // Modify the computation graph to include the weight
// const nn = a.mul(weight).sub(target).pow(2).mean();
//
// const graph = nn.get_computation_graph();
//
// // Define learning rate
// const learningRate = 0.01;
//
// // Training loop
// for (let epoch = 0; epoch < 100; epoch++) {
//     graph.forward();
//     graph.backward();
//
//     // Print loss value
//     console.log(`Epoch ${epoch + 1}: Loss = ${graph.outputs[0].value}`);
//
//     // Update weights using SGD
//     for (let i = 0; i < weight.value.length; i++) {
//         weight.value[i] -= learningRate * weight.grad[i];
//     }
// }
//
// // Print final weight values and their gradients
// weight.print();
// weight.print_grad();



// const input = tensor([3], [1, 2, 3]);
// const target = tensor([3], [1, 2, 3]);
//
// // node "a" always receives the same input for the sake of demonstration
// const a  = source_node([3], () => input);
//
// // the last node of this primitive network is the mean squared error loss
// // the value of this node is a scalar tensor
// // const nn = a.mul(3).mse_loss(source_node([3], () => target));
// const nn = a.mul(3).sub(target).pow(2).mean();
//
// const graph = nn.get_computation_graph();
// graph.forward();
// graph.backward();
//
// graph.outputs[0].print();
// graph.outputs[0].print_grad();



// const dataset_x_0: Tensor = tensor([50, 4, 4]).zeros();    // 50 "images" of size 4x4
// const dataset_x_1: Tensor = tensor([50, 4, 4]).zeros();    // additional information for each image
// const dataset_y: Tensor   = tensor([50, 10]);             // one hot encoding of 10 classes
//
// function get_provider_from_dataset(dataset: Tensor): () => Tensor {
//     const iterator = dataset.get_axis_iterable(0);
//
//     // todo: destruct tensors produced by iterator after use
//
//     // Data provider
//     return function () {
//         const next = iterator.next();
//         if (next.done) throw new Error("Reached end of dataset.");
//         return next.value;
//     };
// }
//
// const input_0 = node([4, 4], get_provider_from_dataset(dataset_x_0));
// const input_1 = node([4, 4], get_provider_from_dataset(dataset_x_1));
//
// /**
//  * TODO:
//  *   NOTE TO SELF: BACKPROP/OUTPUT NODES
//  *      typically, there is only one output node on which we define a single loss function.
//  *      in the current implementation it is possible to have multiple output nodes.
//  *      i dont think this should inherently be prevented as these nodes could be used e.g. for
//  *      introspection or some other purpose.
//  *      imo we could also allow backprop from all output nodes but this cannot really be
//  *      done in a single backward pass.
//  *      (this would be called multi-objective-optimization)
//  */
//
// // Result tensor
// const result = input_0.add(1).add(input_1);
//
// // Graph acquisition
// const graph = result.get_computation_graph();
//
// graph.forward();
// graph.backward();
//
// console.log("Result: ");
// result.primal.print();
//
// console.log("Gradient of result: ");
// result.grad?.print();
