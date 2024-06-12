/**
 * This file is used for validation and debugging during development. 
 */

import {core_ready, tensor} from "../index";
import {source_node} from "../src/node_factory.ts";

// if your runtime does not support top-level await,
// you'll have to use core_ready.then(() => { ... }) instead
await core_ready;

console.log("###########\n".repeat(2));

const input = tensor([3], [1, 2, 3]);
const target = tensor([3], [1, 2, 3]);

// node "a" always receives the same input for the sake of demonstration
const a  = source_node([3], () => input);

// the last node of this primitive network is the mean squared error loss
// the value of this node is a scalar tensor
// const nn = a.mul(3).mse_loss(source_node([3], () => target));
const nn = a.mul(3).sub(target).pow(2).mean();

const graph = nn.get_computation_graph();
graph.forward();
graph.backward();

graph.outputs[0].print();
a.print_grad();

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
