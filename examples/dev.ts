/**
 * This file is used for validation and debugging during development. 
 */

import {core, core_ready, tensor, tensor_like} from "../index";
import {parameter_node, source_node} from "../src/node_factory.ts";
import {tensor_scalar} from "../src/base/Tensor.ts";
import {add, add_acc, mul, mul_acc, sin, sin_acc, sub} from "../src/base/tensor_operations.ts";

// if your runtime does not support top-level await,
// you'll have to use core_ready.then(() => { ... }) instead
await core_ready;

console.log("###########\n".repeat(2));

// Define a weight tensor

// t0 is debroadcasted to [5, 7, 9]
// then [1, 2, 3] is added which results in [6, 9, 12]
// this is then stored in t2

// if t0 is an incoming gradient and t1 is the current node's
// gradient then we can now efficiently accumulate this gradient
// without any additional interims

// const t0 = tensor([2, 3], [1, 2, 3, 4, 5, 6]);
// const t1 = tensor([3], [1, 2, 3]);
// const t2 = tensor_like(t1).zeros();
//
// add_acc
// t2.print();

// const a = tensor_scalar(5);
// const b = tensor([3], [1, 2, 3]);
//
// const input = source_node([3], () => b);
//
// const nn = input.add(a);
// const graph = nn.get_computation_graph();
//
// graph.forward();
// graph.backward();

// const t = tensor([3], [1,2,3]);
// const t1 = tensor([1], [3]);
// t1.add(t).print(); // this should yield [4, 5, 6] but it yields [4, 0, 0]

// // Input and target tensors
// const weight = parameter_node(tensor([3], [1, 2, 3]), true);
// const input = tensor([3], [1, 2, 3]);
// const target = tensor([3], [1, 2, 3]);
//
// // Create a source node for input
// const a = source_node([3], () => input);
//
// // Modify the computation graph to include the weight
// // const nn = a.mul(weight).sub(target).pow(2).mean();
// const nn = a.mul(weight).sum();
//
// const graph = nn.get_computation_graph();
//
// // Define learning rate
// const learningRate = 0.01;
//
// const interim = tensor_like(weight.value);
//
// // Training loop
// for (let epoch = 0; epoch < 3; epoch++) {
//     graph.forward();
//     graph.backward();
//
//     // Print loss value
//     console.log(`Epoch ${epoch + 1}: Loss = ${graph.outputs[0].value.toString()}`);
//
//     // Update weights using SGD
//     mul(weight.grad!, learningRate, interim);
//     sub(weight.value, interim, weight.value);
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
