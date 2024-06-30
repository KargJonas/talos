/**
 * This file is used for validation and debugging during development. 
 */

import {Tensor, core_ready, tensor} from "../index";
import {parameter_node, source_node} from "../src/node_factory.ts";
import {add, mul_acc, sub} from "../src/base/tensor_operations.ts";
import { get_total_allocated } from "../src/base/Management.ts";
import { tensor_scalar } from "../src/base/Tensor.ts";

// if your runtime does not support top-level await,
// you'll have to use core_ready.then(() => { ... }) instead
await core_ready;

console.log("\nRunning SGD demo...\n");

// Input and target tensors
const weight = parameter_node(tensor([3]).rand(), true);
const bias = parameter_node(tensor([3]).rand(), true);
const input = tensor([3]).rand();  // random but constant "input data"
const target = tensor([3]).rand(); // random but constant target/label

// const weight = parameter_node(tensor([3], [6, 2, 8]), true);
// const bias = parameter_node(tensor([3], [0, 0, 0]), true);
// const input = tensor([3], [2, -3, 9]);  // random but constant "input data"
// const target = tensor([3], [23, 2, -3]); // random but constant target/label

// Create a source node for input
const a = source_node([3], () => input);

// Modify the computation graph to include the weight
const nn = a.mul(weight).add(bias).mse_loss(target);

// this is identical to mse_loss
// const nn = a.mul(weight).sub(target).pow(2).mean();

const graph = nn.get_computation_graph();

// Define learning rate
const learningRate = .01;

console.time();

// Training loop
for (let epoch = 0; epoch < 30; epoch++) {
    graph.zero_grad();
    graph.forward();
    graph.backward();

    // Print loss value
    console.log(`\x1b[35m[${get_total_allocated()} Bytes]\x1b[0m Epoch ${epoch + 1}: Loss = ${graph.outputs[0].value.toString()}`);

    // nn.grad?.print(10);

    // Update weights using SGD
    mul_acc(weight.grad!, -learningRate, weight.value);
    mul_acc(bias.grad!, -learningRate, bias.value);

    // print_memory_status();
}

console.write("\nTraining completed in: ");
console.timeEnd();
console.log(`  Weight value: ${weight.value.toString()}`);
console.log(`  Weight grad:  ${weight.grad!.toString()}`);


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
