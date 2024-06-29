/**
 * This file is used for validation and debugging during development. 
 */

import {core_ready, tensor} from "../index";
import {parameter_node, source_node} from "../src/node_factory.ts";
import {mul_acc} from "../src/base/tensor_operations.ts";

// if your runtime does not support top-level await,
// you'll have to use core_ready.then(() => { ... }) instead
await core_ready;

console.log("\nRunning SGD demo...\n");

// Input and target tensors
const weight = parameter_node(tensor([30]).rand(), true);
const bias = parameter_node(tensor([30]).rand(), true);
const input = tensor([30]).rand();  // random but constant "input data"
const target = tensor([30]).rand(); // random but constant target/label

// Create a source node for input
const a = source_node([30], () => input);

// Modify the computation graph to include the weight
const nn = a.mul(weight).add(bias).mse_loss(target);

// this is identical to mse_loss
// const nn = a.mul(weight).sub(target).pow(2).mean();

const graph = nn.get_computation_graph();

// Define learning rate
const learningRate = 10;

console.time();

// Training loop
for (let epoch = 0; epoch < 30; epoch++) {
    graph.zero_grad();
    graph.forward();
    graph.backward();

    // Print loss value
    console.log(`Epoch ${epoch + 1}: Loss = ${graph.outputs[0].value.toString()}`);

    // Update weights using SGD
    mul_acc(weight.grad!, -learningRate, weight.value);
    mul_acc(bias.grad!, -learningRate, bias.value);
}

console.write("\n\nTraining completed in: ");
console.timeEnd();
console.log(`  Weight value: ${weight.value.toString()}`);
// console.log(`  Weight grad:  ${weight.grad!.toString()}`);


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
