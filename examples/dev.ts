/**
 * This file is used for validation and debugging during development. 
 */

import { tensor, tensor_from_array as tensor_from_arr } from "../src/tensor_factory.ts";
import { core_ready, get_total_allocated } from "../src/base/Management.ts";
import { matmul_acc, mul_acc } from "../src/base/raw_tensor_operations.ts";
import { set_rand_seed } from "../src/base/util.ts";
import { RawTensor } from "../src/base/RawTensor.ts";

// if your runtime does not support top-level await,
// you'll have to use core_ready.then(() => { ... }) instead
await core_ready;

console.log("\nRunning SGD demo...\n");

// Input and target tensors

// const weight = parameter_node(RawTensor.create([3]).rand(), true);
// const bias = parameter_node(RawTensor.create([3]).rand(), true);
// const input = RawTensor.create([3]).rand();  // random but constant "input data"
// const target = RawTensor.create([3]).rand(); // random but constant target/label

// const weight = parameter_node(RawTensor.create([3], [6, 2, 8]), true);
// const bias = parameter_node(RawTensor.create([3], [0, 0, 0]), true);
// const input = RawTensor.create([3], [2, -3, 9]);  // random but constant "input data"
// const target = RawTensor.create([3], [23, 2, -3]); // random but constant target/label

// const A = tensor([2, 3], [1, 2, 3, 4, 5, 6]);
// const B = tensor([2, 3], [7, 8, 9, 10, 11, 12]);
// // const nn = A.T.matmul(B).lossFn();
// // => A.grad = B.matmul(loss_grad.T)
// // => B.grad = A.matmul(loss_node.grad)

// const y = A.add(B).T;
// y.realize().print();

const A = tensor([2, 3], [1, 2, 3, 4, 5, 6]);
const B = tensor([3], [10, 11, 12]);

A.print();
A.matmul(B).realize().print_info();

// set_rand_seed(Date.now());
// // const weight = tensor([3], true).rand();
// // const bias   = tensor([3], true).rand();
// const weight = tensor([2, 3], true).rand();
// const bias = tensor([2, 3], true).rand();
// const input  = tensor([2, 3]).rand();
// const target = tensor([3]).rand();

// // Create a source node for input
// // const a = source_node([3], () => input);

// // Modify the computation graph to include the weight
// // const nn = input.mul(weight).add(bias).mse_loss(target);
// // input.matmul(weight.T).realize().print();

// // const nn = input.matmul(weight.T).add(bias).mse_loss(target);

// // this is identical to mse_loss
// // const nn = a.mul(weight).sub(target).pow(2).mean();

// const graph = nn.graph;

// // Define learning rate
// const learningRate = 10;

// console.time();

// // Training loop
// for (let epoch = 0; epoch < 30; epoch++) {
//     graph.zero_grad();
//     graph.forward();
//     graph.backward();

//     // Print loss value
//     console.log(`\x1b[35m[${get_total_allocated()} Bytes]\x1b[0m Epoch ${epoch + 1}: Loss = ${graph.outputs[0].value.toString()}`);

//     // nn.grad?.print(10);

//     // Update weights using SGD
//     mul_acc(weight.grad!, -learningRate, weight.value);
//     mul_acc(bias.grad!, -learningRate, bias.value);

//     // print_memory_status();
// }

// console.write("\nTraining completed in: ");
// console.timeEnd();
// console.log(`  Weight value: ${weight.value.toString()}`);
// console.log(`  Weight grad:  ${weight.grad!.toString()}`);
