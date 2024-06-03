/**
 * This file is used for validation and debugging during development. 
 */

import {core_ready, Tensor, tensor} from "../index";
import {node} from "../src/Node.ts";

// if your runtime does not support top-level await,
// you'll have to use core_ready.then(() => { ... }) instead
await core_ready;

console.log("###########\n".repeat(2));

const a = tensor([2, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
const c = tensor([2, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
const b = tensor([1], [1]);

// note: the addition gives the wrong result
//       should we handle scalar tensors differently (explicit rank 0)
//       or should we just keep it shape [1]?
// b.print_info();
a.print();
a.add(c).print();


// const dataset_x_0: Tensor = tensor([50, 4, 4]).rand();    // 50 "images" of size 4x4
// const dataset_x_1: Tensor = tensor([50, 4, 4]).rand();    // additional information for each image
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
// // Result tensor
// const result = input_0.add(input_1);
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
// result.grad.print();
