/**
 * This file is used for validation and debugging during development. 
 */

import { core_ready, set_rand_seed, tensor, tensor_from_array } from "../dist";

// Wait for the WASM module to load.
// NOTE: If your runtime does not support top-level await,
//       you can use core_ready.then(() => { ... }) instead.
await core_ready;

// Creating tensors:
const t1 = tensor([3, 2], [1, 2, 3, 4, 5, 6]);              // from shape + flat array
const t2 = tensor_from_array([[1, 2], [3, 4], [5, 6]]);     // from a nested array

// Print the two tensors
t1.print();
t2.print();

// tensor.print() will produce this for both tensors:
//   [[ 1,  2 ]
//    [ 3,  4 ]
//    [ 5,  6 ]]
//   ---

// All operations involving randomness will use seeds generated
// from one "master seed".
set_rand_seed(10);
set_rand_seed(Date.now());

// Initializing tensors with different random distributions:
const t3 = tensor([2, 5]);
t3.uniform(-1, 1);
t3.normal(0, 1);
t3.kaiming_uniform(10);
t3.kaiming_normal(10);
t3.xavier_uniform(10, 5);
t3.xavier_normal(10, 5);

// More ways to initialize tensors:
t3.ones();
t3.zeros();
t3.fill(42);
