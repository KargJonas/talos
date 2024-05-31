/**
 * This file is used for validation and debugging during development. 
 */

import { core_ready, tensor, mgmt, core } from "../index";

// if your runtime does not support top-level await,
// you'll have to use core_ready.then(() => { ... }) instead
await core_ready;

console.log("###########\n".repeat(2));

const t1 = tensor([2, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
const t2 = tensor([3, 2],    [1, 2, 3, 4, 5, 6]);
const t3 = tensor([3],       [-1, 2, 3]);
const t4 = tensor([2, 2]).rand_int(1, 6);
const t5 = tensor([2, 2]).rand(1, 6);

t1.sqrt().print();

// // testing access of tensors after a growth event
// const t2 = tensor([3, 2],    [1, 2, 3, 4, 5, 6]);
// t2.print_info();
//
// // forcing a growth event by allocating a lot of memory
// for (let i = 0; i < 100; i++) {
//     // console.log((mgmt.get_total_allocated() / 10 ** 6).toFixed(2) + " MB");
//     const a = tensor([100, 100, 100]);
//     // a.print_info();
// }

// t2.print_info();


// t2.print();

// const t6 = tensor([100, 100, 100]).rand();
// t6.print_info();
// t6.create_view().print_info();
