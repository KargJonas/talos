import { core_ready } from "../src/util";
import tensor from "../src/Tensor";

// if your runtime does not support top-level await,
// you'll have to use core_ready.then(() => { ... }) instead
await core_ready;

const print = console.log;
print("###########\n".repeat(2));

const t1 = tensor([2, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
const t2 = tensor([3, 2],    [1, 2, 3, 4, 5, 6]);
const t3 = tensor([2, 3],    [1, 2, 3, 4, 5, 6]);
const t5 = tensor([3],       [-1, 2, 3]);


t2.T.matmul(t2).clone().print();

// const t4 = tensor([3, 2], [7.5, 5.5, -2, 3.5, 0, 3]);
// const t6 = tensor([2, 3], [-100, 2, 3, 2, 4, 2]);

// t1.dot(t2).print();
// t1.matmul(t2).print();

// t1.T.print();

// t2.print_info();
// t2.T.print_info();

// t2.matmul(t2.T).print();

// t1.matmul(t2).print();

// t1.dot(t2).print_info();
// t1.dot(t2).print();

// const a = t1.clone();
// for (const e of a) {
//     // print("e:");
//     // e.print();
//     // print("t2:");
//     // t2.print();
//     // print("t2.T:");
//     // t2.T.print();
//     // print("e + t2.T:");
//     // e.add(t2.T).print();

//     // e.T.print();
//     e.T.add(e.T).print();
// }

// a.print();


// const t4 = tensor([3],       [-1, 2, 3]);
// const t5 = tensor([2, 2]).rand_int(1, 6);
// const t6 = tensor([2, 2]).rand_int(1, 6);

// print(t1.add(t4));
// print(t1.sub(t4));
// print(t1.mul(t4));
// print(t1.div(t4));
// print(t1.matmul(t2));
// print(t2.dot(t1));
// print(t4.logistic());
// print(t5.matmul(t6, true));
// print(t5.dot(t6, true));

// print(t2.transpose().matmul(t2));
// print(t2.matmul(t2.transpose()));

// todo:
//   create some diagrams of the architecture

// todo:
//   reintroduce Tensor.get(...location). as a nice way to get tensor views.

// todo:
//   fix persisting trailing zeros of Tensor.print()

// todo:
//   add missing pow operation

// todo:
//   currently, matrix multiplication and many other operations would not work on tensor
//   views. we need to update the backend code to accommodate the introduced changes.
//   maybe i can take this opportunity to do some benchmarks on function call overhead in
//   the deeply nested loops and see if i cant create a function for accessing indices
//   of tensors through strides. (refactoring)

// todo:
//   reconsider the way in_place operations are currently performed!
//   (transpose got me thinking... you can never perform a transpose operation in-place)
//   it should always return a view
//   on the other hand, there are operations that should never return views like all operations
//   where data is modified

// todo:
//   work through all todos
