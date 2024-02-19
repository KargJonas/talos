import { core_ready } from "../src/util";
import tensor from "../src/Tensor";

await core_ready;

console.log("###########\n".repeat(2));

const t1 = tensor([2, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
const t2 = tensor([3, 2],    [1, 2, 3, 4, 5, 6]);
const t4 = tensor([3],       [-1, 2, 3]);
const t5 = tensor([2, 2]).rand_int(1, 6);
const t6 = tensor([2, 2]).rand_int(1, 6);

// t2.T.matmul(t2).print();

// t2.print();
// t2.T.div(.9).print();

for (const e of t1.get_axis_iterable(0)) {
    e.T.add(t2, true);
}

t1.print();

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
