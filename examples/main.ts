import { core_ready } from "../index";

// if your runtime does not support top-level await,
// you'll have to use core_ready.then(() => { ... }) instead
await core_ready;

const t1 = RawTensor.create([2, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
const t2 = RawTensor.create([3, 2],    [1, 2, 3, 4, 5, 6]);
const t3 = RawTensor.create([3],       [-1, 2, 3]);
const t5 = RawTensor.create([2, 2]).rand(1, 6);

for (const e of t1) {
    e.T.add(t2).print();
}

t1.pow(2).print();  // scalar op
t1.pow(t1).print(); // pairwise op
t1.pow(t3).print(); // broadcasting op

// some basic ops
t1.add(t3).print();
t1.sub(t3).print();
t1.mul(t3).print();
t1.div(t3).print();

t1.matmul(t2).print();
t2.dot(t1).print();
t3.logistic().print();
t4.matmul(t5, true).print(); // in-place matmul of square matrix
t4.dot(t5, true).print();

t2.transpose().matmul(t2).print();
t2.matmul(t2.transpose()).print();

// in-place operations on views (this is something NumPy does not support)
// maybe useful for some applications, also saves you one transpose if you
// were going to to something like my_tensor.T.add(other_tensor).T
t1.transpose(0, 2, 1).add(t2);
t1.print();

// manual creation of views
const axis = 1;
const offset_in_axis = 1;
const my_view = t1.create_view(axis, offset_in_axis); // getting the second element in the second axis
my_view.add(1, true); // modifying elements in my_view will modify the data of t1

// deep copy of tensors
const my_view_copy = my_view.clone();
my_view_copy.add(1, true); // this in-place operation does not affect the data of t1

// basic reduce operations
console.log(t4.sum());
console.log(t4.min());
console.log(t4.max());
console.log(t4.mean());
