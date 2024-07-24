import { core_ready, tensor } from "../dist";

// if your runtime does not support top-level await,
// you'll have to use core_ready.then(() => { ... }) instead
await core_ready;

const t1 = tensor([2, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
const t2 = tensor([3, 2],    [1, 2, 3, 4, 5, 6]);
const t3 = tensor([3],       [-1, 2, 3]);

t1.uniform();
t2.uniform();
t3.uniform();

t1.pow(2).realize().print();  // scalar op
t1.pow(t1).realize().print(); // pairwise op
t1.pow(t3).realize().print(); // broadcasting op

// some basic ops
t1.add(t3).realize().print();
t1.sub(t3).realize().print();
t1.mul(t3).realize().print();
t1.div(t3).realize().print();

t1.matmul(t2).realize().print();
t2.dot(t1).realize().print();
t3.logistic().realize().print();

t2.transpose().matmul(t2).realize().print();
t2.matmul(t2.transpose()).realize().print();

// in-place operations on views (this is something NumPy does not support)
// maybe useful for some applications, also saves you one transpose if you
// were going to to something like my_tensor.T.add(other_tensor).T
t1.transpose(0, 2, 1).add(t2);
t1.print();
