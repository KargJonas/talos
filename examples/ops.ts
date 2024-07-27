import { core_ready, tensor, tensor_from_array } from "../dist";

// if your runtime does not support top-level await,
// you'll have to use core_ready.then(() => { ... }) instead
await core_ready;

// all of these functions create instances of the `Tensor` class.
// the arrays represent the shapes of the tensors.
// see `examples/create_and_init.ts` for more ways to create and initialize tensors
const t1 = tensor([2, 2, 3]);
const t2 = tensor([3, 2]);
const t3 = tensor([3]);

// create a tensor from a nested array
const t4 = tensor_from_array([[1, 2, 3], [4, 5, 6]]);

// populate tensors with random data
t1.uniform();
t2.normal(0, 3);
t3.xavier_normal(6, 3);

// talos supports broadcasting by repeating values along axes
// such that tensors of different rank may still be used together
t1.pow(2);  // scalar op
t1.pow(t1); // pairwise op
t1.pow(t3); // broadcasting op

// some basic ops
t1.add(t3);
t1.sub(t3);
t1.mul(t3);
t1.div(t3);

t1.matmul(t2);
t2.dot(t1);
t3.logistic();

// transpositions of tensors are implemented through views
// such that no additional data needs to be allocated
t2.T.matmul(t2);
t2.matmul(t2.transpose(1, 0)); // you can also use permutations for transposition

// talos is lazy, this means no actual computation will be performed,
// unless you call .realize()
const my_tensor = t1.add(t3).mul(t3).sub(4).T; // this tensor will contain only zeros (uninitialized)
const my_realized_tensor = t1.add(t3).mul(t3).sub(4).T.realize(); // this will contain a result

my_tensor.print();
my_realized_tensor.print();
