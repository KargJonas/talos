<br>

<p align="center">
  <img src="./talos-logo-big.png" />
</p>

## A minimalistic, zero-deps tensor library with a NumPy-like API
Talos is a recreational programming projects that helps me understand how machine learning works on the lowest levels. The goal for this project is to build a library that provides the basic array features of NumPy (step 1) and to then graft an autograd system and whatever else is necessary for training a basic model, onto it (step 2). Step 1 is now basically complete.

Talos uses C/WebAssembly to accelerate operations on tensors. All tensor data and metadata resides within WASM memory and and is accessed by JS only for validating operation parameters, printing and other things of that nature.

Each tensor is represented through this struct. Views of tensors will reference the data array of another tensor (usually along with an offset and different shape/strides).
```c
struct tensor_t {
    float* data;     // array that contains the actual tensor data/values
    size_t* shape;   // tensor shape of the form [..., n_matrices, n_rows, n_cols]
    size_t* strides; // strides same length as shape, counted in elements, not bytes like NumPy
    size_t rank;     // rank of the tensor. dictates length of shape/strides arrays
    size_t nelem;    // number of elements in the tensor. product of all elements of the shape array
    size_t ndata;    // number of elements in the data array (of topmost parent tensor)
    size_t offset;   // if view: offset of this view inside the parent tensor in number of elements default 0
    bool isview;     // indicates if this tensor is a view of another tensor
};
```

Here is a bit of sample code that demonstrates some of the features.
The full feature list is below.
```js
import { core_ready, tensor } from "../index";

// if your runtime does not support top-level await,
// you'll have to use core_ready.then(() => { ... }) instead
await core_ready;

const t1 = tensor([2, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
const t2 = tensor([3, 2],    [1, 2, 3, 4, 5, 6]);
const t3 = tensor([3],       [-1, 2, 3]);
const t4 = tensor([2, 2]).rand_int(1, 6);
const t5 = tensor([2, 2]).rand(1, 6);

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
t4.dot(t5, true).print();    // last parameter of op is always a bool indicating in-place op

t2.transpose().matmul(t2).print();
t2.matmul(t2.transpose()).print();

// in-place operations on views (this is something NumPy does not support)
// maybe useful for some applications, also saves you one transpose if you
// were going to to something like my_tensor.T.add(other_tensor).T
t1.transpose(0, 2, 1).add(t2, true);
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

```

## Features
### Currently, these are the operations that Talos supports:
Basic unary and binary operators, broadcasting, in-place and out-of-place operations. Pretty printing of tensors. A small set of tensor cloning/initialization methods.

- Binary operations (**broadcasting supported on all operations**)
    - Pairwise operations
        - Addition
        - Subtraction
        - Multiplication
        - Division
        - Exponentiation
    - Matrix multiplication
    - Dot product (mimics behavior of NumPy)
- Unary operations:
  - relu, binstep, logistic, negate, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, exp, log, log10, log2, invsqrt, sqrt, ceil, floor, abs, reciprocal, free, clone
- Reduce operations
  - Min
  - Max
  - Sum
  - Mean
- Metadata operations
  - transpose (with arbitrary permutation of axes)
  - view creation
  - iteration over specific axes
- Initialization
    - rand
    - rand_int
    - fill
    - zeros
    - ones
    - tensor(shape[], data[]?)
    - tensor_like(other_tensor)

### How to build
#### Prerequisites
To build and run this project, you will need to install `emcc` (emscripten), `make` and a js runtime environment like `bun` or `nodejs`.

```bash
# should work identically with node
bun install # install dev dependencies
bun run build # build wasm and ts (output in /dist)
```
