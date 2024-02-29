# Talos
### Talos is a minimalistic, zero-dependencies tensor library with an API similar to NumPy

```js
import { tensor, core_ready, print, set_rand_seed } from "./index";

core_ready.then(() => {
    set_rand_seed(453455);

    //                 SHAPE      DATA
    const t1 = tensor([2, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const t2 = tensor([3, 2],    [1, 2, 3, 4, 5, 6]);
    const t4 = tensor([3],       [-1, 2, 3]);

    const t5 = tensor([2, 2]).rand_int(1, 6);
    const t6 = tensor([2, 2]).rand_int(1, 6);

    print(t1.add(t4));
    print(t1.sub(t4));
    print(t1.mul(t4));
    print(t1.div(t4));
    print(t1.matmul(t2));
    print(t2.dot(t1));
    print(t4.logistic());

    // in-place operations are supported for all ops
    print(t5.matmul(t6, true));
    print(t5.dot(t6, true));
});
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
    - Matrix multiplication
    - Dot product (mimics behavior of NumPy)
- Unary operations:
  - relu, binstep, logistic, negate, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, exp, log, log10, log2, invsqrt, sqrt, ceil, floor, abs, reciprocal, free, clone
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
