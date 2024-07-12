/**
 * This file is used for validation and debugging during development. 
 */

import { core_ready } from "../src/base/Management.ts";
import { RawTensor } from "../src/base/RawTensor.ts";
import { max_tns } from "../src/base/raw_tensor_operations.ts";
import { set_rand_seed } from "../src/base/util.ts";
import { tensor } from "../src/tensor_factory.ts";

// if your runtime does not support top-level await,
// you'll have to use core_ready.then(() => { ... }) instead
await core_ready;

try {
    set_rand_seed(Date.now());

    // const t = tensor([10], true).rand();
    // const v = RawTensor.view_of(t.value);
    // t.print();
    // t.max().realize().print();
    // max_tns(v).print();

    // const t = tensor([10], true).rand();

    // const nn = t.dropout();
    // nn.graph.forward();
    // nn.graph.backward();
    // nn.print();
    // t.grad?.print();

} catch (e) {
    console.log(e);
}
