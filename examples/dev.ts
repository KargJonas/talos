/**
 * This file is used for validation and debugging during development. 
 */

import { RawTensor, core } from "../index.ts";
import { core_ready } from "../src/base/Management.ts";
import { set_rand_seed } from "../src/base/util.ts";
import { tensor } from "../src/tensor_factory.ts";

// if your runtime does not support top-level await,
// you'll have to use core_ready.then(() => { ... }) instead
await core_ready;

try {
    set_rand_seed(Date.now());

    const t = tensor([10], true).uniform();
    const m = t.max().add(1);

    // t.grad!.uniform();

    t.print();
    t.grad!.print();

    m.graph.forward();
    m.graph.backward();

    m.print();
    m.grad!.print();

    t.print();
    t.grad!.print();

} catch (e) {
    console.log(e);
}
