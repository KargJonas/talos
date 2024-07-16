/**
 * This file is used for validation and debugging during development. 
 */

import { core } from "../index.ts";
import { core_ready } from "../src/base/Management.ts";
import { set_rand_seed } from "../src/base/util.ts";
import { tensor } from "../src/tensor_factory.ts";

// if your runtime does not support top-level await,
// you'll have to use core_ready.then(() => { ... }) instead
await core_ready;

try {
    set_rand_seed(Date.now());

    // const t = tensor([10], true).uniform();
    // const m = t.max().realize();
    // t.print();
    // m.print();
    // m.print_info();

    const v = tensor([3, 3], true).uniform();
    const v_max = v.value.create_view(v.value.rank);
    const v_grad_max = v.grad!.create_view(v.grad!.rank);

    v.grad!.data.set([1, 2, 3, 4, 5, 6, 7]);

    v.print();
    v.grad!.print();

    core._max_red_tns(v.value.ptr, v_max.ptr, v.grad!.ptr, v_grad_max.ptr);

    v_max.print();
    v_grad_max!.print();

    // TODO !!!!
    //    handle copying of pos in tensor.c

} catch (e) {
    console.log(e);
}
