/**
 * This file is used for validation and debugging during development. 
 */

import { core_ready } from "../src/base/Management.ts";
import { RawTensor } from "../src/base/RawTensor.ts";
import { add, min_tns } from "../src/base/raw_tensor_operations.ts";

// if your runtime does not support top-level await,
// you'll have to use core_ready.then(() => { ... }) instead
await core_ready;

try {
    console.log("\nRunning SGD demo...\n");

    const t = RawTensor.from_array([1,2,3,4]);
    const s = min_tns(t);
    t.print();
    add(s, 3, s);
    t.print();
} catch (e) {
    console.log(e);
}
