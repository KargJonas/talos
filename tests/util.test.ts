import { describe, test, expect } from "bun:test";
import { set_rand_seed } from "../src/raw_tensor/util.ts";
import { RawTensor } from "../src/raw_tensor/raw_tensor.ts";

describe("utility functions (util.js)", () => {
    test("core.set_rand_seed()", () => {
        // check that two tensors initialized randomly with the same seed are equal
        set_rand_seed(31415);
        const old_tensor = RawTensor.create([2]).rand();

        set_rand_seed(31415);
        const old_tensor_2 = RawTensor.create([2]).rand();
        expect([...old_tensor.data]).toEqual([...old_tensor_2.data]);

        // check that different seeds produce different values
        set_rand_seed(59307);
        const new_tensor = RawTensor.create([2]).rand();
        expect([...old_tensor.data]).not.toEqual([...new_tensor.data]);
    });
});
