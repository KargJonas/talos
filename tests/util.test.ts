import {describe, test, expect} from "bun:test";
import {set_rand_seed} from "../src/base/util.ts";
import tensor from "../src/base/Tensor.ts";

describe("utility functions (util.js)", () => {
    test("core.set_rand_seed()", () => {
        // check that two tensors initialized randomly with the same seed are equal
        set_rand_seed(31415);
        const old_tensor = tensor([2]).rand();

        set_rand_seed(31415);
        const old_tensor_2 = tensor([2]).rand();
        expect([...old_tensor.data]).toEqual([...old_tensor_2.data]);

        // check that different seeds produce different values
        set_rand_seed(59307);
        const new_tensor = tensor([2]).rand();
        expect([...old_tensor.data]).not.toEqual([...new_tensor.data]);
    });
});
