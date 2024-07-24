import { describe, expect, test } from "bun:test";
import { add } from "../src/raw_tensor/raw_tensor_operations.ts";
import { tensor_producer } from "../src/tensor_factory.ts";
import { RawTensor } from "../src/raw_tensor/raw_tensor.ts";

describe("node operations", () => {

    test("source nodes", () => {
        const input = RawTensor.scalar(0);
        const source = tensor_producer([1], () => {
            add(input, 1, input);
            return input;
        });

        expect(source.grad).toBeUndefined();

        source.fw();
        expect([...source.value.shape]).toEqual([1]);
        expect(source.grad).toBeUndefined();
        expect(source.value.item).toBeCloseTo(1);
        source.fw();
        expect(source.value.item).toBeCloseTo(2);
        source.fw();
        expect(source.value.item).toBeCloseTo(3);
        source.bw();
        expect(source.value.item).toBeCloseTo(3);
    });

    test("parameter nodes", () => {
        // todo: it may be a little too early to write tests for this.
        //       the api needs to be refined further.
        //       the library needs a refactor in general.
    });
});
