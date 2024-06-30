import { describe, expect, test } from "bun:test";
import { add } from "../src/base/tensor_operations";
import { source_node } from "../src/node_factory";
import { RawTensor } from "../src/base/RawTensor.ts";

describe("node operations", () => {

    test("source nodes", () => {
        const input = RawTensor.scalar(0);
        const source = source_node([1], () => {
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
        // todo: it may be a little too early to write tests for this
        //       the api needs to be refined further.
        //       the library needs a refactor in general.
    });
});
