import { describe, expect, test } from "bun:test";
import { RawTensor } from "../src/base/raw_tensor.ts";
import { core_ready } from "../src/base/management.ts";
import * as ops from "../src/base/raw_tensor_operations.ts";

describe("to_string", async () => {
    await core_ready;

    test("to_string", () => {
        // matrix
        const t = RawTensor.create([2, 3], [1, 2, 3, 4, 5, 6]);
        expect(t.toString()).toEqual("[[ 1,  2,  3 ]\n [ 4,  5,  6 ]]");
        expect(t.T.toString()).toEqual("[[ 1,  4 ]\n [ 2,  5 ]\n [ 3,  6 ]]");
        ops.mul(t, Math.PI, t);
        expect(t.toString()).toEqual("[[ 3.14159,  6.28318,  9.42477 ]\n [ 12.56637, 15.70796, 18.84955]]");
        expect(t.toString(2)).toEqual("[[ 3.14,  6.28,  9.42 ]\n [ 12.56, 15.7,  18.84]]");
        expect(t.toString(0)).toEqual("[[ 3,   6,   9  ]\n [ 12,  15,  18 ]]");

        // vector
        const v = RawTensor.create([3], [2, 1, 3]);
        expect(v.toString()).toEqual("[ 2, 1, 3 ]");
        ops.mul(v, Math.PI, v);
        expect(v.toString()).toEqual("[ 6.28319, 3.14159, 9.42478 ]");

        // scalar
        const s = RawTensor.scalar(5);
        expect(s.toString()).toEqual("[ 5 ]");
        ops.mul(s, Math.PI, s);
        expect(s.toString()).toEqual("[ 15.70796 ]");
        expect(s.toString(15)).toEqual("[ 15.707963943481445 ]");
        expect(s.toString(0)).toEqual("[ 16 ]");
    });
});
