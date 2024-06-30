import { describe, expect } from "bun:test";
import tensor, { tensor_scalar } from "../src/base/Tensor";
import { core_ready } from "../src/base/Management";

describe("to_string", async () => {
    await core_ready;

    const t = tensor([2, 3], [1, 2, 3, 4, 5, 6]);
    expect(t.toString()).toEqual(`[[ 1,  2,  3 ]\n [ 4,  5,  6 ]]`);
    expect(t.T.toString()).toEqual(`[[ 1,  4 ]\n [ 2,  5 ]\n [ 3,  6 ]]`);
    expect(t.mul(Math.PI).toString()).toEqual(`[[ 3.14159,  6.28318,  9.42477 ]\n [ 12.56637, 15.70796, 18.84955]]`);
    expect(t.mul(Math.PI).toString(2)).toEqual(`[[ 3.14,  6.28,  9.42 ]\n [ 12.56, 15.7,  18.84]]`);
    expect(t.mul(Math.PI).toString(0)).toEqual(`[[ 3,   6,   9  ]\n [ 12,  15,  18 ]]`);

    const v = tensor([3], [2, 1, 3]);
    expect(v.toString()).toEqual(`[ 2, 1, 3 ]`);
    expect(v.mul(Math.PI).toString()).toEqual(`[ 6.28319, 3.14159, 9.42478 ]`);

    const s = tensor_scalar(5);
    expect(s.toString()).toEqual(`[ 5 ]`);
    expect(s.mul(Math.PI).toString()).toEqual(`[ 15.70796 ]`);
    expect(s.mul(Math.PI).toString(15)).toEqual(`[ 15.707963943481445 ]`);
    expect(s.mul(Math.PI).toString(0)).toEqual(`[ 16 ]`);
});
