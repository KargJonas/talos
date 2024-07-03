import { describe, expect, test } from "bun:test";
import { core_ready } from "../src/base/Management";
import Shape from "../src/base/Shape";
import Strides from "../src/base/Strides";
import Tensor from "../src/Tensor.ts";
import {tensor} from "../src/tensor_factory.ts";

describe("tensor creation", async () => {
    await core_ready;

    const t1 = tensor([2, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const t2  = tensor([3, 2], [1, 2, 3, 4, 5, 6]);
    const t13 = tensor([2, 2, 5, 2], [0.261, 0.983, 0.857, 0.279, 0.211, 0.75, 0.671, 0.32, 0.641, 0.317, 0.003, 0.951, 0.332, 0.226, 0.409, 0.475, 0.348, 0.205, 0.11, 0.353, 0.557, 0.309, 0.06, 0.963, 0.368, 0.607, 0.047, 0.52, 0.5, 0.532, 0.138, 0.686, 0.982, 0.134, 0.984, 0.488, 0.856, 0.766, 0.208, 0.29]);
    const t14 = tensor([5, 5], [0.93, 0.793, 0.149, 0.827, 0.167, 0.563, 0.485, 0.969, 0.63, 0.909, 0.113, 0.11, 0.627, 0.198, 0.708, 0.574, 0.565, 0.678, 0.013, 0.195, 0.913, 0.131, 0.016, 0.418, 0.277]);

    test("transpose", () => {
        // make sure invalid permutations are caught
        expect(() => t1.transpose(1, 2, 3)).toThrow();
        expect(() => t1.transpose(0, 1, 1)).toThrow();
        expect(() => t1.transpose(0, 1, 2, 3)).toThrow();
        expect(() => t1.transpose(0, 1)).toThrow();

        // basic testing of transpose functionality
        test_chained_ops(
            t2,
            t => t.T,
            [1, 2, 3, 4, 5, 6],
            [2, 3],
            [1, 2]
        );

        test_chained_ops(
            t2,
            t => t.T.clone(),
            [1, 3, 5, 2, 4, 6],
            [2, 3],
            [3, 1]
        );

        // transposition is its own inverse operation,
        // this means transposing twice should yield the original tensor
        test_chained_ops(
            t1,
            t => t.T.T,
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [2, 2, 3],
            [6, 3, 1]
        );

        test_chained_ops(
            t1,
            t => t.T.T.clone(),
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [2, 2, 3],
            [6, 3, 1]
        );

        // double transpose with interlaced clone (caused some issues during implementation)
        test_chained_ops(
            t2,
            t => t.T.clone().T.clone(),
            [1, 2, 3, 4, 5, 6],
            [3, 2],
            [2, 1]
        );

        test_chained_ops(
            t1,
            t => t.T.clone().T.clone(),
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [2, 2, 3],
            [6, 3, 1]
        );

        // testing of transpose on rank 3 tensors
        test_chained_ops(
            t1,
            t => t.T,
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [3, 2, 2],
            [1, 3, 6]
        );

        test_chained_ops(
            t1,
            t => t.T.clone(),
            [1, 7, 4, 10, 2, 8, 5, 11, 3, 9, 6, 12],
            [3, 2, 2],
            [4, 2, 1]
        );

        // testing of custom permutations
        test_chained_ops(
            t1,
            t => t.transpose(0, 2, 1),
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [2, 3, 2],
            [6, 1, 3]
        );

        test_chained_ops(
            t1,
            t => t.transpose(0, 2, 1).clone(),
            [1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12],
            [2, 3, 2],
            [6, 2, 1]
        );

        test_chained_ops(
            t1,
            t => t.transpose(1, 0, 2),
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [2, 2, 3],
            [3, 6, 1]
        );

        test_chained_ops(
            t1,
            t => t.transpose(1, 0, 2).clone(),
            [1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12],
            [2, 2, 3],
            [6, 3, 1]
        );
    });

    test("transpose + other ops", () =>  {
        expect_arrays_closeto(t2.T.matmul(t2).clone().data, [ 35, 44, 44, 56 ]);
        expect_arrays_closeto(t2.matmul(t2.T).clone().data, [ 5, 11, 17, 11, 25, 39, 17, 39, 61 ]);

        test_chained_ops(
            t2,
            t => t.T.matmul(t2),
            [35, 44, 44, 56],
            [2, 2],
            [2, 1]
        );

        test_chained_ops(
            t2,
            t => t.clone().T.clone().matmul(t2.clone()),
            [35, 44, 44, 56],
            [2, 2],
            [2, 1]
        );

        test_chained_ops(
            t1,
            t => t.add(t2.T),
            [2, 5, 8, 6, 9 , 12, 8, 11, 14, 12, 15, 18],
            [2, 2, 3],
            [6, 3, 1]
        );

        test_chained_ops(
            t1,
            t => t.clone().transpose(0, 2, 1).add(t2),
            [2, 6, 5, 9, 8, 12, 8, 12, 11, 15, 14, 18],
            [2, 3, 2],
            [6, 2, 1]
        );

        test_chained_ops(
            t14,
            t => t.T.matmul(t14).T.clone(),
            [2.358, 1.467, 1.159, 1.535, 1.112, 1.467, 1.213, 1.042, 1.045, 0.798, 1.159, 1.042, 1.814, 0.873, 1.486, 1.535, 1.045, 0.873, 1.295, 0.969, 1.112, 0.798, 1.486, 0.969, 1.47],
            [5, 5],
            [5, 1]
        );

        test_chained_ops(
            t14,
            t => t.matmul(t14.T).clone(),
            [2.228, 1.725, 0.568, 1.126, 1.347, 1.725, 2.714, 1.493, 1.44, 1.108, 0.568, 1.493, 0.958, 0.693, 0.406, 1.126, 1.44, 0.693, 1.147, 0.668, 1.347, 1.108, 0.406, 0.668, 1.102],
            [5, 5],
            [5, 1]
        );

        // todo might want to add some tests for in-place ops on views
    });

    test("reduce operations", () => {
        expect(t14.sum()).toBeCloseTo(11.958);
        expect(t13.sum()).toBeCloseTo(18.697);
        expect(t14.min()).toBeCloseTo(0.013);
        expect(t13.min()).toBeCloseTo(0.003);
        expect(t14.max()).toBeCloseTo(0.969);
        expect(t13.max()).toBeCloseTo(0.985);
        expect(t14.mean()).toBeCloseTo(0.478);
        expect(t13.mean()).toBeCloseTo(0.467);
    });
});

function expect_arrays_closeto(a: number[] | Float32Array, b: number[] | Float32Array) {
    [...a].map((v, i) => {
        if (Number.isNaN(v)) expect(b[i]).toBeNaN();
        else expect(v).toBeCloseTo(b[i]);
    });
}

function test_chained_ops(
    input_tensor:     Tensor,
    operations:       (tensor: Tensor) => Tensor,
    expected_data:    number[] | Float32Array,
    expected_shape:   number[] | Shape,
    expected_strides: number[] | Strides
) {
    const result = operations(input_tensor);

    expect([...result.value.shape]).toEqual([...expected_shape]);
    expect([...result.value.strides]).toEqual([...expected_strides]);
    expect_arrays_closeto(result.value.data, expected_data);
}