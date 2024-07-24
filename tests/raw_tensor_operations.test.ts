import { describe, expect, test } from "bun:test";
import { RawTensor } from "../src/raw_tensor/raw_tensor.ts";
import Shape from "../src/raw_tensor/shape.ts";
import * as ops from "../src/raw_tensor/raw_tensor_operations.ts";
import { core_ready } from "../src/raw_tensor/management.ts";
import Strides from "../src/raw_tensor/strides.ts";

// todo:
//  - potentially add tests with large identity-matrices (easy to validate without other libraries)
//  - add large rand test that checks that there are lots of zeros in a tensor (good indicator for some bugs)
//  - add tests for finding memory leaks

describe("tensor operations", async () => {
    await core_ready;

    const t1 = RawTensor.create([2, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const t2 = RawTensor.create([3, 2], [1, 2, 3, 4, 5, 6]);
    const t3 = RawTensor.create([3, 2], [-100, 2, 3, 2, 4, 2]);
    const t4 = RawTensor.create([3, 2], [7.5, 5.5, -2, 3.5, 0, 3]);
    const t5 = RawTensor.create([3], [-1, 2, 3]);
    const t6 = RawTensor.create([2, 3], [-100, 2, 3, 2, 4, 2]);
    const t7 = RawTensor.create([2, 3, 4], [8, 4, 4, 2, 3, 1, 4, 9, 2, 0, 9, 0, 4, 5, 1, 3, 2, 3, 4, 5, 8, 1, 2, 3]);
    const t8 = RawTensor.create([4, 3], [0, 2, 3, 0, 3, 6, 5, 7, 3, 1, 2, 1]);
    const t9 = RawTensor.create([2, 2, 4, 3], [0.82, 0.447, 0.716, 0.057, 0.245, 0.855, 0.288, 0.902, 0.162, 0.091, 0.225, 0.892, 0.808, 0.924, 0.967, 0.084, 0.623, 0.686, 0.042, 0.358, 0.54, 0.54, 0.195, 0.298, 0.783, 0.477, 0.117, 0.894, 0.927, 0.532, 0.184, 0.86, 0.543, 0.57, 0.719, 0.72, 0.184, 0.989, 0.863, 0.784, 0.143, 0.156, 0.403, 0.187, 0.304, 0.824, 0.514, 0.731]);
    const t10 = RawTensor.create([3, 1], [0.021, 0.782, 0.253]);
    const t11 = RawTensor.create([3, 4, 5], [0.316, 0.057, 0.639, 0.295, 0.726, 0.135, 0.742, 0.346, 0.789, 0.503, 0.745, 0.907, 0.91, 0.239, 0.999, 0.372, 0.118, 0.414, 0.275, 0.76, 0.98, 0.574, 0.886, 0.247, 0.259, 0.574, 0.129, 0.546, 0.508, 0.403, 0.097, 0.072, 0.286, 0.141, 0.37, 0.153, 0.585, 0.994, 0.399, 0.74, 0.063, 0.904, 0.384, 0.158, 0.904, 0.478, 0.237, 0.714, 0.732, 0.231, 0.814, 0.88, 0.91, 0.764, 0.778, 0.912, 0.764, 0.977, 0.158, 0.493]);
    const t12 = RawTensor.create([3, 5, 6], [0.084, 0.078, 0.868, 0.891, 0.331, 0.668, 0.829, 0.305, 0.899, 0.636, 0.855, 0.854, 0.627, 0.078, 0.884, 0.297, 0.52, 0.722, 0.248, 0.663, 0.353, 0.68, 0.298, 0.384, 0.929, 0.714, 0.273, 0.737, 0.644, 0.072, 0.304, 0.755, 0.405, 0.676, 0.597, 0.55, 0.785, 0.333, 0.26, 0.385, 0.895, 0.062, 0.396, 0.904, 0.518, 0.316, 0.839, 0.581, 0.057, 0.555, 0.101, 0.986, 0.348, 0.549, 0.733, 0.629, 0.745, 0.723, 0.752, 0.515, 0.501, 0.148, 0.816, 0.901, 0.534, 0.748, 0.197, 0.581, 0.747, 0.295, 0.048, 0.488, 0.727, 0.313, 0.39, 0.652, 0.645, 0.686, 0.21, 0.18, 0.557, 0.779, 0.926, 0.272, 0.005, 0.539, 0.023, 0.208, 0.243, 0.39]);
    const t13 = RawTensor.create([2, 2, 5, 2], [0.261, 0.983, 0.857, 0.279, 0.211, 0.75, 0.671, 0.32, 0.641, 0.317, 0.003, 0.951, 0.332, 0.226, 0.409, 0.475, 0.348, 0.205, 0.11, 0.353, 0.557, 0.309, 0.06, 0.963, 0.368, 0.607, 0.047, 0.52, 0.5, 0.532, 0.138, 0.686, 0.982, 0.134, 0.984, 0.488, 0.856, 0.766, 0.208, 0.29]);
    const t14 = RawTensor.create([5, 5], [0.93, 0.793, 0.149, 0.827, 0.167, 0.563, 0.485, 0.969, 0.63, 0.909, 0.113, 0.11, 0.627, 0.198, 0.708, 0.574, 0.565, 0.678, 0.013, 0.195, 0.913, 0.131, 0.016, 0.418, 0.277]);
    const t15 = RawTensor.create([5, 1], [0.891, 0.549, 0.65, 0.02, 0.676]);
    const t16 = RawTensor.create([1, 3], [7, 3, 1]);
    const t17 = RawTensor.create([8, 1], [9, 8, 7, 6, 5, 4, 3, 2]);

    function expect_arrays_closeto(a: number[] | Float32Array, b: number[] | Float32Array) {
        [...a].map((v, i) => {
            if (Number.isNaN(v)) expect(b[i]).toBeNaN();
            else expect(v).toBeCloseTo(b[i]);
        });
    }

    function unary(unary_op: ops.UnaryOp, a: RawTensor, expectation: (v: number) => number, in_place = false) {
        let t = a.clone();
        const input = [...t.data];

        t = unary_op(t, in_place ? t : undefined);
        expect([...t.shape]).toEqual([...a.shape]);

        [...t.data].map((v, i) => {
            if (Number.isNaN(v)) expect(expectation(input[i])).toBeNaN();
            else expect(v).toBeCloseTo(expectation(input[i]));
        });

        t.free();
    }

    function binary(
        binary_op: ops.BinaryOp<RawTensor | number> | ops.BinaryOp<RawTensor>,
        a: RawTensor, b: RawTensor | number,
        expected_shape: Shape | number[],
        expected_data: number[],
        in_place = false
    ) {
        let t = a.clone();

        // // @ts-expect-error Incompatibility between type of b. Ugly to fix. Guaranteed to work anyways.
        // todo: fix underlying type issue: should distinguish between binary ops that take tensors and ones that take scalars in raw_tensor_operations.ts
        t = binary_op(t, b, in_place ? t : undefined);
        expect([...t.shape]).toEqual([...expected_shape]);

        [...t.clone().data].map((v, i) => {
            if (Number.isNaN(v)) expect(expected_data[i]).toBeNaN();
            else expect(v).toBeCloseTo(expected_data[i], 3);
        });

        t.free();
    }

    function test_chained_ops(
        input_tensor:     RawTensor,
        operations:       (tensor: RawTensor) => RawTensor,
        expected_data:    number[] | Float32Array,
        expected_shape:   number[] | Shape,
        expected_strides: number[] | Strides
    ) {
        const result = operations(input_tensor);

        expect([...result.shape]).toEqual([...expected_shape]);
        expect([...result.strides]).toEqual([...expected_strides]);
        expect_arrays_closeto(result.data, expected_data);
    }

    describe("in-place", () => {
        test("unary ops", () => {
            unary(ops.relu, t5, v => v < 0 ? 0 : v, true);
            unary(ops.binstep, t5, v => v < 0 ? 0 : 1, true);
            unary(ops.logistic, t5, v => 1 / (Math.exp(-v) + 1), true);
            unary(ops.negate, t5, v => -v, true);

            unary(ops.sin, t5, Math.sin, true);
            unary(ops.cos, t5, Math.cos, true);
            unary(ops.tan, t5, Math.tan, true);
            unary(ops.asin, t5, Math.asin, true);
            unary(ops.acos, t5, Math.acos, true);
            unary(ops.atan, t5, Math.atan, true);
            unary(ops.sinh, t5, Math.sinh, true);
            unary(ops.cosh, t5, Math.cosh, true);
            unary(ops.tanh, t5, Math.tanh, true);

            unary(ops.exp, t5, Math.exp, true);
            unary(ops.log, t5, Math.log, true);
            unary(ops.log10, t5, Math.log10, true);
            unary(ops.log2, t5, Math.log2, true);
            unary(ops.invsqrt, t1, v => 1 / Math.sqrt(v), true);
            unary(ops.sqrt, t1, Math.sqrt, true);
            unary(ops.ceil, t4, Math.ceil, true);
            unary(ops.floor, t4, Math.floor, true);
            unary(ops.floor, t4, Math.floor, true);
            unary(ops.abs, t4, Math.abs, true);
            unary(ops.reciprocal, t4, v => 1 / v, true);
        });

        test("scalar ops", () => {
            const t = t1.clone();
            let expected = [...t.data];

            ops.add(t, 1, t);
            // t.add(1, true);
            expected = expected.map(v => v + 1);
            expect([...t.data]).toEqual(expected);

            // t.sub(2.5, true);
            ops.sub(t, 2.5, t);
            expected = expected.map(v => v - 2.5);
            expect([...t.data]).toEqual(expected);

            // t.mul(2, true);
            ops.mul(t, 2, t);
            expected = expected.map(v => v * 2);
            expect([...t.data]).toEqual(expected);

            // t.div(4, true);
            ops.div(t, 4, t);
            expected = expected.map(v => v / 4);
            expect([...t.data]).toEqual(expected);

            // t.pow(Math.PI, true);
            ops.pow(t, Math.PI, t);
            expected = expected.map(v => Math.pow(v, Math.PI));
            expect_arrays_closeto(t.data, expected);

            expect([...t1.shape]).toEqual([...t.shape]);
        });

        // warning: this test can cause precision-based errors!
        test("pairwise ops", () => {
            const t = t2.clone();
            let expected = [...t.data];

            // t.add(t3, true);
            ops.add(t, t3, t);
            expected = expected.map((v, i) => v + t3.data[i]);
            expect_arrays_closeto(t.data, expected);

            // t.sub(t4, true);
            ops.sub(t, t4, t);
            expected = expected.map((v, i) => v - t4.data[i]);
            expect_arrays_closeto(t.data, expected);

            // t.mul(t3, true);
            ops.mul(t, t3, t);
            expected = expected.map((v, i) => v * t3.data[i]);
            expect_arrays_closeto(t.data, expected);

            // t.div(t2, true);
            ops.div(t, t2, t);
            expected = expected.map((v, i) => v / t2.data[i]);
            expect_arrays_closeto(t.data, expected);

            // t.pow(t2, true);
            ops.pow(t, t2, t);
            expected = expected.map((v, i) => Math.pow(v, t2.data[i]));
            expect_arrays_closeto(t.data, expected);
        });

        test("broadcasting ops", () => {
            binary(ops.add, t1, t5, t1.shape, [0, 4, 6, 3, 7, 9, 6, 10, 12, 9, 13, 15]);
            binary(ops.sub, t1, t5, t1.shape, [2, 0, 0, 5, 3, 3, 8, 6, 6, 11, 9, 9]);
            binary(ops.mul, t1, t5, t1.shape, [-1, 4, 9, -4, 10, 18, -7, 16, 27, -10, 22, 36]);
            binary(ops.div, t1, t5, t1.shape, [-1, 1, 1, -4, 2.5, 2, -7, 4, 3, -10, 5.5, 4]);
            binary(ops.pow, t1, t6, t1.shape, [1, 4, 27, 16, 625, 36, 0, 64, 729, 100, 14641, 144]);
        });

        test("matmul", () => {
            expect(() => ops.matmul(t1, t2, t1)).toThrow();
            expect(() => ops.matmul(t4, t6, t4)).toThrow();
            expect(() => ops.matmul(t7, t8, t7)).toThrow();
            expect(() => ops.matmul(t9, t10, t9)).toThrow();

            binary(ops.matmul, t14, t15, [5, 1], [1.49, 2.025, 1.051, 1.394, 1.091]);
            // more thorough testing in out-of-place tests
        });

        test("dot", () => {
            expect(() => ops.dot(t1, t5, t1)).toThrow();
            expect(() => ops.dot(t4, t5, t4)).toThrow();

            binary(ops.matmul, t14, t15, [5, 1], [1.49, 2.025, 1.051, 1.394, 1.091]);
            // more thorough testing in out-of-place tests
        });
    });

    describe("out-of-place", () => {
        describe("unary ops", () => {
            describe("direct assignment", () => {
                test("pairwise", () => {
                    unary(ops.relu, t5, v => v < 0 ? 0 : v);
                    unary(ops.binstep, t5, v => v < 0 ? 0 : 1);
                    unary(ops.logistic, t5, v => 1 / (Math.exp(-v) + 1));
                    unary(ops.negate, t5, v => -v);

                    unary(ops.sin, t5, Math.sin);
                    unary(ops.cos, t5, Math.cos);
                    unary(ops.tan, t5, Math.tan);
                    unary(ops.asin, t5, Math.asin);
                    unary(ops.acos, t5, Math.acos);
                    unary(ops.atan, t5, Math.atan);
                    unary(ops.sinh, t5, Math.sinh);
                    unary(ops.cosh, t5, Math.cosh);
                    unary(ops.tanh, t5, Math.tanh);

                    unary(ops.exp, t5, Math.exp);
                    unary(ops.log, t5, Math.log);
                    unary(ops.log10, t5, Math.log10);
                    unary(ops.log2, t5, Math.log2);
                    unary(ops.invsqrt, t1, v => 1 / Math.sqrt(v));
                    unary(ops.sqrt, t1, Math.sqrt);
                    unary(ops.ceil, t4, Math.ceil);
                    unary(ops.floor, t4, Math.floor);
                    unary(ops.floor, t4, Math.floor);
                    unary(ops.abs, t4, Math.abs);
                    unary(ops.reciprocal, t4, v => 1 / v);
                });
            });

            test("accumulative assignment", () => {
                // const t6 = RawTensor.create([2, 3], [-100, 2, 3, 2, 4, 2]);
                const res = RawTensor.like(t6).zeros();

                ops.relu_acc(t6, res);
                expect_arrays_closeto(res.data, [0, 2, 3, 2, 4, 2]);
                ops.sin_acc(t6, res);
                expect_arrays_closeto(res.data, [0.50636566, 2.9092975, 3.14112, 2.9092975, 3.2431974, 2.9092975]);
            });
        });

        test("scalar ops", () => {
            binary(ops.add, t1, 1, t1.shape, [...t1.data].map(v => v + 1));
            binary(ops.sub, t1, 2.5, t1.shape, [...t1.data].map(v => v - 2.5));
            binary(ops.mul, t1, 2, t1.shape, [...t1.data].map(v => v * 2));
            binary(ops.div, t1, 4, t1.shape, [...t1.data].map(v => v / 4));
            binary(ops.pow, t1, 4, t1.shape, [...t1.data].map(v => Math.pow(v, 4)));
        });

        // warning: this test can cause precision-based errors!
        test("pairwise ops", () => {
            binary(ops.add, t2, t3, t2.shape, [...t2.data].map((v, i) => v + t3.data[i]));
            binary(ops.sub, t2, t4, t2.shape, [...t2.data].map((v, i) => v - t4.data[i]));
            binary(ops.mul, t2, t3, t2.shape, [...t2.data].map((v, i) => v * t3.data[i]));
            binary(ops.div, t2, t4, t2.shape, [...t2.data].map((v, i) => v / t4.data[i]));
            binary(ops.pow, t2, t4, t2.shape, [...t2.data].map((v, i) => Math.pow(v, t4.data[i])));
        });

        test("broadcasting ops", () => {
            binary(ops.add, t1, t5, t1.shape, [0, 4, 6, 3, 7, 9, 6, 10, 12, 9, 13, 15]);
            binary(ops.sub, t1, t5, t1.shape, [2, 0, 0, 5, 3, 3, 8, 6, 6, 11, 9, 9]);
            binary(ops.mul, t1, t5, t1.shape, [-1, 4, 9, -4, 10, 18, -7, 16, 27, -10, 22, 36]);
            binary(ops.div, t1, t5, t1.shape, [-1, 1, 1, -4, 2.5, 2, -7, 4, 3, -10, 5.5, 4]);
            binary(ops.add, t16, t17, [8, 3], [16, 12, 10, 15, 11, 9, 14, 10, 8, 13, 9, 7, 12, 8, 6, 11, 7, 5, 10, 6, 4, 9, 5, 3]);
            binary(ops.pow, t1, t6, [2, 2, 3], [1, 4, 27, 16, 625, 36, 0, 64, 729, 100, 14641, 144]);
        });

        test("matmul", () => {
            expect(() => ops.matmul(t1, t5)).toThrow();
            expect(() => ops.matmul(t4, t5)).toThrow();
            expect(() => ops.matmul(t11, t13)).toThrow();

            binary(ops.matmul, t1, t2, [2, 2, 2], [22, 28, 49, 64, 76, 100, 103, 136]);
            binary(ops.matmul, t4, t6, [3, 3], [-739.0, 37.0, 33.5, 207.0, 10.0, 1.0, 6.0, 12.0, 6.0]);
            binary(ops.matmul, t7, t8, [2, 3, 3], [22, 60, 62, 29, 55, 36, 45, 67, 33, 8, 36, 48, 25, 51, 41, 13, 39, 39]);
            binary(ops.matmul, t9, t10, [2, 2, 4, 1], [0.548, 0.409, 0.752, 0.404, 0.984, 0.663, 0.417, 0.239, 0.419, 0.878, 0.814, 0.756, 0.996, 0.168, 0.232, 0.604]);
            binary(ops.matmul, t11, t12, [3, 4, 6], [1.222, 0.806, 1.193, 1.243, 1.041, 0.887, 1.506, 1.146, 1.506, 1.602, 1.418, 1.313, 2.372, 1.277, 2.624, 2.41, 2.21, 2.093, 1.163, 0.822, 1.1, 1.277, 1.011, 0.808, 1.303, 2.032, 1.223, 1.594, 2.123, 1.358, 0.816, 1.505, 0.9, 1.402, 1.396, 1.127, 0.479, 0.667, 0.496, 0.59, 0.69, 0.492, 1.465, 1.896, 1.321, 1.571, 2.144, 1.298, 0.527, 1.17, 0.985, 0.885, 0.691, 1.147, 0.96, 0.688, 1.259, 1.584, 1.461, 1.252, 1.407, 1.473, 2.12, 2.343, 1.96, 2.174, 1.353, 1.179, 1.795, 1.91, 1.42, 1.96]);
        });

        test("dot", () => {
            expect(() => ops.dot(t1, t5)).toThrow();
            expect(() => ops.dot(t4, t5)).toThrow();

            binary(ops.dot, t1, t2, [2, 2, 2], [22, 28, 49, 64, 76, 100, 103, 136]);
            binary(ops.dot, t4, t6, [3, 3], [-739.0, 37.0, 33.5, 207.0, 10.0, 1.0, 6.0, 12.0, 6.0]);
            binary(ops.dot, t1, t2, [2, 2, 2], [22, 28, 49, 64, 76, 100, 103, 136]);
            binary(ops.dot, t7, t8, [2, 3, 3], [22, 60, 62, 29, 55, 36, 45, 67, 33, 8, 36, 48, 25, 51, 41, 13, 39, 39]);
            binary(ops.dot, t9, t10, [2, 2, 4, 1], [0.548, 0.409, 0.752, 0.404, 0.984, 0.663, 0.417, 0.239, 0.419, 0.878, 0.814, 0.756, 0.996, 0.168, 0.232, 0.604]);
            binary(ops.dot, t11, t13, [3, 4, 2, 2, 2], [0.929, 1.13, 0.464, 0.934, 0.791, 1.08, 1.132, 0.973, 1.596, 1.011, 0.718, 0.8, 0.536, 1.644, 1.868, 1.111, 1.964, 2.061, 0.869, 1.747, 1.315, 2.312, 2.301, 1.549, 0.957, 1.038, 0.389, 0.902, 0.76, 1.027, 0.968, 0.904, 1.266, 1.949, 0.67, 1.625, 1.047, 1.66, 1.836, 1.446, 0.975, 1.3, 0.489, 1.081, 0.754, 1.112, 1.262, 1.183, 0.479, 0.492, 0.231, 0.404, 0.355, 0.543, 0.563, 0.431, 1.493, 1.421, 0.821, 1.093, 0.875, 1.815, 2.069, 1.189, 1.558, 0.939, 0.612, 0.798, 0.69, 1.686, 1.598, 0.735, 1.118, 1.379, 0.652, 1.079, 0.693, 1.313, 1.676, 1.336, 2.17, 2.219, 1.018, 1.836, 1.266, 2.463, 2.688, 1.931, 1.521, 2.049, 0.765, 1.71, 1.167, 1.955, 2.075, 1.469]);
        });
    
        describe("accumulative operations", () => {
            test("pairwise", () => {
                const a = RawTensor.create([4, 2], [4, 7, 22, 8, 3, 2, 89, 2]);
                const b = RawTensor.create([4, 2], [1, 2, 3,  4, 5, 6, 7,  8]);
                const res = RawTensor.like(a).zeros();
    
                ops.add_acc(a, b, res);
                expect_arrays_closeto(res.data, [5, 9, 25, 12, 8, 8, 96, 10]);
                ops.mul_acc(a, b, res);
                expect_arrays_closeto(res.data, [9, 23, 91, 44, 23, 20, 719, 26]);
                ops.sub_acc(b, a, res);
                expect_arrays_closeto(res.data, [6, 18, 72, 40, 25, 24, 637, 32]);
                ops.div_acc(a, b, res);
                // expect_arrays_closeto(res.data, [6+4/1, 18+7/2, 72+22/3, 40+8/4, 25+3/5, 24+2/6, 637+89/7, 32+2/8]);
                expect_arrays_closeto(res.data, [10, 21.5, 79.333, 42, 25.6, 24.333, 649.714, 32.25]);
                ops.pow_acc(a, b, res);
                expect_arrays_closeto(res.data, [10+4**1, 21.5+7**2, 79.333+22**3, 42+8**4, 25.6+3**5, 24.333+2**6, 44231334821888, 32.25+2**8]);
            
                a.free();
                b.free();
                res.free();
            });
    
            test("broadcasting", () => {
                const a = RawTensor.create([2, 2], [1, 2, 3, 4]);  
                const b = RawTensor.create([2], [10, 20]);        
                const res = RawTensor.like(a).zeros();     
            
                ops.add_acc(a, b, res);  
                expect_arrays_closeto(res.data, [11, 22, 13, 24]);
                ops.sub_acc(a, b, res);  
                expect_arrays_closeto(res.data, [2, 4, 6, 8]);
                ops.mul_acc(a, b, res);  
                expect_arrays_closeto(res.data, [12, 44, 36, 88]);
                ops.div_acc(a, b, res);  
                expect_arrays_closeto(res.data, [12.1, 44.1, 36.3, 88.2]);
                ops.pow_acc(a, b, res);  
                expect_arrays_closeto(res.data, [13.1, 1048620.125, 59085.301, 1099511627776]);
            
                a.free();
                b.free();
                res.free();
            });
    
            test("debroadcasting", () => {
                const a = RawTensor.create([2, 3], [1, 2, 3, 4, 5, 6]);
                let res = RawTensor.like(t5).zeros();
                ops.add_acc(a, t5, res);
                expect([...res.data]).toEqual([3, 11, 15]);
                ops.add_acc(a, t5, res);
                expect([...res.data]).toEqual([6, 22, 30]);
                res.zeros();
                ops.mul_acc(a, t5, res);
                expect([...res.data]).toEqual([-5, 14, 27]);
                ops.mul_acc(a, t5, res);
                expect([...res.data]).toEqual([-10, 28, 54]);
                res.free();
                res = RawTensor.scalar(0);
                ops.add_acc(t5, res, res);
                expect(res.item).toEqual(ops.sum(t5));
    
                a.free();
                res.free();
            });
        });
    });

    // const t1 = tensor([2, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    // const t2  = tensor([3, 2], [1, 2, 3, 4, 5, 6]);
    // const t13 = tensor([2, 2, 5, 2], [0.261, 0.983, 0.857, 0.279, 0.211, 0.75, 0.671, 0.32, 0.641, 0.317, 0.003, 0.951, 0.332, 0.226, 0.409, 0.475, 0.348, 0.205, 0.11, 0.353, 0.557, 0.309, 0.06, 0.963, 0.368, 0.607, 0.047, 0.52, 0.5, 0.532, 0.138, 0.686, 0.982, 0.134, 0.984, 0.488, 0.856, 0.766, 0.208, 0.29]);
    // const t14 = tensor([5, 5], [0.93, 0.793, 0.149, 0.827, 0.167, 0.563, 0.485, 0.969, 0.63, 0.909, 0.113, 0.11, 0.627, 0.198, 0.708, 0.574, 0.565, 0.678, 0.013, 0.195, 0.913, 0.131, 0.016, 0.418, 0.277]);

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
        // expect_arrays_closeto(t2.T.matmul(t2).clone().data, [ 35, 44, 44, 56 ]);
        expect_arrays_closeto(ops.matmul(t2.T, t2).clone().data, [ 35, 44, 44, 56 ]);
        expect_arrays_closeto(ops.matmul(t2, t2.T).clone().data, [ 5, 11, 17, 11, 25, 39, 17, 39, 61 ]);

        test_chained_ops(
            t2,
            t => ops.matmul(t.T, t2),
            [35, 44, 44, 56],
            [2, 2],
            [2, 1]
        );

        test_chained_ops(
            t2,
            t => ops.matmul(t.clone().T.clone(), t2.clone()),
            [35, 44, 44, 56],
            [2, 2],
            [2, 1]
        );

        test_chained_ops(
            t1,
            t => ops.add(t, t2.T),
            [2, 5, 8, 6, 9 , 12, 8, 11, 14, 12, 15, 18],
            [2, 2, 3],
            [6, 3, 1]
        );

        test_chained_ops(
            t1,
            // t => t.clone().transpose(0, 2, 1).add(t2),
            t => ops.add(t.clone().transpose(0, 2, 1), t2),
            [2, 6, 5, 9, 8, 12, 8, 12, 11, 15, 14, 18],
            [2, 3, 2],
            [6, 2, 1]
        );

        test_chained_ops(
            t14,
            // t => t.T.matmul(t14).T.clone(),
            t => ops.matmul(t.T, t14).T.clone(),
            [2.358, 1.467, 1.159, 1.535, 1.112, 1.467, 1.213, 1.042, 1.045, 0.798, 1.159, 1.042, 1.814, 0.873, 1.486, 1.535, 1.045, 0.873, 1.295, 0.969, 1.112, 0.798, 1.486, 0.969, 1.47],
            [5, 5],
            [5, 1]
        );

        test_chained_ops(
            t14,
            // t => t.matmul(t14.T).clone(),
            t => ops.matmul(t, t14.T).clone(),
            [2.228, 1.725, 0.568, 1.126, 1.347, 1.725, 2.714, 1.493, 1.44, 1.108, 0.568, 1.493, 0.958, 0.693, 0.406, 1.126, 1.44, 0.693, 1.147, 0.668, 1.347, 1.108, 0.406, 0.668, 1.102],
            [5, 5],
            [5, 1]
        );
    });

    test("reduce operations", () => {
        expect(ops.sum(t14)).toBeCloseTo(11.958);
        expect(ops.sum(t13)).toBeCloseTo(18.697);
        expect(ops.min(t14)).toBeCloseTo(0.013);
        expect(ops.min(t13)).toBeCloseTo(0.003);
        expect(ops.max(t14)).toBeCloseTo(0.969);
        expect(ops.max(t13)).toBeCloseTo(0.985);
        expect(ops.mean(t14)).toBeCloseTo(0.478);
        expect(ops.mean(t13)).toBeCloseTo(0.467);
    });
});
