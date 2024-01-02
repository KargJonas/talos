import { describe, expect, test } from "bun:test";
import tensor, { Tensor, tensor_like } from "../src/Tensor";
import { core_ready } from "../src/util";
import Shape from "../src/Shape";
import Strides from "../src/Strides";
import * as ops from "../src/tensor_operations";

describe("tensor creation", async () => {
    await core_ready;

    test("tensor()", () => {
        // creates a tensor through tensor() and ensures that the created
        // tensor matches the expectation
        function init_and_validate_tensor(shape: Shape | number[], strides: number[], data?: number[]) {
            // create tensor
            let new_tensor = tensor(shape, data);
            let expected_nelem = shape.reduce((acc: number, val: number) => acc * val, 1);

            // check view types
            expect(new_tensor).toBeInstanceOf(Tensor);
            expect(new_tensor.shape).toBeInstanceOf(Shape);
            expect(new_tensor.strides).toBeInstanceOf(Strides);
            expect(new_tensor.data).toBeInstanceOf(Float32Array);

            // ensure that get_rank() and get_nelem() return correct values
            expect(new_tensor.get_rank()).toBe(shape.length);
            expect(new_tensor.get_nelem()).toBe(expected_nelem);
            if (data !== undefined) expect(data.length).toBe(expected_nelem);

            // ensure that the views have the correct sizes
            expect(new_tensor.shape.length).toBe(shape.length);
            expect(new_tensor.strides.length).toBe(shape.length);
            expect(new_tensor.data.length).toBe(expected_nelem);

            // ensure that the shape array is populated with correct data
            expect([...new_tensor.shape]).toEqual([...shape]);
            expect([...new_tensor.strides]).toEqual([...strides]);
            if (data !== undefined) expect([...new_tensor.data]).toEqual([...data]);

            new_tensor.free();
        }

        // create and validate tensors of various shapes from arrays
        init_and_validate_tensor([3, 7, 2, 8],          [112, 16, 8, 1]);
        init_and_validate_tensor([3, 7, 2, 8],          [112, 16, 8, 1],    (new Array(3 * 7 * 2 * 8)).fill(1));
        init_and_validate_tensor([645],                 [1],                (new Array(645)).fill(1));
        init_and_validate_tensor([1, 645],              [645, 1],           (new Array(1 * 645)).fill(1));
        init_and_validate_tensor([1, 78],               [78, 1],            (new Array(1 * 78)).fill(1));
        init_and_validate_tensor([1, 1, 1, 1, 1, 1],    [1, 1, 1, 1, 1, 1], (new Array(1 * 1 * 1 * 1 * 1 * 1)).fill(1));
        init_and_validate_tensor([1, 1, 1, 0, 1, 1],    [0, 0, 0, 1, 1, 1], (new Array(1 * 1 * 1 * 0 * 1 * 1)).fill(1));

        const new_shape = new Shape([3, 7, 2, 8]);
        init_and_validate_tensor(new_shape,          [112, 16, 8, 1]);
        init_and_validate_tensor(new_shape,          [112, 16, 8, 1],    (new Array(3 * 7 * 2 * 8)).fill(1));
    });

    // ensures that modifying a tensor does not cause changes in another tensor
    function check_tensor_independence(old_tensor: Tensor, new_tensor: Tensor) {
        expect(new_tensor.get_view_ptr()).not.toBe(old_tensor.get_view_ptr());
        expect(new_tensor.get_shape_ptr()).not.toBe(old_tensor.get_shape_ptr());
        expect(new_tensor.get_strides_ptr()).not.toBe(old_tensor.get_strides_ptr());
        expect(new_tensor.get_data_ptr()).not.toBe(old_tensor.get_data_ptr());

        // modify new tensor
        new_tensor.shape[0] = 50;
        new_tensor.strides[0] = 50;
        new_tensor.data[0] = 50;
        new_tensor.set_rank(50);
        new_tensor.set_nelem(50);

        // modify old tensor
        old_tensor.shape[0] = 10;
        old_tensor.strides[0] = 10;
        old_tensor.data[0] = 10;
        old_tensor.set_rank(10);
        old_tensor.set_nelem(10);

        // check that new tensor was not changed
        expect([...new_tensor.shape]).not.toEqual([...old_tensor.shape]);
        expect([...new_tensor.strides]).not.toEqual([...old_tensor.strides]);
        expect([...new_tensor.data]).not.toEqual([...old_tensor.data]);
        expect(new_tensor.get_rank()).not.toEqual(old_tensor.get_rank());
        expect(new_tensor.get_nelem()).not.toEqual(old_tensor.get_nelem());
    }

    function check_tensor_metadata_equality(a: Tensor, b: Tensor) {
        expect(a.get_rank()).toBe(b.get_rank());
        expect(a.get_nelem()).toBe(b.get_nelem());
        expect([...a.shape]).toEqual([...b.shape]);
        expect([...a.strides]).toEqual([...b.strides]);
    }

    test("tensor_like()", () => {
        let old_tensor = tensor([1, 2, 3, 4], (new Array(1 * 2 * 3 * 4)).fill(1));
        let new_tensor = tensor_like(old_tensor);

        check_tensor_metadata_equality(new_tensor, old_tensor);
        check_tensor_independence(old_tensor, new_tensor);

        old_tensor.free();
        new_tensor.free();

        // make sure that data was *not* copied to the new tensor        
        old_tensor = tensor([1, 2, 3, 4]).rand();
        new_tensor = tensor_like(old_tensor);
        expect([...old_tensor.data]).not.toEqual([...new_tensor.data]);

        old_tensor.free();
        new_tensor.free();
    });

    test("Tensor.clone()", () => {
        let old_tensor = tensor([1, 2, 3, 4], (new Array(1 * 2 * 3 * 4)).fill(1));
        let new_tensor = old_tensor.clone();

        // make sure metadata is copied correctly and that tensors are independent of one another
        check_tensor_metadata_equality(new_tensor, old_tensor);
        check_tensor_independence(old_tensor, new_tensor);

        old_tensor.free();
        new_tensor.free();

        // make sure that data *was* copied to the new tensor properly      
        old_tensor = tensor([1, 2, 3, 4]).rand();
        new_tensor = old_tensor.clone();
        expect([...old_tensor.data]).toEqual([...new_tensor.data]);

        old_tensor.free();
        new_tensor.free();
    });

    test("Tensor referencing/(meta-)data-dependence", () => {
        let old_tensor = tensor([1, 2, 3, 4]).zeros();
        let new_tensor = new Tensor(old_tensor.shape, old_tensor.strides, old_tensor.data);

        // make sure metadata is copied correctly and that tensors are independent of one another
        check_tensor_metadata_equality(new_tensor, old_tensor);
        
        old_tensor.shape[0] = 4;
        old_tensor.strides[0] = 4;
        old_tensor.data[0] = 4;
        old_tensor.set_rank(6);
        old_tensor.set_nelem(0);

        expect(old_tensor.shape[0]).toEqual(new_tensor.shape[0]);
        expect(old_tensor.strides[0]).toEqual(new_tensor.strides[0]);
        expect(old_tensor.data[0]).toEqual(new_tensor.data[0]);
        expect(old_tensor.get_rank()).not.toEqual(new_tensor.get_rank()); // nelem and rank are not dependent (might change this in the future)
        expect(old_tensor.get_nelem()).not.toEqual(new_tensor.get_nelem());
    });
});

describe("tensor operations", async () => {
    await core_ready;

    let t1  = tensor([2, 2, 3],     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    let t2  = tensor([3, 2],        [1, 2, 3, 4, 5, 6]);
    let t3  = tensor([3, 2],        [-100, 2, 3, 2, 4, 2]);
    let t4  = tensor([3, 2],        [7.5, 5.5, -2, 3.5, 0, 3]);
    let t5  = tensor([3],           [-1, 2, 3]);
    let t6  = tensor([2, 3],        [-100, 2, 3, 2, 4, 2]);
    let t7  = tensor([2, 3, 4],     [8, 4, 4, 2, 3, 1, 4, 9, 2, 0, 9, 0, 4, 5, 1, 3, 2, 3, 4, 5, 8, 1, 2, 3]);
    let t8  = tensor([4, 3],        [0, 2, 3, 0, 3, 6, 5, 7, 3, 1, 2, 1]);
    let t9  = tensor([2, 2, 4, 3],  [0.82, 0.447, 0.716, 0.057, 0.245, 0.855, 0.288, 0.902, 0.162, 0.091, 0.225, 0.892, 0.808, 0.924, 0.967, 0.084, 0.623, 0.686, 0.042, 0.358, 0.54, 0.54, 0.195, 0.298, 0.783, 0.477, 0.117, 0.894, 0.927, 0.532, 0.184, 0.86, 0.543, 0.57, 0.719, 0.72, 0.184, 0.989, 0.863, 0.784, 0.143, 0.156, 0.403, 0.187, 0.304, 0.824, 0.514, 0.731]);
    let t10 = tensor([3, 1],        [0.021, 0.782, 0.253]);
    let t11 = tensor([3, 4, 5],     [0.316, 0.057, 0.639, 0.295, 0.726, 0.135, 0.742, 0.346, 0.789, 0.503, 0.745, 0.907, 0.91, 0.239, 0.999, 0.372, 0.118, 0.414, 0.275, 0.76, 0.98, 0.574, 0.886, 0.247, 0.259, 0.574, 0.129, 0.546, 0.508, 0.403, 0.097, 0.072, 0.286, 0.141, 0.37, 0.153, 0.585, 0.994, 0.399, 0.74, 0.063, 0.904, 0.384, 0.158, 0.904, 0.478, 0.237, 0.714, 0.732, 0.231, 0.814, 0.88, 0.91, 0.764, 0.778, 0.912, 0.764, 0.977, 0.158, 0.493]);
    let t12 = tensor([3, 5, 6],     [0.084, 0.078, 0.868, 0.891, 0.331, 0.668, 0.829, 0.305, 0.899, 0.636, 0.855, 0.854, 0.627, 0.078, 0.884, 0.297, 0.52, 0.722, 0.248, 0.663, 0.353, 0.68, 0.298, 0.384, 0.929, 0.714, 0.273, 0.737, 0.644, 0.072, 0.304, 0.755, 0.405, 0.676, 0.597, 0.55, 0.785, 0.333, 0.26, 0.385, 0.895, 0.062, 0.396, 0.904, 0.518, 0.316, 0.839, 0.581, 0.057, 0.555, 0.101, 0.986, 0.348, 0.549, 0.733, 0.629, 0.745, 0.723, 0.752, 0.515, 0.501, 0.148, 0.816, 0.901, 0.534, 0.748, 0.197, 0.581, 0.747, 0.295, 0.048, 0.488, 0.727, 0.313, 0.39, 0.652, 0.645, 0.686, 0.21, 0.18, 0.557, 0.779, 0.926, 0.272, 0.005, 0.539, 0.023, 0.208, 0.243, 0.39]);
    let t13 = tensor([2, 2, 5, 2],  [0.261, 0.983, 0.857, 0.279, 0.211, 0.75, 0.671, 0.32, 0.641, 0.317, 0.003, 0.951, 0.332, 0.226, 0.409, 0.475, 0.348, 0.205, 0.11, 0.353, 0.557, 0.309, 0.06, 0.963, 0.368, 0.607, 0.047, 0.52, 0.5, 0.532, 0.138, 0.686, 0.982, 0.134, 0.984, 0.488, 0.856, 0.766, 0.208, 0.29]);
    let t14 = tensor([5, 5],        [0.93, 0.793, 0.149, 0.827, 0.167, 0.563, 0.485, 0.969, 0.63, 0.909, 0.113, 0.11, 0.627, 0.198, 0.708, 0.574, 0.565, 0.678, 0.013, 0.195, 0.913, 0.131, 0.016, 0.418, 0.277]);
    let t15 = tensor([5, 1],        [0.891, 0.549, 0.65, 0.02, 0.676]);

    function unary(unary_op: ops.UnaryOp, a: Tensor, expectation: (v: number) => number, in_place = false) {
        let t = a.clone();
        const input = [...t.data];

        t = unary_op(t, in_place);
        expect([...t.shape]).toEqual([...a.shape]);

        [...t.data].map((v, i) => {
            if (Number.isNaN(v)) expect(expectation(input[i])).toBeNaN();
            else expect(v).toBeCloseTo(expectation(input[i]));
        });

        t.free();
    }

    function binary(
        binary_op: ops.BinaryOp<Tensor | number> | ops.BinaryOp<Tensor>,
        a: Tensor, b: Tensor | number,
        expected_shape: Shape | number[],
        expected_data: number[],
        in_place = false
    ) {
        let t = a.clone();

        // @ts-ignore
        t = binary_op(t, b, in_place);
        expect([...t.shape]).toEqual([...expected_shape]);
        
        [...t.data].map((v, i) => {
            if (Number.isNaN(v)) expect(expected_data[i]).toBeNaN();
            else expect(v).toBeCloseTo(expected_data[i], 3);
        });

        t.free();
    }
    
    describe("in-place", () => {
        test("unary ops", () => {
            unary(ops.relu, t5, v => v < 0 ? 0 : v, true);
            unary(ops.binstep, t5, v => v < 0 ? 0 : 1, true);
            unary(ops.logistic, t5, v => 1 / (Math.exp(-v) + 1), true);
            unary(ops.negate, t5, v => -v, true);
            unary(ops.identity, t5, v => v, true);

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

            t.add(1, true);
            expected = expected.map(v => v + 1);
            expect([...t.data]).toEqual(expected);
    
            t.sub(2.5, true);
            expected = expected.map(v => v - 2.5);
            expect([...t.data]).toEqual(expected);

            t.mul(2, true);
            expected = expected.map(v => v * 2);
            expect([...t.data]).toEqual(expected);

            t.div(4, true);
            expected = expected.map(v => v / 4);
            expect([...t.data]).toEqual(expected);

            expect([...t1.shape]).toEqual([...t.shape]);
        });

        // warning: this test can cause precision-based errors!
        test("pairwise ops", () => {
            const t = t2.clone();
            let expected = [...t.data];

            t.add(t3, true);
            expected = expected.map((v, i) => v + t3.data[i]);
            expect([...t.data]).toEqual(expected);

            t.sub(t4, true);
            expected = expected.map((v, i) => v - t4.data[i]);
            expect([...t.data]).toEqual(expected);

            t.mul(t3, true);
            expected = expected.map((v, i) => v * t3.data[i]);
            // expect([...t.data]).toEqual(expected);
            t.data.forEach((v, i) => expect(v).toBeCloseTo(expected[i]));

            t.div(t2, true);
            expected = expected.map((v, i) => v / t2.data[i]);
            t.data.forEach((v, i) => expect(v).toBeCloseTo(expected[i]));
        });

        test("broadcasting ops", () => {
            binary(ops.add, t1, t5, t1.shape, [0, 4, 6, 3, 7, 9, 6, 10, 12, 9, 13, 15], true);
            binary(ops.sub, t1, t5, t1.shape, [2, 0, 0, 5, 3, 3, 8, 6, 6, 11, 9, 9]);
            binary(ops.mul, t1, t5, t1.shape, [-1, 4, 9, -4, 10, 18, -7, 16, 27, -10, 22, 36], true);
            binary(ops.div, t1, t5, t1.shape, [-1, 1, 1, -4, 2.5, 2, -7, 4, 3, -10, 5.5, 4], true);
        });

        test("matmul", () => {
            expect(() => ops.matmul(t1,  t2,  true)).toThrow();
            expect(() => ops.matmul(t4,  t6,  true)).toThrow();
            expect(() => ops.matmul(t7,  t8, true)).toThrow();
            expect(() => ops.matmul(t9,  t10, true)).toThrow();

            binary(ops.matmul, t14, t15, [5, 1], [1.49, 2.025, 1.051, 1.394, 1.091]);
            // more thorough testing in out-of-place tests
        });

        test("dot", () => {
            expect(() => ops.dot(t1, t5, true)).toThrow();
            expect(() => ops.dot(t4, t5, true)).toThrow();

            binary(ops.matmul, t14, t15, [5, 1], [1.49, 2.025, 1.051, 1.394, 1.091]);
            // more thorough testing in out-of-place tests
        });
    });

    describe("out-of-place", () => {
        test("unary ops", () => {
            unary(ops.relu, t5, v => v < 0 ? 0 : v);
            unary(ops.binstep, t5, v => v < 0 ? 0 : 1);
            unary(ops.logistic, t5, v => 1 / (Math.exp(-v) + 1));
            unary(ops.negate, t5, v => -v);
            unary(ops.identity, t5, v => v);

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

        test("scalar ops", () => {
            binary(ops.add, t1, 1,   t1.shape, [...t1.data].map(v => v + 1));
            binary(ops.sub, t1, 2.5, t1.shape, [...t1.data].map(v => v - 2.5));
            binary(ops.mul, t1, 2,   t1.shape, [...t1.data].map(v => v * 2));
            binary(ops.div, t1, 4,   t1.shape, [...t1.data].map(v => v / 4));
        });

        // warning: this test can cause precision-based errors!
        test("pairwise ops", () => {
            let t;
            let expected: number[];

            binary(ops.add, t2, t3, t2.shape, [...t2.data].map((v, i) => v + t3.data[i]));
            binary(ops.sub, t2, t4, t2.shape, [...t2.data].map((v, i) => v - t4.data[i]));
            binary(ops.mul, t2, t3, t2.shape, [...t2.data].map((v, i) => v * t3.data[i]));
            binary(ops.div, t2, t4, t2.shape, [...t2.data].map((v, i) => v / t4.data[i]));
        });

        test("broadcasting ops", () => {
            binary(ops.add, t1, t5, t1.shape, [0, 4, 6, 3, 7, 9, 6, 10, 12, 9, 13, 15]);
            binary(ops.sub, t1, t5, t1.shape, [2, 0, 0, 5, 3, 3, 8, 6, 6, 11, 9, 9]);
            binary(ops.mul, t1, t5, t1.shape, [-1, 4, 9, -4, 10, 18, -7, 16, 27, -10, 22, 36]);
            binary(ops.div, t1, t5, t1.shape, [-1, 1, 1, -4, 2.5, 2, -7, 4, 3, -10, 5.5, 4]);
        });

        test("matmul", () => {
            expect(() => ops.matmul(t1, t5)).toThrow();
            expect(() => ops.matmul(t4, t5)).toThrow();
            expect(() => ops.matmul(t11, t13)).toThrow();

            binary(ops.matmul, t1,  t2,  [2, 2, 2],    [22, 28, 49, 64, 76, 100, 103, 136]);
            binary(ops.matmul, t4,  t6,  [3, 3],       [-739.0, 37.0, 33.5, 207.0, 10.0, 1.0, 6.0, 12.0, 6.0]);
            binary(ops.matmul, t7,  t8,  [2, 3, 3],    [22, 60, 62, 29, 55, 36, 45, 67, 33, 8, 36, 48, 25, 51, 41, 13, 39, 39]);
            binary(ops.matmul, t9,  t10, [2, 2, 4, 1], [0.548, 0.409, 0.752, 0.404, 0.984, 0.663, 0.417, 0.239, 0.419, 0.878, 0.814, 0.756, 0.996, 0.168, 0.232, 0.604]);
            binary(ops.matmul, t11, t12, [3, 4, 6],    [1.222, 0.806, 1.193, 1.243, 1.041, 0.887, 1.506, 1.146, 1.506, 1.602, 1.418, 1.313, 2.372, 1.277, 2.624, 2.41, 2.21, 2.093, 1.163, 0.822, 1.1, 1.277, 1.011, 0.808, 1.303, 2.032, 1.223, 1.594, 2.123, 1.358, 0.816, 1.505, 0.9, 1.402, 1.396, 1.127, 0.479, 0.667, 0.496, 0.59, 0.69, 0.492, 1.465, 1.896, 1.321, 1.571, 2.144, 1.298, 0.527, 1.17, 0.985, 0.885, 0.691, 1.147, 0.96, 0.688, 1.259, 1.584, 1.461, 1.252, 1.407, 1.473, 2.12, 2.343, 1.96, 2.174, 1.353, 1.179, 1.795, 1.91, 1.42, 1.96]);
        });

        test("dot", () => {
            expect(() => ops.dot(t1, t5)).toThrow();
            expect(() => ops.dot(t4, t5)).toThrow();

            binary(ops.dot, t1, t2,   [2, 2, 2],        [22, 28, 49, 64, 76, 100, 103, 136]);
            binary(ops.dot, t4, t6,   [3, 3],           [-739.0, 37.0, 33.5, 207.0, 10.0, 1.0, 6.0, 12.0, 6.0]);
            binary(ops.dot, t1, t2,   [2, 2, 2],        [22, 28, 49, 64, 76, 100, 103, 136]);
            binary(ops.dot, t7, t8,   [2, 3, 3],        [22, 60, 62, 29, 55, 36, 45, 67, 33, 8, 36, 48, 25, 51, 41, 13, 39, 39]);
            binary(ops.dot, t9, t10,  [2, 2, 4, 1],     [0.548, 0.409, 0.752, 0.404, 0.984, 0.663, 0.417, 0.239, 0.419, 0.878, 0.814, 0.756, 0.996, 0.168, 0.232, 0.604]);
            binary(ops.dot, t11, t13, [3, 4, 2, 2, 2],  [0.929, 1.13, 0.464, 0.934, 0.791, 1.08, 1.132, 0.973, 1.596, 1.011, 0.718, 0.8, 0.536, 1.644, 1.868, 1.111, 1.964, 2.061, 0.869, 1.747, 1.315, 2.312, 2.301, 1.549, 0.957, 1.038, 0.389, 0.902, 0.76, 1.027, 0.968, 0.904, 1.266, 1.949, 0.67, 1.625, 1.047, 1.66, 1.836, 1.446, 0.975, 1.3, 0.489, 1.081, 0.754, 1.112, 1.262, 1.183, 0.479, 0.492, 0.231, 0.404, 0.355, 0.543, 0.563, 0.431, 1.493, 1.421, 0.821, 1.093, 0.875, 1.815, 2.069, 1.189, 1.558, 0.939, 0.612, 0.798, 0.69, 1.686, 1.598, 0.735, 1.118, 1.379, 0.652, 1.079, 0.693, 1.313, 1.676, 1.336, 2.17, 2.219, 1.018, 1.836, 1.266, 2.463, 2.688, 1.931, 1.521, 2.049, 0.765, 1.71, 1.167, 1.955, 2.075, 1.469]);
        });
    });
});

// todo: add test case for broadcasting where result shape is different from both input tensors

// todo: potentially add tests with large identity-matrices (easy to validate without other libraries)

// todo: add large rand test that checks that there are no zero elements in a tensor (good indicator for some bugs)

// todo: add tests for finding memory leaks