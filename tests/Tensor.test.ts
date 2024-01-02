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

    let t1 = tensor([2, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    let t2 = tensor([3, 2],    [1, 2, 3, 4, 5, 6]);
    let t3 = tensor([3, 2],    [-100, 2, 3, 2, 4, 2]);
    let t4 = tensor([3, 2],    [7.5, 5.5, -2, 3.5, 0, 3]);
    let t5 = tensor([3],       [-1, 2, 3]);
    let t6 = tensor([2, 3],    [-100, 2, 3, 2, 4, 2]);

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
        binary_op: ops.BinaryOp<Tensor> | ops.BinaryOp<Tensor | number>,
        a: Tensor, b: Tensor,
        expected_data: number[],
        expected_shape: Shape | number[],
        in_place = false
    ) {
        let t = a.clone();

        t = binary_op(t, b, in_place);
        expect([...t.shape]).toEqual([...expected_shape]);

        [...t.data].map((v, i) => {
            if (Number.isNaN(v)) expect(expected_data[i]).toBeNaN();
            else expect(v).toBeCloseTo(expected_data[i]);
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
            binary(ops.add, t1, t5, [0, 4, 6, 3, 7, 9, 6, 10, 12, 9, 13, 15], t1.shape, true);
            binary(ops.sub, t1, t5, [2, 0, 0, 5, 3, 3, 8, 6, 6, 11, 9, 9], t1.shape);
            binary(ops.mul, t1, t5, [-1, 4, 9, -4, 10, 18, -7, 16, 27, -10, 22, 36], t1.shape, true);
            binary(ops.div, t1, t5, [-1, 1, 1, -4, 2.5, 2, -7, 4, 3, -10, 5.5, 4], t1.shape, true);
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
            let t;

            t = t1.add(1);
            expect([...t.data]).toEqual([...t1.data.map(v => v + 1)]);
            t.free();
    
            t = t1.sub(2.5);
            expect([...t.data]).toEqual([...t1.data.map(v => v - 2.5)]);
            t.free();

            t = t1.mul(2);
            expect([...t.data]).toEqual([...t1.data.map(v => v * 2)]);
            t.free();

            t = t1.div(4);
            expect([...t.data]).toEqual([...t1.data.map(v => v / 4)]);
            t.free();
        });

        // warning: this test can cause precision-based errors!
        test("pairwise ops", () => {
            let t;
            let expected: number[];

            t = t2.add(t3);
            expect([...t2.shape]).toEqual([...t.shape]);
            expected = [...t2.data].map((v, i) => v + t3.data[i]);
            expect([...t.data]).toEqual(expected);
            t.free();

            t = t2.sub(t4);
            expected = [...t2.data].map((v, i) => v - t4.data[i]);
            expect([...t.data]).toEqual(expected);
            t.free();

            t = t2.mul(t3);
            expected = [...t2.data].map((v, i) => v * t3.data[i]);
            t.data.forEach((v, i) => expect(v).toBeCloseTo(expected[i]));
            t.free();

            t = t2.div(t4);
            expected = [...t2.data].map((v, i) => v / t4.data[i]);
            t.data.forEach((v, i) => expect(v).toBeCloseTo(expected[i]));
            t.free();
        });

        test("broadcasting ops", () => {
            binary(ops.add, t1, t5, [0, 4, 6, 3, 7, 9, 6, 10, 12, 9, 13, 15], t1.shape);
            binary(ops.sub, t1, t5, [2, 0, 0, 5, 3, 3, 8, 6, 6, 11, 9, 9], t1.shape);
            binary(ops.mul, t1, t5, [-1, 4, 9, -4, 10, 18, -7, 16, 27, -10, 22, 36], t1.shape);
            binary(ops.div, t1, t5, [-1, 1, 1, -4, 2.5, 2, -7, 4, 3, -10, 5.5, 4], t1.shape);
        });

        // let t1 = tensor([2, 2, 3],   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        // let t2 = tensor([3, 2],      [1, 2, 3, 4, 5, 6]);
        // let t3 = tensor([3, 2],      [-100, 2, 3, 2, 4, 2]);
        // let t4 = tensor([3, 2],      [7.5, 5.5, -2, 3.5, 0, 3]);
        // let t5 = tensor([3],         [-1, 2, 3]);
        // let t6 = tensor([2, 3],      [-100, 2, 3, 2, 4, 2]);
        let t7 = tensor([2, 3, 4], [8, 4, 4, 2, 3, 1, 4, 9, 2, 0, 9, 0, 4, 5, 1, 3, 2, 3, 4, 5, 8, 1, 2, 3]);
        let t8 = tensor([4, 3],    [0, 2, 3, 0, 3, 6, 5, 7, 3, 1, 2, 1]);
        let t9 = tensor([2, 2, 4, 3], [0.82, 0.447, 0.716, 0.057, 0.245, 0.855, 0.288, 0.902, 0.162, 0.091, 0.225, 0.892, 0.808, 0.924, 0.967, 0.084, 0.623, 0.686, 0.042, 0.358, 0.54, 0.54, 0.195, 0.298, 0.783, 0.477, 0.117, 0.894, 0.927, 0.532, 0.184, 0.86, 0.543, 0.57, 0.719, 0.72, 0.184, 0.989, 0.863, 0.784, 0.143, 0.156, 0.403, 0.187, 0.304, 0.824, 0.514, 0.731]);
        let t10 = tensor([3, 1], [0.021, 0.782, 0.253]);

        test("matmul", () => {
            expect(() => ops.matmul(t1, t5, false)).toThrow();
            expect(() => ops.matmul(t4, t5, false)).toThrow();

            binary(ops.matmul, t1, t2,  [22, 28, 49, 64, 76, 100, 103, 136], [2, 2, 2]);
            binary(ops.matmul, t4, t6,  [-739.0, 37.0, 33.5, 207.0, 10.0, 1.0, 6.0, 12.0, 6.0], [3, 3]);
            binary(ops.matmul, t1, t2,  [22, 28, 49, 64, 76, 100, 103, 136], [2, 2, 2]);
            binary(ops.matmul, t7, t8,  [22, 60, 62, 29, 55, 36, 45, 67, 33, 8, 36, 48, 25, 51, 41, 13, 39, 39], [2, 3, 3]);
            binary(ops.matmul, t9, t10, [0.548, 0.409, 0.752, 0.404, 0.984, 0.663, 0.417, 0.239, 0.419, 0.878, 0.814, 0.756, 0.996, 0.168, 0.232, 0.604], [2, 2, 4, 1]);

            // todo: potentially add tests with large unit-matrices (easy to validate without other libraries)
        });
    });
});
