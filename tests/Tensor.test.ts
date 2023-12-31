import { describe, expect, test } from "bun:test";
import tensor, { Tensor, tensor_like } from "../src/Tensor";
import { core_ready } from "../src/util";
import Shape from "../src/Shape";
import Strides from "../src/Strides";

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

    describe("in-place", () => {
        test("unary ops", () => {
            let t;

            t = t5.clone().relu();
            expect([...t.data]).toEqual([0, 2, 3]);
            t.free();

            t = t5.clone().binstep();
            expect([...t.data]).toEqual([0, 1, 1]);
            t.free();

            t = t5.clone().logistic();
            expect([...t.data]).toEqual([0.2689414322376251, 0.8807970881462097, 0.9525741338729858]);
            t.free();

            // todo: add proper values in equals expectation
            // todo: fix sigmoid. seems to be implemented incorrectly
            t = t5.clone().sigmoid();
            console.log(t.data)
            expect([...t.data]).toEqual([0.2689414322376251, 0.8807970881462097, 0.9525741338729858]);
            t.free();

            t = t5.clone().negate();
            expect([...t.data]).toEqual([1, -2, -3]);
            t.free();
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
    });

    describe("out-of-place", () => {
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
    });
});
