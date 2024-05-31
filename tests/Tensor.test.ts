import { describe, expect, test } from "bun:test";
import tensor, { Tensor, tensor_like } from "../src/base/Tensor";
import { core_ready } from "../src/base/Management";
import Shape from "../src/base/Shape";
import Strides from "../src/base/Strides";

describe("tensor creation", async () => {
    await core_ready;

    test("tensor()", () => {
        // creates a tensor through tensor() and ensures that the created
        // tensor matches the expectation
        function init_and_validate_tensor(shape: Shape | number[], strides: number[], data?: number[]) {
            // create tensor
            const new_tensor = tensor(shape, data);

            // @ts-expect-error Obscure signature incompatibility between Int32Array.reduce() and Array.reduce()
            const expected_nelem = shape.reduce((acc: number, val: number) => acc * val, 1);

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
        // new_tensor.set_rank(50); // todo
        // new_tensor.set_nelem(50);

        // modify old tensor
        old_tensor.shape[0] = 10;
        old_tensor.strides[0] = 10;
        old_tensor.data[0] = 10;
        // old_tensor.set_rank(10);
        // old_tensor.set_nelem(10);

        // check that new tensor was not changed
        expect([...new_tensor.shape]).not.toEqual([...old_tensor.shape]);
        expect([...new_tensor.strides]).not.toEqual([...old_tensor.strides]);
        expect([...new_tensor.data]).not.toEqual([...old_tensor.data]);
        // expect(new_tensor.get_rank()).not.toEqual(old_tensor.get_rank());
        // expect(new_tensor.get_nelem()).not.toEqual(old_tensor.get_nelem());
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

    // basic tests for tensor address space overlaps
    // not really tests for views
    test("Tensor views/data-dependence", () => {
        const old_tensor = tensor([1, 2, 3, 4]).zeros();
        const new_tensor = old_tensor.create_view();

        // make sure metadata is copied correctly and that tensors are independent of one another
        check_tensor_metadata_equality(new_tensor, old_tensor);
        old_tensor.data[0] = 4;
        expect(old_tensor.data[0]).toEqual(new_tensor.data[0]);
    });

    const t1 = tensor([2, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const t2 = tensor([3, 2], [1, 2, 3, 4, 5, 6]);
    const t3 = tensor([3, 2], [-100, 2, 3, 2, 4, 2]);
    const t4 = tensor([3, 2], [7.5, 5.5, -2, 3.5, 0, 3]);
    const t5 = tensor([3], [-1, 2, 3]);
    const t6 = tensor([2, 3], [-100, 2, 3, 2, 4, 2]);
    const t7 = tensor([2, 3, 4], [8, 4, 4, 2, 3, 1, 4, 9, 2, 0, 9, 0, 4, 5, 1, 3, 2, 3, 4, 5, 8, 1, 2, 3]);

    function expect_shape(t: Tensor, ...shape: number[]) {
        expect([...t.shape]).toEqual(shape);
    }

    describe("views", () => {
        test("get_axis_iterable", () => {
            // check that subtensors have appropriate shapes
            for (const e of t1.get_axis_iterable(0)) {
                expect_shape(e, 2, 3);
                expect_shape(e.T, 3, 2);
            }

            for (const e of t1.get_axis_iterable(1)) {
                expect_shape(e, 3);
            }

            // todo stride

            // things we should test:
            // - proper handling of strides of axis_iterable
            // - proper offsets in get_axis_iterable
            // - transposition correctness (shape, stride, offset)
            // - views of views
        });
    });
});
