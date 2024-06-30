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

            // ensure that rank and get_nelem() return correct values
            expect(new_tensor.rank).toBe(shape.length);
            expect(new_tensor.nelem).toBe(expected_nelem);
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
        expect(new_tensor.ptr).not.toBe(old_tensor.ptr);
        expect(new_tensor.shape_ptr).not.toBe(old_tensor.shape_ptr);
        expect(new_tensor.strides_ptr).not.toBe(old_tensor.strides_ptr);
        expect(new_tensor.data_ptr).not.toBe(old_tensor.data_ptr);

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
        // expect(new_tensor.rank).not.toEqual(old_tensor.rank);
        // expect(new_tensor.nelem).not.toEqual(old_tensor.nelem);
    }

    function check_tensor_metadata_equality(a: Tensor, b: Tensor) {
        expect(a.rank).toBe(b.rank);
        expect(a.nelem).toBe(b.nelem);
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
    const t13 = tensor([2, 2, 5, 2], [0.261, 0.983, 0.857, 0.279, 0.211, 0.75, 0.671, 0.32, 0.641, 0.317, 0.003, 0.951, 0.332, 0.226, 0.409, 0.475, 0.348, 0.205, 0.11, 0.353, 0.557, 0.309, 0.06, 0.963, 0.368, 0.607, 0.047, 0.52, 0.5, 0.532, 0.138, 0.686, 0.982, 0.134, 0.984, 0.488, 0.856, 0.766, 0.208, 0.29]);
    const t14 = tensor([5, 5], [0.93, 0.793, 0.149, 0.827, 0.167, 0.563, 0.485, 0.969, 0.63, 0.909, 0.113, 0.11, 0.627, 0.198, 0.708, 0.574, 0.565, 0.678, 0.013, 0.195, 0.913, 0.131, 0.016, 0.418, 0.277]);

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

function expect_shape(t: Tensor, ...shape: number[]) {
    expect([...t.shape]).toEqual(shape);
}

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

    expect([...result.shape]).toEqual([...expected_shape]);
    expect([...result.strides]).toEqual([...expected_strides]);
    expect_arrays_closeto(result.data, expected_data);
}