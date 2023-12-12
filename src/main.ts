// import tensor from './tensor';

// // matrix with 2 rows and 3 columns
// let t1 = tensor([1, 2, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
// let t2 = tensor([2, 2, 4, 4]).rand();

// // // matrix with 3 rows and 2 columns
// // let t2 = new Tensor([3, 2], [1, 2, 3, 4, 5, 6]);

// // for (let mat of t1.get_axis_iterable(0)) {
// //     console.log(mat.toString());
// // }

// // for (let mat of t1.get_axis_iterable(1)) {
// //     console.log(mat.toString());
// // }

// console.log(t1.toString());
// console.log(t2.toString())

import core_ready from './core';

core_ready.then(({ core, memory }) => {
    const arr_size = 4;
    const arr = new Float32Array(
        memory.buffer, core._alloc_farr(arr_size), arr_size);

    core._rand_seed(1319);

    // NOTE
    // arr.byteOffset is the pointer of the array within the memory object
    // arr.length is the number of items in the array

    arr.set([3, 2, 1, 0]);
    // core._rand_i(arr.byteOffset, arr.length, -1, 1);
    // core._rand_f(arr.byteOffset, arr.length, -1, 1);
    // core._scl_add(arr.byteOffset, 3, arr.length);
    // core._prw_add(arr.byteOffset, arr.byteOffset, 8); // todo: figure out if func(data, shape, parameter) is more reasonable
    console.log(arr)
});
