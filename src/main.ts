import core from './core/build';
import { core_ready } from './util';
import tensor from './tensor';

core_ready.then(() => {

    // let t1 = tensor([1, 2, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);    
    // let t2 = tensor([3, 2], [1, 2, 3, 4, 5, 6]);
    // let t3 = tensor([2, 4], [1, 2, 3, 2, 4, 2, 4, 1]);


    let t1 = tensor([400, 400]);
    let t2 = tensor([400, 400]);

    t1.rand_f();
    t2.rand_f();

    console.time();
    for (let i = 0; i < 100; i++) t1.mul(t2).free();
    console.timeEnd();

    // // for (let mat of t1.get_axis_iterable(1)) {
    // //     console.log(mat.toString());
    // // }
    
    
    // const arr_size = 4;
    // const arr = new Float32Array(
    //     core.memory.buffer, core._alloc_farr(arr_size), arr_size);

    // core._rand_seed(1319);

    // // NOTE
    // // arr.byteOffset is the pointer of the array within the memory object
    // // arr.length is the number of items in the array

    // arr.set([3, 2, 1, 0]);
    // // core._rand_i(arr.byteOffset, arr.length, -1, 1);
    // // core._rand_f(arr.byteOffset, arr.length, -1, 1);
    // // core._scl_add(arr.byteOffset, 3, arr.length);
    // // core._prw_add(arr.byteOffset, arr.byteOffset, 8); // todo: figure out if func(data, shape, parameter) is more reasonable
    // console.log(arr)
});