import { core_ready } from '../src/util';
import tensor from '../src/Tensor';

const rand_int = (min: number, max: number) => ((Math.random() * (max - min)) | 0) + min;

core_ready.then(() => {
    console.log('###########\n'.repeat(2));

    const start = Date.now();
    let t1, t2;

    console.log("--- Scalar [large] [in-place] ---");
    console.time();
    t1 = tensor([10, 200, 200]).rand();
    for (let i = 0; i < 100; i++) t1.mul(Math.random(), true);
    t1.free();
    console.timeEnd();

    console.log("--- Scalar [large] [out-of-place] ---");
    console.time();
    t1 = tensor([10, 200, 200]).rand();
    for (let i = 0; i < 100; i++) t1.mul(Math.random()).free();
    t1.free();
    console.timeEnd();

    console.log("--- Pairwise [large] [out-of-place] ---");
    console.time();
    t1 = tensor([10, 100, 100]).rand();
    t2 = tensor([10, 100, 100]).rand();
    for (let i = 0; i < 100; i++) t1.mul(t2).free();
    t1.free();
    t2.free();
    console.timeEnd();

    console.log("--- Broadcasting [large] [out-of-place] ---");
    console.time();
    t1 = tensor([10, 200, 200]).rand();
    t2 = tensor([200, 200]).rand();
    for (let i = 0; i < 100; i++) t1.mul(t2).free();
    t1.free();
    t2.free();

    t1 = tensor([10, 100, 100]).rand();
    t2 = tensor([100]).rand();
    for (let i = 0; i < 100; i++) t1.mul(t2).free();
    t1.free();
    t2.free();

    t1 = tensor([10, 1, 200]).rand();
    t2 = tensor([200, 1]).rand();
    for (let i = 0; i < 100; i++) t1.mul(t2).free();
    t1.free();
    t2.free();
    console.timeEnd();

    console.log("--- Matmul [large] [in-place] ---");
    console.time();
    t1 = tensor([2, 200, 200]).rand();
    t2 = tensor([2, 200, 200]).rand();
    for (let i = 0; i < 100; i++) t1.matmul(t2, true);
    console.timeEnd();

    console.log("--- Matmul [large] [out-of-place] ---");
    console.time();
    t1 = tensor([2, 200, 200]).rand();
    t2 = tensor([2, 200, 200]).rand();
    for (let i = 0; i < 100; i++) t1.matmul(t2).free();
    console.timeEnd();

    console.log("--- Matmul [small] [in-place] ---");
    console.time();
    t1 = tensor([2, 3, 3]).rand();
    t2 = tensor([2, 3, 3]).rand();
    for (let i = 0; i < 200000; i++) t1.matmul(t2, true);
    console.timeEnd();

    console.log("--- Matmul [small] [out-of-place] ---");
    console.time();
    t1 = tensor([2, 3, 3]).rand();
    t2 = tensor([2, 3, 3]).rand();
    for (let i = 0; i < 200000; i++) t1.matmul(t2).free();
    console.timeEnd();

    console.log("--- Matmul [multi-shape] [out-of-place] ---");
    console.time();
    console.time();
    for (let i = 0; i < 100000; i++) {
        let inner = rand_int(1, 10);
        t1 = tensor([2, i % 19, inner]).rand();
        t2 = tensor([2, inner, i % 21]).rand();
        t1.matmul(t2).free();
        t1.free();
        t2.free();
    }
    console.timeEnd();

    console.log("--- Matmul [multi-shape] [in-place] ---");
    console.time();
    for (let i = 0; i < 100000; i++) {
        let a = i % 19;
        t1 = tensor([2, a, a]).rand();
        t2 = tensor([2, a, a]).rand();
        t1.matmul(t2, true);
        t1.free();
        t2.free();
    }
    console.timeEnd();

    const total_time = Date.now() - start;
    console.log(`Total: ${total_time}ms`);
});
