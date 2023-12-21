import { core_ready } from '../src/util';
import tensor from '../src/tensor';

const print = (t) => console.log(t?.toString() + "\n---");

core_ready.then(() => {
    const t1 = tensor([200, 300, 3]).rand();
    const t2 = tensor([200, 300, 3]).rand();
    const t3 = tensor([3]).rand();

    console.log('no broadcasting')
    console.time();
    for (let i = 0; i < 100; i++) {
        t1.add(t2).free();
    }
    console.timeEnd();

    console.log('broadcasting')
    console.time();
    for (let i = 0; i < 100; i++) {
        t1.add(t3).free();
    }
    console.timeEnd();
});
