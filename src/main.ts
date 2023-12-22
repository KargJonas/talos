import { core_ready, tensor_like } from './util';
import { add, mul, sub, div, matmul, dot, sigmoid } from './tensor/tensor_operations';
import { tensor } from './util';

import './graph/graph';

const print = (t) => console.log(t?.toString() + "\n---");

core_ready.then(() => {
    console.log('###########\n'.repeat(2));

    let t1 = tensor([2, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    let t2 = tensor([3, 2],    [1, 2, 3, 4, 5, 6]);
    let t3 = tensor([2, 3],    [-100, 2, 3, 2, 4, 2]);
    let t4 = tensor([3],       [-1, 2, 3]);

    let t5 = tensor([2, 2]);
    let t6 = tensor_like(t5).rand_int(0, 10);

    print(add(t5, 3));
    print(add(t5, 3, true));
    print(add(t5, t5, true));
    // print(matmul(t5, t6, true));
    // print(matmul(t5, t6, true));

    // let t5 = tensor([2, 2],    [2, 5, 2, 2]);
    // print(matmul(t5, t5, true));

    // print(t1.add(t4))
    // print(t1.sub(t4))
    // print(t1.mul(t4))
    // print(t1.div(t4))
    // print(t1.matmul(t2));
    // print(t2.dot(t1));
    // print(t4.sigmoid());

    // print(add(t1, t4))
    // print(sub(t1, t4))
    // print(mul(t1, t4))
    // print(div(t1, t4))
    // print(matmul(t1, t2));
    // print(dot(t1, t2));
    // print(sigmoid(t4));
});
