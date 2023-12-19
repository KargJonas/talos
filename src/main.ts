import { core_ready } from './util';
import tensor from './tensor';

const print = (t) => console.log(t?.toString() + "\n---");

core_ready.then(() => {

    let t1 = tensor([2, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);    
    let t2 = tensor([3, 2],    [1, 2, 3, 4, 5, 6]);
    let t3 = tensor([2, 3],    [-100, 2, 3, 2, 4, 2]);

    // console.log(t1.matmul(t2).tanh().toString() + "\n---");

    print(t2.matmul(t1));

    print(t2.dot(t1))
});
