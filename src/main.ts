import { core_ready } from './util';
import tensor from './tensor';

const print = (t) => console.log(t?.toString() + "\n---");

core_ready.then(() => {

    let t1 = tensor([2, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);    
    let t2 = tensor([3, 2],    [1, 2, 3, 4, 5, 6]);
    let t3 = tensor([2, 3],    [-100, 2, 3, 2, 4, 2]);
    let t4 = tensor([3],       [1, 2, 3]);

    // console.log(t1.matmul(t2).tanh().toString() + "\n---");

    // basic printing
    // print(t1);

    // flatten/unflatten to specific ranks
    // print(t1.flatten(1));
    // print(t1.flatten(2));
    // print(t1.flatten(10));

    // matmul/dot product
    // print(t2.matmul(t1));
    // print(t2.dot(t1));

    // console.log(t1.shape.broadcast(t4.shape))
    
    // let t1 = tensor([3, 1, 2]).rand_int(1, 5);
    // let t2 = tensor([3, 2, 1]).rand_int(1, 5);

    // print(t1.add(t4));

    // print(t1)
    // print(t2)
    const res = t1.add(t4);
    print(res)

    // res.free();
    print(t1.add(t4))
});