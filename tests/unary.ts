import { core_ready } from '../src/util';
import tensor from '../src/tensor';

const print = (t) => console.log(t?.toString() + "\n---");

core_ready.then(() => {
    console.log('###########\n'.repeat(2));

    let t1 = tensor([2, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    let t2 = tensor([3, 2],    [1, 2, 3, 4, 5, 6]);
    let t3 = tensor([2, 3],    [-100, 2, 3, 2, 4, 2]);
    let t4 = tensor([3],       [-1, 2, 3]);

    print(t4.negate());
    print(t4.sin());
    print(t4.cos());
    print(t4.tan());
    print(t4.asin());
    print(t4.acos());
    print(t4.atan());
    print(t4.sinh());
    print(t4.cosh());
    print(t4.tanh());
    print(t4.exp());
    print(t4.log());
    print(t4.log10());
    print(t4.log2());
    print(t4.invsqrt());
    print(t4.sqrt());
    print(t4.ceil());
    print(t4.floor());
    print(t4.abs());
    print(t4.reciprocal());
    print(t4.pow(2));
    print(t4.relu())
    print(t4.binstep())
    print(t4.logistic())
    print(t4.sigmoid())
});
