import { core_ready } from '../src/util';
import tensor, { Tensor } from '../src/Tensor';

const print = (t: Tensor) => console.log(t?.toString() + "\n---");

core_ready.then(() => {
    console.log('###########')

    let a = tensor([2, 2], [1, 2, 3, 4]);
    let b = tensor([2], [5, 6]);
    let c = tensor([1, 2], [7, 8]);

    // Test 1: a + b
    print(a.add(b));

    // Test 2: a - c
    print(a.sub(c));

    // Test 3: b * a
    print(b.mul(a));

    // Test 4: c / a
    print(c.div(a));
});