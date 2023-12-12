import core from './core/build';
import { core_ready } from './util';
import tensor from './tensor';

core_ready.then(() => {

    let t1 = tensor([1, 2, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);    
    let t2 = tensor([3, 2], [1, 2, 3, 4, 5, 6]);
    let t3 = tensor([2, 4], [1, 2, 3, 2, 4, 2]);
    let t4 = tensor([2, 4], [-10, -10, -10, -10, -10, -10]);

    t2.dot(t3.add(t4));


    console.log(t3.toString());

    // for (let mat of t1.get_axis_iterable(1)) {
    //     console.log(mat.toString());
    // }
    
});
