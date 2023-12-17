import { core_ready } from './util';
import tensor from './tensor';

core_ready.then(() => {

    let t1 = tensor([2, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);    
    let t2 = tensor([3, 2], [1, 2, 3, 4, 5, 6]);
    let t3 = tensor([2, 3], [1, 2, 3, 2, 4, 2]);
    let t4 = tensor([3, 2], [-10, -10, -10, -10, -10, -10]);
    let t5 = tensor([3, 2]).rand()

    console.log(t1.toString() + "\n");
    console.log(t1.get(1,1).toString() + "\n");

    for (const st of t1.get_axis_iterable(1)) {
        console.log(st.toString())
    }
});

// class C {
//     constructor() {

//         const p = new Proxy(this , {
//             get(target, p, receiver) {
//                 if (target === Symbol.toPrimitive) {
//                     return () => 23;
//                 }

//                 return 34;
//             },

//             // [Symbol.toPrimitive]() {
//             //     return my_other_val;
//             // }
//         });
//     }
// }

// const c = new C();
// console.log(+c);
