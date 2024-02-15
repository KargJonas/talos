import { core_ready, print } from "../src/util";
import tensor from "../src/Tensor";

core_ready.then(() => {
    console.log("###########\n".repeat(2));

    const t1 = tensor([2, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const t2 = tensor([3, 2],    [1, 2, 3, 4, 5, 6]);
    const t4 = tensor([3],       [-1, 2, 3]);
    const t5 = tensor([2, 2]).rand_int(1, 6);
    const t6 = tensor([2, 2]).rand_int(1, 6);

    print(t1.add(t4));
    print(t1.sub(t4));
    print(t1.mul(t4));
    print(t1.div(t4));
    print(t1.matmul(t2));
    print(t2.dot(t1));
    print(t4.logistic());
    print(t5.matmul(t6, true));
    print(t5.dot(t6, true));
});