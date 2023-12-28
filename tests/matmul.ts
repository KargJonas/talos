import { core_ready } from '../src/util';
import tensor from '../src/Tensor';

core_ready.then(() => {
    console.log("Test 1: 2x2 multiplied by 2x2");
    let t1 = tensor([2, 2], [1, 2, 3, 4]);
    let t2 = tensor([2, 2], [5, 6, 7, 8]);
    console.log(t1.matmul(t2).toString());

    console.log("\nTest 2: 2x3 multiplied by 3x2");
    let t3 = tensor([2, 3], [1, 2, 3, 4, 5, 6]);
    let t4 = tensor([3, 2], [7, 8, 9, 10, 11, 12]);
    console.log(t3.matmul(t4).toString());

    console.log("\nTest 3: Incompatible shapes (should raise error)");
    try {
        let t5 = tensor([3, 2], [1, 2, 3, 4, 5, 6]);
        let t6 = tensor([3, 1], [7, 8, 9]);
        console.log(t5.matmul(t6).toString());
    } catch (error) {
        console.log("Error:", error.message);
    }

    console.log("\nTest 4: Higher dimensional tensors");
    let t7 = tensor([2, 3, 4]).rand();
    let t8 = tensor([2, 4, 3]).rand();
    console.log(t7.matmul(t8).toString());

    console.log("\nTest 5: Vector and Matrix");
    let t9 = tensor([3], [1, 2, 3]);
    let t10 = tensor([3, 2], [4, 5, 6, 7, 8, 9]);
    console.log(t9.matmul(t10).toString());
});
