import tensor from './tensor';

// matrix with 2 rows and 3 columns
let t1 = tensor([1, 2, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
let t2 = tensor([2, 2, 4, 4]).rand();

// // matrix with 3 rows and 2 columns
// let t2 = new Tensor([3, 2], [1, 2, 3, 4, 5, 6]);

// for (let mat of t1.get_axis_iterable(0)) {
//     console.log(mat.toString());
// }

// for (let mat of t1.get_axis_iterable(1)) {
//     console.log(mat.toString());
// }

console.log(t1.toString());
console.log(t2.toString())
