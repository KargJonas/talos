/**
 * This file is used for validation and debugging during development. 
 */

import { core_ready, tensor } from "../index";

// if your runtime does not support top-level await,
// you'll have to use core_ready.then(() => { ... }) instead
await core_ready;

console.log("###########\n".repeat(2));

const dataset_x_0 = tensor([50, 28, 28]).rand();  // 50 images of size 28x28
const dataset_x_1 = tensor([50, 12]);             // additional information for each image
const dataset_y = tensor([50, 10]);               // one hot encoding of 10 classes

function* preprocessor() {

    const iterator_x_0 = dataset_x_0.get_axis_iterable(0);
    const iterator_x_1 = dataset_x_1.get_axis_iterable(0);
    const iterator_y = dataset_y.get_axis_iterable(0);

    while (true) {
        const x_0 = iterator_x_0.next();
        const x_1 = iterator_x_1.next();
        const y = iterator_y.next();
        if (x_0.done || x_1.done || y.done) break;
        yield [x_0.value, , y.value];
    }
}

for (const sample of preprocessor()) {

}