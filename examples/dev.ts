/**
 * This file is used for validation and debugging during development. 
 */

import { tensor, tensor_from_array as tensor_from_arr, tensor_producer } from "../src/tensor_factory.ts";
import { core_ready, get_total_allocated } from "../src/base/Management.ts";
import { dot, matmul_acc, mul_acc } from "../src/base/raw_tensor_operations.ts";
import { set_rand_seed } from "../src/base/util.ts";
import { RawTensor } from "../src/base/RawTensor.ts";

// if your runtime does not support top-level await,
// you'll have to use core_ready.then(() => { ... }) instead
await core_ready;

try {
    console.log("\nRunning SGD demo...\n");

    // Input and target tensors

    // const weight = parameter_node(RawTensor.create([3]).rand(), true);
    // const bias = parameter_node(RawTensor.create([3]).rand(), true);
    // const input = RawTensor.create([3]).rand();  // random but constant "input data"
    // const target = RawTensor.create([3]).rand(); // random but constant target/label

    // const weight = parameter_node(RawTensor.create([3], [6, 2, 8]), true);
    // const bias = parameter_node(RawTensor.create([3], [0, 0, 0]), true);
    // const input = RawTensor.create([3], [2, -3, 9]);  // random but constant "input data"
    // const target = RawTensor.create([3], [23, 2, -3]); // random but constant target/label

    set_rand_seed(Date.now());
    // // const weight = tensor([3], true).rand();
    // // const bias   = tensor([3], true).rand();

    const size = 100;

    const weight = tensor([2, size], true).rand(0, 1);
    const bias = tensor([size], true).rand(0, 1);
    const target = tensor([size]).rand(0, 1);

    // note to self:
    //   if you initialize with negative values and use relu,
    //   then you will oftentimes get zero values/gradients which
    //   can prevent any progress from happening
    //   apparently, leaky relu can fix this

    const a = RawTensor.create([size]);
    const input = tensor_producer([size], () =>  a.normal(3, 1));

    // Modify the computation graph to include the weight
    const nn = weight.matmul(input).add(bias).relu().set_name("relu").mse_loss(target);

    // this is identical to mse_loss
    // const nn = a.mul(weight).sub(target).pow(2).mean();

    const graph = nn.graph;

    // Define learning rate
    const learningRate = 10;

    console.time();

    // Training loop
    for (let epoch = 0; epoch < 30; epoch++) {
        graph.zero_grad();
        graph.forward();
        graph.backward();

        // Print loss value
        console.log(`\x1b[35m[${get_total_allocated()} Bytes]\x1b[0m Epoch ${epoch + 1}: Loss = ${graph.outputs[0].value.toString()}`);

        // Update weights using SGD
        mul_acc(weight.grad!, -learningRate, weight.value);
        mul_acc(bias.grad!, -learningRate, bias.value);

        // graph.get_node("relu")?.grad?.print();
    }

    console.write("\nTraining completed in: ");
    console.timeEnd();
} catch (e) {
    console.log(e);
}
