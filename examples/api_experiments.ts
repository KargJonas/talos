import {  core_ready } from "../index";
import { get_total_allocated } from "../src/base/management.ts";
import { mul_acc } from "../src/base/raw_tensor_operations.ts";

// if your runtime does not support top-level await,
// you'll have to use core_ready.then(() => { ... }) instead
await core_ready;

console.log("\nRunning SGD demo...\n");

const optim = Adam();

// trainable parameters
// first parameter is shape, second parameter is requires_grad (default = false)
const weight = Tensor.create([3], true).rand();
const bias = Tensor.create([3], true).rand();

// random but constant target/label, no grad necessary
const target = Tensor.create([3]).rand();

// Create a source node for input
const dataset = Tensor.create([10000, 3, 28, 28], big_flat_array);
const get_sample = dataset.get_axis_iterable(0);
const input = Tensor.producer_source(() => [[1,2,3], [4,3,2]]);

// Modify the computation graph to include the weight
const nn = input.mul(weight).add(bias).mse_loss(target).get_computation_graph();

// this is identical to mse_loss
// const nn = a.mul(weight).sub(target).pow(2).mean();

// Define learning rate
const learningRate = .01;

console.time();

// Training loop
for (let epoch = 0; epoch < 30; epoch++) {
    nn.zero_grad();
    nn.forward();
    nn.backward();

    // Print loss value
    console.log(`\x1b[35m[${get_total_allocated()} Bytes]\x1b[0m Epoch ${epoch + 1}: Loss = ${nn.value.toString()}`);

    // nn.grad?.print(10);

    // Update weights using SGD
    mul_acc(weight.grad!, -learningRate, weight.value);
    mul_acc(bias.grad!, -learningRate, bias.value);

    // print_memory_status();
}

console.write("\nTraining completed in: ");
console.timeEnd();
console.log(`  Weight value: ${weight.value.toString()}`);
console.log(`  Weight grad:  ${weight.grad!.toString()}`);
