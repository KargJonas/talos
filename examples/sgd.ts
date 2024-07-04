import { RawTensor, core_ready, set_rand_seed } from "../index";
import { get_total_allocated } from "../src/base/Management.ts";
import { mul_acc } from "../src/base/raw_tensor_operations.ts";
import { tensor, tensor_producer } from "../src/tensor_factory.ts";

// if your runtime does not support top-level await,
// you'll have to use core_ready.then(() => { ... }) instead
await core_ready;

console.log("\nRunning SGD demo...\n");

set_rand_seed(Date.now());

const size = 100;

const weight = tensor([2, size], true).rand(0, 1);
const bias = tensor([size], true).rand(0, 1);
const target = tensor([size]).rand(0, 1);

const a = RawTensor.create([size]);
const input = tensor_producer([size], () =>  a.normal(3, 1));

// define computation graph: mean((target - relu(Weight * input + Bias))^2)
const nn = weight.matmul(input).add(bias).relu().set_name("relu").mse_loss(target);

// this is identical to above
// const nn = weight.matmul(input).add(bias).relu().set_name("relu").sub(target).pow(2).mean();

// finds an execution sequence for the operations involved in the previously defined graph
const graph = nn.graph;

const learningRate = 10;

console.time();

// training loop
for (let epoch = 0; epoch < 30; epoch++) {
    graph.zero_grad();
    graph.forward();
    graph.backward();

    // print loss value
    console.log(`\x1b[35m[${get_total_allocated()} Bytes]\x1b[0m Epoch ${epoch + 1}: Loss = ${graph.outputs[0].value.toString()}`);

    // update weights using SGD
    mul_acc(weight.grad!, -learningRate, weight.value);
    mul_acc(bias.grad!, -learningRate, bias.value);

    // graph.get_node("relu")?.grad?.print();
}

console.write("\nTraining completed in: ");
console.timeEnd();
