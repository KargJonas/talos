import { RawTensor, core_ready } from "../index";
import { parameter_node, source_node } from "../src/node_factory.ts";
import { mul_acc } from "../src/base/raw_tensor_operations.ts";

// if your runtime does not support top-level await,
// you'll have to use core_ready.then(() => { ... }) instead
await core_ready;

console.log("\nRunning SGD demo...\n");

// Input and target tensors
const weight = parameter_node(RawTensor.create([30]).rand(), true);
const bias = parameter_node(RawTensor.create([30]).rand(), true);
const input = RawTensor.create([30]).rand();  // random but constant "input data"
const target = RawTensor.create([30]).rand(); // random but constant target/label

// Create a source node for input
const a = source_node([30], () => input);

// Modify the computation graph to include the weight
const nn = a.mul(weight).add(bias).mse_loss(target);

// this is identical to mse_loss
// const nn = a.mul(weight).sub(target).pow(2).mean();

const graph = nn.get_computation_graph();

// Define learning rate
const learningRate = 10;

console.time();

// Training loop
for (let epoch = 0; epoch < 30; epoch++) {
    graph.zero_grad();
    graph.forward();
    graph.backward();

    // Print loss value
    console.log(`Epoch ${epoch + 1}: Loss = ${graph.outputs[0].value.toString()}`);

    // Update weights using SGD
    mul_acc(weight.grad!, -learningRate, weight.value);
    mul_acc(bias.grad!, -learningRate, bias.value);
}

console.write("\n\nTraining completed in: ");
console.timeEnd();
console.log(`  Weight value: ${weight.value.toString()}`);
console.log(`  Weight grad:  ${weight.grad!.toString()}`);
