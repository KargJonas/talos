/**
 * This file is used for validation and debugging during development. 
 */

import { RawTensor, core, mgmt } from "../index.ts";
import { core_ready } from "../src/raw_tensor/management.ts";
import { mul_acc } from "../src/raw_tensor/raw_tensor_operations.ts";
import { set_rand_seed } from "../src/raw_tensor/util.ts";
import { tensor, tensor_from_array, tensor_producer } from "../src/tensor_factory.ts";

// We have to wait for the WASM module to load.
// This is done using the `core_ready` promise.
// 
// NOTE: If your runtime does not support top-level await,
//       you can use core_ready.then(() => { ... }) instead.
await core_ready;

try {
    // if your runtime does not support top-level await,
    // you'll have to use core_ready.then(() => { ... }) instead
    set_rand_seed(Date.now());

    console.log("\nRunning SGD demo...\n");

    const size = 100;
    const weight = tensor([2, size], true).kaiming_normal(size * 2);
    const bias = tensor([size], true).kaiming_normal(size);
    const target = tensor([size]).uniform(0, 1);

    weight.print();
    bias.print();

    const a = RawTensor.create([size]);
    const input = tensor_producer([size], () => a.normal(3, 1));

    // define computation graph: mean((target - relu(Weight * input + Bias))^2)
    const nn = weight.matmul(input).add(bias).set_name("add").leaky_relu(.05).mse_loss(target);

    // finds an execution sequence for the operations involved in the previously defined graph
    const graph = nn.graph;
    const learningRate = 3;

    console.time();

    // training loop
    for (let epoch = 0; epoch < 1000; epoch++) {
        graph.zero_grad();
        graph.forward();
        graph.backward();    

        console.log(`\x1b[35m[${mgmt.get_total_allocated()} Bytes]\x1b[0m Epoch ${epoch + 1}: Loss = ${graph.outputs[0].value.toString()}`);
        // graph.get_node("add")?.grad?.print(10);

        // update weights using SGD
        mul_acc(weight.grad!, -learningRate, weight.value);
        mul_acc(bias.grad!, -learningRate, bias.value);
    }

    console.write("\nTraining completed in: ");
    console.timeEnd();


} catch (e) {
    console.log(e);
}
