import { RawTensor, core_ready, mgmt, optim, tensor, tensor_producer } from "../index";

// if your runtime does not support top-level await,
// you'll have to use core_ready.then(() => { ... }) instead
await core_ready;

// no nondeterminism such that results can be compared to examples/sgd2.ts
// set_rand_seed(Date.now());

console.log("\nRunning SGD demo...\n");

const size_0 = 50;
const size_1 = 50;
const weight = tensor([size_1, size_0], true).kaiming_normal(size_0 * size_1).set_name("weight");
const bias = tensor([size_0], true).kaiming_normal(size_0).set_name("bias");
const target = tensor([size_0]).uniform(0, 1);

const a = RawTensor.create([size_0]);
const input = tensor_producer([size_0], () => a.normal(3, 1));

// define computation graph: mean((target - relu(Weight * input + Bias))^2)
const nn = weight.matmul(input).add(bias).set_name("add").leaky_relu(.05).mse_loss(target);

// finds an execution sequence for the operations involved in the previously defined graph
const graph = nn.graph;
const optimizer = new optim.sgd(graph, { lr: .05  });

graph.print({ show_shape: true });
console.time();

for (let iteration = 0; iteration <= 1000; iteration++) {
    graph.zero_grad();
    graph.forward();
    graph.backward();
    optimizer.step();

    console.log(`\x1b[35m[${mgmt.get_total_allocated()} Bytes]\x1b[0m Iteration ${iteration}: Loss = ${graph.output.value.toString()}`);
}

console.write("\nTraining completed in: ");
console.timeEnd();
