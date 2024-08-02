import { RawTensor, core_ready, set_rand_seed, mgmt, optim, tensor, tensor_producer } from "../index";

// if your runtime does not support top-level await,
// you'll have to use core_ready.then(() => { ... }) instead
await core_ready;
set_rand_seed(Date.now());

console.log("\nRunning SGD demo...\n");

const size = 100;
const weight = tensor([2, size], true).kaiming_normal(size * 2).set_name("weight");
const bias = tensor([size], true).kaiming_normal(size).set_name("bias");
const target = tensor([size]).uniform(0, 1);

const a = RawTensor.create([size]);
const input = tensor_producer([size], () => a.normal(3, 1));

// define computation graph: mean((target - relu(Weight * input + Bias))^2)
const nn = weight.matmul(input).add(bias).set_name("add").leaky_relu(.05).mse_loss(target);

// finds an execution sequence for the operations involved in the previously defined graph
const graph = nn.graph;
const optimizer = new optim.sgd(graph, { lr: 20 });

graph.print();
console.time();

for (let iteration = 0; iteration <= 1000; iteration++) {
    graph.zero_grad();
    graph.forward();
    graph.backward();
    optimizer.step();

    if (iteration % 50 === 0) console.log(`\x1b[35m[${mgmt.get_total_allocated()} Bytes]\x1b[0m Iteration ${iteration}: Loss = ${graph.output.value.toString()}`);
}

console.write("\nTraining completed in: ");
console.timeEnd();
