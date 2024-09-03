/**
 * This file is used for validation and debugging during development. 
 */

import { RawTensor, core_ready, set_rand_seed, mgmt, optim, tensor, tensor_producer, Tensor } from "../index";

// if your runtime does not support top-level await,
// you'll have to use core_ready.then(() => { ... }) instead

try {
    await core_ready;
    set_rand_seed(Date.now());

    console.log("\nRunning SGD demo...\n");

    const size = 100;
    const weight = tensor([2, size], true).kaiming_normal(size * 2).set_name("weight");
    const bias = tensor([size], true).kaiming_normal(size).set_name("bias");
    const target = tensor([size]).uniform(0, 1);

    const a = RawTensor.create([size]);
    const dataset = tensor_producer([size], () => a.normal(3, 1));

    // define computation graph: mean((target - relu(Weight * input + Bias))^2)
    // const nn = weight.matmul(input).add(bias).set_name("add").leaky_relu(.05).mse_loss(target);

    // the way i have designed talos is quite different from how e.g. pytorch
    // works. in pytorch, you usually pass data into a model using the __call__
    // method of the layer's class.
    // in talos, we predefine input nodes during graph creation
    // that act as pull-mode data providers
    // in pytorch, if there are multiple inputs, i think you would typically
    // have two disjoint graphs with one input each. then you push data
    // into each of the graphs and later join the results (and thus the graphs)
    // in talos, you would define the graphs including the input nodes
    // and then join the graphs even before any computation has happened.
    // (maybe pytorch does the same implicitly, im not sure)
    // one downside of the talos approach is that the inputs may become
    // un-synchronized under certain conditions
    // aside from this, the main issue is a significant divergence
    // from the pytorch-like api, because pushing data into subgraphs
    // like this: x = layer1(x); x = layer2(x); ...
    // will not work.
    // i guess ill have to decide still if this is actually bad or not.
    //
    // i suppose one way to fix this is to temporarily connect graphs
    // if we pass in a tensor that is an output of some graph
    // into another graph.
    // we could set the parent of the input node of the second graph
    // to be the output node of the first graph

    // select 20 random tensors from the 0th axis of the dataset tensor
    // the axis argument should be optional and default 0

    class Model {
        layers: Layer[] = [
            nn.conv2d(1, 32, 5).relu(),     // layers is essentially a list
            nn.conv2d(32, 32, 5).relu(),    // of subgraphs that will then
            nn.batch_norm(32).maxpool_2d(), // be connected temporarily
            nn.conv2d(32, 64, 3).relu(),    // during execution
            nn.conv2d(64, 64, 3).relu(),
            nn.batch_norm(64).maxpool_2d(),
        ];

        run(x: Tensor) {
            // this only works with runtime graph connections
            return x.sequential(this.layers);
        }
    }

    // exploring another path.
    // here, we could (theoretically) construct the entire graph
    // of the model before execution
    // note: execution-time graph connections should still
    // be implemented because define-by-run provides a higher
    // degree of flexibility
    const model = input
        .conv2d(1, 32, 5).relu()
        .conv2d(32, 32, 5).relu()
        .batch_norm(32).maxpool_2d()
        .conv2d(32, 64, 3).relu()
        .conv2d(64, 64, 3).relu()
        .batch_norm(64).maxpool_2d();

    // REFERENCE:
    //   this is how batch collection works in tinygrad
    // samples = Tensor.randint(batch_size, high=X_train.shape[0])
    // loss = model(X_train[samples]).sparse_categorical_crossentropy(Y_train[samples]).backward()


    // ### on an init() method and when to allocate memory
    // as you can see, memory is allocated as soon as the constructor of each node is called. this means that something like
    // input.add(other).mul(another);
    // will immediately allocate memory (but won't perform computation).
    // im currently undecided wether i should move this to a dedicated alloc() or init() method (and maybe add a dealloc() method) because for allocation, we need the shape of the tensors.
    // if i want to implement something like this:
    // const my_model = create_source_node().add(other).mul(another);
    // my_model(my_data);
    // then i need to consider that the data passed into my_model could have different shapes between passes.
    // if i want to support that, i need to allocate the tensors only when i get data (when i do a second pass with differently shaped data, then i can sort of cache this, because i can store the shape of the previous data and only reallocate if the shape has changed, but thats beside the point).
    // (an alternative to all this would be to pass a shape into create_source_node() and allocate the tensors using that.)


    // should maybe return a tensor provider that randomly selects
    // slices of the dataset tensor
    const shuffled = dataset.shuffle(0);

    // finds an execution sequence for the operations involved in the previously defined graph
    const graph = model.graph;
    const optimizer = new optim.sgd(graph, { lr: 20 });
    
    graph.print();
    console.time();
    
    for (let iteration = 0; iteration <= 100; iteration++) {
        // batch is not a real tensor but rather a list of tensor views
        // to slices of the dataset
        // todo dispose of batch after use
        const batch = shuffled.gather(20);

        graph.zero_grad();
        graph.forward(batch);
        graph.backward();
        optimizer.step();
    
        if (iteration % 10 === 0) console.log(`\x1b[35m[${mgmt.get_total_allocated()} Bytes]\x1b[0m Iteration ${iteration}: Loss = ${graph.output.value.toString()}`);
    }
    
    console.write("\nTraining completed in: ");
    console.timeEnd();


} catch (e) {
    console.log(e);
}
