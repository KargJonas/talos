import { Input } from "../autograd/node_operations";
import Tensor from "../tensor";
import { tensor, tensor_input } from "../tensor_factory";

class Layer {
    input: Input;
    output: Tensor;

    constructor(input: Input, output: Tensor) {
        this.input = input;
        this.output = output;
    }

    connect(input: Tensor) {
        this.input.connect(input);
    }
}

class Linear extends Layer {
    constructor(in_features: number, out_features: number, needs_bias: boolean) {
        const weight = tensor([out_features, in_features], true);
        const input = tensor_input([in_features]);
        let output = weight.matmul(input); // todo undo, use dot instead
        // this.output = this.weight.dot(this.input);

        if (needs_bias) {
            const bias = tensor([out_features], true);
            output = output.add(bias);
        }

        super(input, output);
    }
}

// "fuses" multiple layers into just one
export function sequential(layers: Layer[]): Layer {
    if (!layers.length) throw new Error("Sequential composition expects at least one layer.");
    const input: Input = layers[0].input;
    const output: Tensor = layers[layers.length - 1].output;
    let previous: Layer = layers[0];

    for (let i = 1; i < layers.length; i++) {
        const current = layers[i];
        current.connect(previous.output);
        previous = current;
    }

    return new Layer(input, output);
}

// factory functions
export const linear = (in_features: number, out_features: number, needs_bias: boolean = true) => new Linear(in_features, out_features, needs_bias);
