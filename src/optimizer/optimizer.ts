import Graph from "../autograd/graph";
import { mul_acc } from "../raw_tensor/raw_tensor_operations";
import Tensor from "../tensor";

type sgd_options = { lr: number };

export class Optimizer {
    model: Graph;
    parameters: Tensor[] = [];

    constructor(model: Graph) {
        this.model = model;
    }
}

export class sgd extends Optimizer {
    lr: number;
    
    constructor(model: Graph, { lr }: sgd_options) {
        super(model);
        this.lr = lr;
    }

    step() {
        for (const param of this.model.parameters) {
            mul_acc(param.grad, -this.lr, param.value);
        }
    }
}
