import { get_total_allocated, core_ready, get_ntensors } from "./src/raw_tensor/management.ts";
import core from "./src/core/build/index.js";

export { RawTensor } from "./src/raw_tensor/raw_tensor.ts";
export { set_rand_seed } from "./src/raw_tensor/util.ts";
export * from "./src/tensor_factory.ts";
export const mgmt = { get_total_allocated, get_ntensors };

import Tensor from "./src/tensor.ts";
export { core, core_ready, Tensor };
