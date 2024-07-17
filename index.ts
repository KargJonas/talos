import { get_total_allocated, core_ready, get_ntensors } from "./src/base/management.ts";
import core from "./src/base/core/build";

export { RawTensor } from "./src/base/raw_tensor.ts";
export { set_rand_seed } from "./src/base/util";
export { core, core_ready };
export const mgmt = { get_total_allocated, get_ntensors };
