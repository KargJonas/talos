import { get_total_allocated, core_ready, get_ntensors } from "./src/base/Management";
import core from "./src/base/core/build";

export { RawTensor } from "./src/base/RawTensor.ts";
export { set_rand_seed } from "./src/base/util";
export { core, core_ready };
export const mgmt = { get_total_allocated, get_ntensors };
