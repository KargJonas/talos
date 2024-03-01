import { get_total_allocated, core_ready } from "./src/Management";
export { default as tensor, tensor_like, Tensor } from "./src/Tensor";
import core from "./src/core/build";
export { set_rand_seed } from "./src/util";
export const mgmt = {
    get_total_allocated
};
export { core, core_ready };
