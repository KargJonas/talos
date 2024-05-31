import { get_total_allocated, core_ready } from "./src/base/Management";
export { default as tensor, tensor_like, Tensor } from "./src/base/Tensor";
import core from "./src/base/core/build";
export { set_rand_seed } from "./src/base/util";
export const mgmt = {
    get_total_allocated
};
export { core, core_ready };
