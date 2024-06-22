import transform from "./tree_transform.ts";

const input_dir = process.env.CORE_SRC_DIR!;
const output_dir = process.env.CORE_PREPROC_OUT_DIR!;

transform(input_dir, output_dir, preprocess);

function preprocess(content: string, name: string) {


    return {
        content, name
    };
}