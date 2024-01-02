import Shape from "./Shape";

export function get_column_major(shape: Shape): Int32Array {
    let stride = 1;
    const strides = new Int32Array(shape.length);
    strides.fill(1);

    for (let i = shape.length - 2; i >= 0; i--) {
        strides[i] = stride *= shape[i + 1];
    }

    return strides;
}
