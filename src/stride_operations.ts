import Shape from "./Shape";

export function get_row_major(shape: Shape | number[]): number[] {
    let stride = 1;
    // const strides = new Int32Array(shape.length);
    const strides = new Array(shape.length);
    strides.fill(1);

    for (let i = shape.length - 2; i >= 0; i--) {
        strides[i] = stride *= shape[i + 1];
    }

    return strides;
}
