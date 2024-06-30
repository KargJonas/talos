export default interface ITensor<TensorType> {
    rank: number;
    nelem: number;
    size: number;
    rows: number;
    cols: number;

    get_axis_size: (axis_index: number) => number;

    print: () => void;
    print_info: (title: string) => void;

    get_axis_iterable: (n: number) => Generator<TensorType>;

    // init operations
    zeros: () => TensorType;
    ones: () => TensorType;
    clone: () => TensorType;
    rand: (min: number, max: number) => TensorType;
    rand_int: (min: number, max: number) => TensorType;
    fill: (value: number) => TensorType;

    // memory dealloc operation
    free: () => void;

    // view operations
    create_view: (axis: number, offset: number) => TensorType;
    transpose: (...permutation: number[]) => TensorType;

    // unary operations
    relu:       (in_place?: boolean) => TensorType;
    binstep:    (in_place?: boolean) => TensorType;
    logistic:   (in_place?: boolean) => TensorType;
    negate:     (in_place?: boolean) => TensorType;
    sin:        (in_place?: boolean) => TensorType;
    cos:        (in_place?: boolean) => TensorType;
    tan:        (in_place?: boolean) => TensorType;
    asin:       (in_place?: boolean) => TensorType;
    acos:       (in_place?: boolean) => TensorType;
    atan:       (in_place?: boolean) => TensorType;
    sinh:       (in_place?: boolean) => TensorType;
    cosh:       (in_place?: boolean) => TensorType;
    tanh:       (in_place?: boolean) => TensorType;
    exp:        (in_place?: boolean) => TensorType;
    log:        (in_place?: boolean) => TensorType;
    log10:      (in_place?: boolean) => TensorType;
    log2:       (in_place?: boolean) => TensorType;
    invsqrt:    (in_place?: boolean) => TensorType;
    sqrt:       (in_place?: boolean) => TensorType;
    ceil:       (in_place?: boolean) => TensorType;
    floor:      (in_place?: boolean) => TensorType;
    abs:        (in_place?: boolean) => TensorType;
    reciprocal: (in_place?: boolean) => TensorType;

    // *** binary operations ***
    add: (other: TensorType | number, in_place?: boolean) => TensorType;
    sub: (other: TensorType | number, in_place?: boolean) => TensorType;
    mul: (other: TensorType | number, in_place?: boolean) => TensorType;
    div: (other: TensorType | number, in_place?: boolean) => TensorType;
    pow: (other: TensorType | number, in_place?: boolean) => TensorType;
    dot: (other: TensorType, in_place?: boolean) => TensorType;
    matmul: (other: TensorType, in_place?: boolean) => TensorType;

    // reduce operations
    min: () => TensorType | number;
    max: () => TensorType | number;
    sum: () => TensorType | number;
    mean: () => TensorType | number;

    // operation shorthands
    readonly T: TensorType;

    [Symbol.iterator](): Iterator<TensorType>;
}
