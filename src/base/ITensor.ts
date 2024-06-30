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
    relu:       () => TensorType;
    binstep:    () => TensorType;
    logistic:   () => TensorType;
    negate:     () => TensorType;
    sin:        () => TensorType;
    cos:        () => TensorType;
    tan:        () => TensorType;
    asin:       () => TensorType;
    acos:       () => TensorType;
    atan:       () => TensorType;
    sinh:       () => TensorType;
    cosh:       () => TensorType;
    tanh:       () => TensorType;
    exp:        () => TensorType;
    log:        () => TensorType;
    log10:      () => TensorType;
    log2:       () => TensorType;
    invsqrt:    () => TensorType;
    sqrt:       () => TensorType;
    ceil:       () => TensorType;
    floor:      () => TensorType;
    abs:        () => TensorType;
    reciprocal: () => TensorType;

    // binary operations
    add: (other: TensorType | number) => TensorType;
    sub: (other: TensorType | number) => TensorType;
    mul: (other: TensorType | number) => TensorType;
    div: (other: TensorType | number) => TensorType;
    pow: (other: TensorType | number) => TensorType;
    dot: (other: TensorType, ) => TensorType;
    matmul: (other: TensorType, ) => TensorType;

    // reduce operations
    min: () => TensorType;
    max: () => TensorType;
    sum: () => TensorType;
    mean: () => TensorType;

    // operation shorthands
    readonly T: TensorType;

    [Symbol.iterator](): Iterator<TensorType>;
}
