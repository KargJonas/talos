export default interface ITensor<TensorType> {
    rank: number;
    nelem: number;
    size: number;
    // get_isview: () => this.view[STRUCT_LAYOUT.ISVIEW];
    // get_data_ptr: () => this.view[STRUCT_LAYOUT.DATA];
    // get_shape_ptr: () => this.view[STRUCT_LAYOUT.SHAPE];
    // get_strides_ptr: () => this.view[STRUCT_LAYOUT.STRIDES];
    rows: number;
    cols: number;
    get_axis_size: (axis_index: number) => number;

    print: () => void;
    print_info: (title: string) => void;

    /* VIEW */ get_axis_iterable: (n: number) => Generator<TensorType>;

    // init operations
    /* INIT */ zeros: () => TensorType;
    /* INIT */ ones: () => TensorType;
    /* INIT */ clone: () => TensorType;
    /* INIT */ rand: (min: number, max: number) => TensorType;
    /* INIT */ rand_int: (min: number, max: number) => TensorType;
    /* INIT */ fill: (value: number) => TensorType;

    // memory dealloc operation
    free: () => void;

    // view operations
    /* VIEW */ create_view: (axis: number, offset: number) => TensorType;
    /* VIEW */ transpose: (...permutation: number[]) => TensorType;

    // unary operations
    /* UNARY */ relu: (in_place?: boolean) => TensorType;
    /* UNARY */ binstep: (in_place?: boolean) => TensorType;
    /* UNARY */ logistic: (in_place?: boolean) => TensorType;
    /* UNARY */ negate: (in_place?: boolean) => TensorType;
    /* UNARY */ sin: (in_place?: boolean) => TensorType;
    /* UNARY */ cos: (in_place?: boolean) => TensorType;
    /* UNARY */ tan: (in_place?: boolean) => TensorType;
    /* UNARY */ asin: (in_place?: boolean) => TensorType;
    /* UNARY */ acos: (in_place?: boolean) => TensorType;
    /* UNARY */ atan: (in_place?: boolean) => TensorType;
    /* UNARY */ sinh: (in_place?: boolean) => TensorType;
    /* UNARY */ cosh: (in_place?: boolean) => TensorType;
    /* UNARY */ tanh: (in_place?: boolean) => TensorType;
    /* UNARY */ exp: (in_place?: boolean) => TensorType;
    /* UNARY */ log: (in_place?: boolean) => TensorType;
    /* UNARY */ log10: (in_place?: boolean) => TensorType;
    /* UNARY */ log2: (in_place?: boolean) => TensorType;
    /* UNARY */ invsqrt: (in_place?: boolean) => TensorType;
    /* UNARY */ sqrt: (in_place?: boolean) => TensorType;
    /* UNARY */ ceil: (in_place?: boolean) => TensorType;
    /* UNARY */ floor: (in_place?: boolean) => TensorType;
    /* UNARY */ abs: (in_place?: boolean) => TensorType;
    /* UNARY */ reciprocal: (in_place?: boolean) => TensorType;

    // *** binary operations ***
    /* BINARY */ add: (other: TensorType | number, in_place?: boolean) => TensorType;
    /* BINARY */ sub: (other: TensorType | number, in_place?: boolean) => TensorType;
    /* BINARY */ mul: (other: TensorType | number, in_place?: boolean) => TensorType;
    /* BINARY */ div: (other: TensorType | number, in_place?: boolean) => TensorType;
    /* BINARY */ pow: (other: TensorType | number, in_place?: boolean) => TensorType;
    /* BINARY */ dot: (other: TensorType, in_place?: boolean) => TensorType;
    /* BINARY */ matmul: (other: TensorType, in_place?: boolean) => TensorType;

    // reduce operations
    min: () => number;
    max: () => number;
    sum: () => number;
    mean: () => number;

    // operation shorthands
    readonly T: TensorType;

    [Symbol.iterator](): Iterator<TensorType>;
}
