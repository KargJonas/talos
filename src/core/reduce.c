#ifndef CORE_REDUCE
#define CORE_REDUCE

#include "float.h"

// #define SCALAR_OP(NAME, OP, INITIAL_VALUE) \
//     float NAME(struct tensor_t* a) { \
//         size_t dim, ia, remainder, iaxis; \
//         float acc = INITIAL_VALUE; \
//         \
//         for (size_t ires = 0; ires < a->nelem; ires++) { \
//             ia = a->offset; remainder = ires; \
//             \
//             for (dim = a->rank; dim-- > 0;) { \
//                 iaxis = remainder % a->shape[dim]; \
//                 ia += iaxis * a->strides[dim]; \
//                 remainder /= a->shape[dim]; \
//             } \
//             \
//             acc = OP(acc, a->data[ia]); \
//         } \
//         \
//         return max; } \

/**
 * TODO
 * This approach is a bit of a departure of how i previously
 * handled operations like this (monolithic macros).
 * In this case it seems to be extremely cumbersome to do it
 * like that so I decided to introduce a get_index function
 * that allows us to iterate over all elements of a tensor
 * with ease.
 * This, of course, comes with the overhead of function calls
 * which i initially wanted to avoid at all cost.
 * But... don't know how bad the overhead truly is.
 * Benchmarking will bring the truth to the surface.
 */

size_t get_index(struct tensor_t* a, size_t linear_index) {
    size_t ia = a->offset;
    size_t remainder = linear_index;
    size_t iaxis;

    for (size_t dim = a->rank; dim-- > 0;) {
        iaxis = remainder % a->shape[dim];
        ia += iaxis * a->strides[dim];
        remainder /= a->shape[dim];
    }

    return ia;
}

float max_red(struct tensor_t* a) {
    float val, max = FLT_MIN;
    
    for (size_t ires = 0; ires < a->nelem; ires++) {
        val = a->data[get_index(a, ires)];
        if (val > max) max = val;
    }

    return max;
}

float min_red(struct tensor_t* a) {
    float val, min = FLT_MAX;
    
    for (size_t ires = 0; ires < a->nelem; ires++) {
        val = a->data[get_index(a, ires)];
        if (val < min) min = val;
    }

    return min;
}

float sum_red(struct tensor_t* a) {
    float sum = 0;
    
    for (size_t ires = 0; ires < a->nelem; ires++) {
        sum += a->data[get_index(a, ires)];
    }

    return sum;
}

#endif//CORE_REDUCE
