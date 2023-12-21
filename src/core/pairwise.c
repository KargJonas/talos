// pairwise operations on two arrays of identical size

#ifndef CORE_PAIRWISE
#define CORE_PAIRWISE

#include <stddef.h>
#include "./util.c"
#include <stdio.h>

#define PARIWISE_OP(NAME, OP) \
    void NAME(float* a, float* b, float* res, size_t size) { \
        for (size_t i = 0; i < size; i++) res[i] = a[i] OP b[i]; }

// not proud of this one...
// todo: optimize for perf, make macro smaller
#define BAROADCASTING_PARIWISE_OP(NAME, OP) \
    void NAME(float* a, float* b, float* res, size_t* metadata, size_t rank) { \
        size_t* strides_a = metadata; \
        size_t* strides_b = metadata + rank; \
        size_t* shape_res = strides_b + rank; \
        size_t ia, ib, iaxis, remainder, dim, total_elements = 1; \
        /* compute total number of elements of the resuting tensor */ \
        for (size_t i = 0; i < rank; ++i) total_elements *= shape_res[i]; \
        /* iterate over flat result tensor */ \
        for (size_t ires = 0; ires < total_elements; ires++) { \
            ia = ib = 0; \
            /* get indices of a, b from ires */ \
            remainder = ires; \
            for (dim = rank; dim-- > 0;) { \
                /* index of current element on current axis */ \
                iaxis = remainder % shape_res[dim]; \
                ia += iaxis * strides_a[dim]; \
                ib += iaxis * strides_b[dim]; \
                remainder /= shape_res[dim]; \
            } \
            res[ires] = a[ia] OP b[ib]; \
        } \
    } \

PARIWISE_OP(add_prw, +) // add
PARIWISE_OP(sub_prw, -) // sub
PARIWISE_OP(mul_prw, *) // mul
PARIWISE_OP(div_prw, /) // div

BAROADCASTING_PARIWISE_OP(add_prw_brc, +) // div
BAROADCASTING_PARIWISE_OP(sub_prw_brc, -) // sub
BAROADCASTING_PARIWISE_OP(mul_prw_brc, *) // mul
BAROADCASTING_PARIWISE_OP(div_prw_brc, /) // div

#endif //CORE_PAIRWISE
