// pairwise operations on two arrays of identical size

#ifndef CORE_PAIRWISE
#define CORE_PAIRWISE

#include <stddef.h>
#include <stdio.h>
#include "./util.h"

// todo:
//   think about moving pairwise op control code to c (only if it reduces complexity)


// simple pairwise operations

#define PARIWISE_OP(NAME, OP) \
    void NAME(struct tensor_t* a, struct tensor_t* b, struct tensor_t* res) { \
        for (size_t i = 0; i < a->nelem; i++) res->data[get_index(res, i)] = get_item(a, i) OP get_item(b, i); }

PARIWISE_OP(add_prw, +) // add
PARIWISE_OP(sub_prw, -) // sub
PARIWISE_OP(mul_prw, *) // mul
PARIWISE_OP(div_prw, /) // div


// broadcasting operations

#define BAROADCASTING_PARIWISE_OP(NAME, OP) \
    void NAME(struct tensor_t *a, struct tensor_t *b, struct tensor_t *res) { \
        size_t ia = 0, ib = 0, iaxis, remainder, dim, total_elements = 1; \
        size_t strides_a[res->rank], strides_b[res->rank]; \
        /* extend stride arrays of a and b with zeros to match rank of result tensor */ \
        for (dim = res->rank; dim-- > 0;) { \
            /* original condition was (res->rank - a->rank > dim) but we cannot safely do */ \
            /* subtractions here because size_t would underflow so i reformulated the inequality */ \
            /*               [pad with zeros to the left]     [when shape[dim] is 1 we can"t step to the next element, so set stride to 0]*/ \
            strides_a[dim] = (res->rank > dim + a->rank ? 0 : (a->shape[dim] == 1 ? 0 : a->strides[dim - (res->rank - a->rank)])); \
            strides_b[dim] = (res->rank > dim + b->rank ? 0 : (b->shape[dim] == 1 ? 0 : b->strides[dim - (res->rank - b->rank)])); \
        } \
        for (size_t ires = 0; ires < res->nelem; ires++) { \
            ia = ib = 0; \
            /* get indices of a, b from ires */ \
            remainder = ires; \
            for (dim = res->rank; dim-- > 0;) { \
                /* index of current element on current axis */ \
                iaxis = remainder % res->shape[dim]; \
                ia += iaxis * strides_a[dim]; \
                ib += iaxis * strides_b[dim]; \
                remainder /= res->shape[dim]; \
            } \
            res->data[res->offset + ires] = a->data[a->offset + ia] OP b->data[b->offset + ib]; \
        } \
    }

BAROADCASTING_PARIWISE_OP(add_prw_brc, +) // div
BAROADCASTING_PARIWISE_OP(sub_prw_brc, -) // sub
BAROADCASTING_PARIWISE_OP(mul_prw_brc, *) // mul
BAROADCASTING_PARIWISE_OP(div_prw_brc, /) // div

#endif //CORE_PAIRWISE
