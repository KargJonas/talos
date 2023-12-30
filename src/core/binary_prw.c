// pairwise operations on two arrays of identical size

#ifndef CORE_PAIRWISE
#define CORE_PAIRWISE

#include <stddef.h>
#include "./util.h"
#include <stdio.h>

#define PARIWISE_OP(NAME, OP) \
    void NAME(struct tensor_t* a, struct tensor_t* b, struct tensor_t* res) { \
        for (size_t i = 0; i < a->nelem; i++) res->data[i] = a->data[i] OP b->data[i]; }

// not proud of this one...
// todo: optimize for perf, make macro smaller
#define BAROADCASTING_PARIWISE_OP(NAME, OP) \
    void NAME(struct tensor_t* a, struct tensor_t* b, struct tensor_t* res) { \
        size_t ia, ib, iaxis, remainder, dim, total_elements = 1; \
        /* iterate over flat result tensor */ \
        for (size_t ires = 0; ires < res->nelem; ires++) { \
            ia = ib = 0; \
            /* get indices of a, b from ires */ \
            remainder = ires; \
            for (dim = a->rank; dim-- > 0;) { \
                /* index of current element on current axis */ \
                iaxis = remainder % res->shape[dim]; \
                ia += iaxis * a->strides[dim]; \
                ib += iaxis * b->strides[dim]; \
                remainder /= res->shape[dim]; \
            } \
            res->data[ires] = a->data[ia] OP b->data[ib]; \
        } \
    }

PARIWISE_OP(add_prw, +) // add
PARIWISE_OP(sub_prw, -) // sub
PARIWISE_OP(mul_prw, *) // mul
PARIWISE_OP(div_prw, /) // div

BAROADCASTING_PARIWISE_OP(add_prw_brc, +) // div
BAROADCASTING_PARIWISE_OP(sub_prw_brc, -) // sub
BAROADCASTING_PARIWISE_OP(mul_prw_brc, *) // mul
BAROADCASTING_PARIWISE_OP(div_prw_brc, /) // div

// size_t mul_prw_brc(struct tensor_t* a, struct tensor_t* b, struct tensor_t* res) {
//     size_t ia, ib, iaxis, remainder, dim, total_elements = 1;

//     // return b->strides[0];

//     /* iterate over flat result tensor */
//     for (size_t ires = 0; ires < res->nelem; ires++) {
//         ia = 0;
//         ib = 0;
//         /* get indices of a, b from ires */
//         remainder = ires;
//         for (dim = a->rank; dim-- > 0;) {
//             /* index of current element on current axis */
//             iaxis = remainder % res->shape[dim];
//             ia += iaxis * a->strides[dim];
//             ib += iaxis * b->strides[dim];
//             remainder /= res->shape[dim];
//         }
//         res->data[ires] = a->data[ia] * b->data[ib];
//     }
// }

#endif //CORE_PAIRWISE
