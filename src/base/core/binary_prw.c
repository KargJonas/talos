// pairwise operations on two arrays of identical size

#ifndef CORE_PAIRWISE
#define CORE_PAIRWISE

#include <stddef.h>
#include <stdio.h>
#include "./util.h"

#define INFIX_OP(a, b, operator) a operator b
#define PREFIX_OP(a, b, operator) operator(a, b)


#define PARIWISE_OP(NAME, RESULT) \
    void NAME(struct tensor_t* a, struct tensor_t* b, struct tensor_t* res) { \
        for (size_t i = 0; i < a->nelem; i++) res->data[get_index(res, i)] = RESULT; }

PARIWISE_OP(add_prw, INFIX_OP(get_item(a, i), get_item(b, i), +)) // add
PARIWISE_OP(sub_prw, INFIX_OP(get_item(a, i), get_item(b, i), -)) // sub
PARIWISE_OP(mul_prw, INFIX_OP(get_item(a, i), get_item(b, i), *)) // mul
PARIWISE_OP(div_prw, INFIX_OP(get_item(a, i), get_item(b, i), /)) // div
PARIWISE_OP(pow_prw, PREFIX_OP(get_item(a, i), get_item(b, i), pow)) // div

#define BAROADCASTING_PARIWISE_OP(NAME, RESULT) \
    void NAME(struct tensor_t *a, struct tensor_t *b, struct tensor_t *res) { \
        size_t ia = 0, ib = 0, iaxis, remainder, dim; \
        size_t strides_a[res->rank], strides_b[res->rank]; \
        bool a_scl = a->rank == 1 && a->shape[0] == 1; \
        bool b_scl = a->rank == 1 && a->shape[0] == 1; \
        /* at least of the two tensors is a scalar */ \
        if (a_scl || b_scl) { \
            ia = get_index(a, 0); ib = get_index(b, 0); /* scalar value is in 0th index of data array */ \
            if (!a_scl) for (size_t i = 0; i < a->nelem; i++) res->data[get_index(res, i)] = a->data[get_index(a, i)] + b->data[ib]; \
            if (!b_scl) for (size_t i = 0; i < a->nelem; i++) res->data[get_index(res, i)] = a->data[ia] + b->data[get_index(b, i)]; \
            if (a_scl && b_scl) res->data[get_index(res, 0)] = a->data[ia] + b->data[ib]; \
            return; \
        } \
        /* extend stride arrays of a and b with zeros to match rank of result tensor */ \
        for (dim = res->rank; dim-- > 0;) { \
            /* original condition was (res->rank - a->rank > dim) but we cannot safely do */ \
            /* subtractions here because size_t would underflow so i reformulated the inequality */ \
            /*               [pad with zeros to the left]     [when shape[dim] is 1 we can"t step to the next element, so set stride to 0]*/ \
            strides_a[dim] = (res->rank > dim + a->rank ? 0 : (a->shape[dim - (res->rank - a->rank)] == 1 ? 0 : a->strides[dim - (res->rank - a->rank)])); \
            strides_b[dim] = (res->rank > dim + b->rank ? 0 : (b->shape[dim - (res->rank - b->rank)] == 1 ? 0 : b->strides[dim - (res->rank - b->rank)])); \
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
            res->data[res->offset + ires] = RESULT; \
        } \
    }

BAROADCASTING_PARIWISE_OP(add_prw_brc, INFIX_OP(a->data[a->offset + ia], b->data[b->offset + ib], +)) // div
BAROADCASTING_PARIWISE_OP(sub_prw_brc, INFIX_OP(a->data[a->offset + ia], b->data[b->offset + ib], -)) // sub
BAROADCASTING_PARIWISE_OP(mul_prw_brc, INFIX_OP(a->data[a->offset + ia], b->data[b->offset + ib], *)) // mul
BAROADCASTING_PARIWISE_OP(div_prw_brc, INFIX_OP(a->data[a->offset + ia], b->data[b->offset + ib], /)) // div
BAROADCASTING_PARIWISE_OP(pow_prw_brc, PREFIX_OP(a->data[a->offset + ia], b->data[b->offset + ib], pow)) // div

#endif //CORE_PAIRWISE
