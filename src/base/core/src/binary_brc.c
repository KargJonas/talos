// pairwise operations on two arrays of identical size

#ifndef CORE_BROADCASTING
#define CORE_BROADCASTING

#include <stddef.h>
#include <stdio.h>
#include "./util.h"

#define BROADCASTING_OP(NAME, ASSIGNMENT, RESULT) [[[
void NAME(struct tensor_t *_a, struct tensor_t *_b, struct tensor_t *res) {
    size_t ia, ib, ires, iaxis, remainder, dim;
    size_t strides_a[res->rank], strides_b[res->rank];

    // extend stride arrays of a and b with zeros to match rank of result tensor
    for (dim = res->rank; dim-- > 0;) {
        // original condition was (res->rank - a->rank > dim) but we cannot safely do
        // subtractions here because size_t would underflow so i reformulated the inequality
        //               [pad with zeros to the left]     [when shape[dim] is 1 we can't step to the next element, so set stride to 0]
        strides_a[dim] = (res->rank > dim + _a->rank ? 0 : (_a->shape[dim - (res->rank - _a->rank)] == 1 ? 0 : _a->strides[dim - (res->rank - _a->rank)]));
        strides_b[dim] = (res->rank > dim + _b->rank ? 0 : (_b->shape[dim - (res->rank - _b->rank)] == 1 ? 0 : _b->strides[dim - (res->rank - _b->rank)]));
    }

    for (size_t i = 0; i < res->nelem; i++) {
        // get indices of a, b and result
        ia = _a->offset; ib = _b->offset; ires = res->offset; remainder = i;

        for (dim = res->rank; dim-- > 0;) {
            // index of current element on current axis 
            iaxis = remainder % res->shape[dim];
            remainder /= res->shape[dim];
            ia += iaxis * strides_a[dim];
            ib += iaxis * strides_b[dim];
            ires += iaxis * res->strides[dim];
        }

        float a = _a->data[_a->offset + ia], b = _b->data[_b->offset + ib];
        res->data[ires] ASSIGNMENT RESULT;
    }
}
]]]

@GENERATE (BROADCASTING_OP) [[[
    add_brc: a + b
    sub_brc: a - b
    mul_brc: a * b
    div_brc: a / b
    pow_brc: pow(a, b)
]]]

// // Regular broadcasting operations
// BROADCASTING_OP(add_brc, = a + b)
// BROADCASTING_OP(sub_brc, = a - b)
// BROADCASTING_OP(mul_brc, = a * b)
// BROADCASTING_OP(div_brc, = a / b)
// BROADCASTING_OP(pow_brc, = pow(a, b))

// // Accumulative broadcasting operations
// BROADCASTING_OP(add_brc_acc, += a + b)
// BROADCASTING_OP(sub_brc_acc, += a - b)
// BROADCASTING_OP(mul_brc_acc, += a * b)
// BROADCASTING_OP(div_brc_acc, += a / b)
// BROADCASTING_OP(pow_brc_acc, += pow(a, b))

#endif //CORE_BROADCASTING
