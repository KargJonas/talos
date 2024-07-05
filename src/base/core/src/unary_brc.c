// pairwise operations on two arrays of identical size

#ifndef CORE_UNARY_BRC
#define CORE_UNARY_BRC

#include <stddef.h>
#include <math.h>
#include <string.h>
#include "./util.h"
#include "./tensor.h"

#define BROADCASTING_UNARY_OP(NAME, ASSIGNMENT, RESULT) [[[
void NAME(struct tensor_t *_a, struct tensor_t *res) {
    size_t ia, ires, iaxis, remainder, dim;
    size_t strides_a[res->rank];

    // extend stride arrays of a and with zeros to match rank of result tensor
    for (dim = res->rank; dim-- > 0;) {
        // original condition was (res->rank - a->rank > dim) but we cannot safely do
        // subtractions here because size_t would underflow so i reformulated the inequality
        //               [pad with zeros to the left]     [when shape[dim] is 1 we can't step to the next element, so set stride to 0]
        strides_a[dim] = (res->rank > dim + _a->rank ? 0 : (_a->shape[dim - (res->rank - _a->rank)] == 1 ? 0 : _a->strides[dim - (res->rank - _a->rank)]));
    }

    for (size_t i = 0; i < res->nelem; i++) {
        // get indices of a and result
        ia = _a->offset; ires = res->offset; remainder = i;

        for (dim = res->rank; dim-- > 0;) {
            // index of current element on current axis 
            iaxis = remainder % res->shape[dim];
            remainder /= res->shape[dim];
            ia += iaxis * strides_a[dim];
            ires += iaxis * res->strides[dim];
        }

        float a = _a->data[_a->offset + ia];
        res->data[ires] ASSIGNMENT RESULT;
    }
}
]]]

@GENERATE (BROADCASTING_UNARY_OP) [[[
    sin_brc:        sin(a)
    cos_brc:        cos(a)
    tan_brc:        tan(a)
    asin_brc:       asin(a)
    acos_brc:       acos(a)
    atan_brc:       atan(a)
    sinh_brc:       sinh(a)
    cosh_brc:       cosh(a)
    tanh_brc:       tanh(a)
    exp_brc:        exp(a)
    log_brc:        log(a)
    log2_brc:       log2(a)
    log10_brc:      log10(a)
    invsqrt_brc:    fast_inv_sqrt(a)
    sqrt_brc:       sqrt(a)
    ceil_brc:       ceil(a)
    floor_brc:      floor(a)
    abs_brc:        fabs(a)
    sign_brc:       SIGN(a)
    negate_brc:     -a
    reciprocal_brc: 1. / a
    relu_brc:       a < 0 ? 0 : a
    binstep_brc:    a < 0 ? 0 : 1
    logistic_brc:   1. / (exp(-a) + 1.)
]]]

#endif //CORE_UNARY_BRC
