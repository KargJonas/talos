// pairwise operations on two arrays of identical size

#ifndef CORE_UNARY
#define CORE_UNARY

#include <stddef.h>
#include <math.h>
#include <string.h>
#include "./util.h"
#include "./tensor.h"

#define UNARY_OP(NAME, RESULT) [[[
void NAME(struct tensor_t* _a, struct tensor_t* res) {
    if (_a->isview || res->isview) {
        for (size_t i = 0; i < _a->nelem; i++) {
            float a = get_index(_a, i);
            res->data[get_index(res, i)] RESULT;
        }

        return;
    }

    for (size_t i = 0; i < _a->nelem; i++) {
        float a = _a->data[i];
        res->data[i] RESULT;
    }
}
]]]

@GENERATE_UNARY (UNARY_OP) [[[
    sin_tns:        sin(a)
    cos_tns:        cos(a)
    tan_tns:        tan(a)
    asin_tns:       asin(a)
    acos_tns:       acos(a)
    atan_tns:       atan(a)
    sinh_tns:       sinh(a)
    cosh_tns:       cosh(a)
    tanh_tns:       tanh(a)
    exp_tns:        exp(a)
    log_tns:        log(a)
    log2_tns:       log2(a)
    log10_tns:      log10(a)
    invsqrt_tns:    fast_inv_sqrt(a)
    sqrt_tns:       sqrt(a)
    ceil_tns:       ceil(a)
    floor_tns:      floor(a)
    abs_tns:        fabs(a)
    negate_tns:     -a
    reciprocal_tns: 1. / a
    relu_tns:       a < 0 ? 0 : a
    binstep_tns:    a < 0 ? 0 : 1
    logistic_tns:   1. / (exp(-a) + 1.)
]]]

#endif //CORE_UNARY
