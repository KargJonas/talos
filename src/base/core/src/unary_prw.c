// pairwise operations on two arrays of identical size

#ifndef CORE_UNARY_PRW
#define CORE_UNARY_PRW

#include <stddef.h>
#include <math.h>
#include <string.h>
#include "./util.h"
#include "./tensor.h"

#define PAIRWISE_UNARY_OP(NAME, ASSIGNMENT, RESULT) [[[
void NAME(struct tensor_t* _a, struct tensor_t* res, float param) {
    if (_a->isview || res->isview) {
        for (size_t i = 0; i < _a->nelem; i++) {
            float a = get_index(_a, i);
            res->data[get_index(res, i)] ASSIGNMENT RESULT;
        }

        return;
    }

    for (size_t i = 0; i < _a->nelem; i++) {
        float a = _a->data[i];
        res->data[i] ASSIGNMENT RESULT;
    }
}
]]]

@GENERATE (PAIRWISE_UNARY_OP) [[[
    sin_prw:        sin(a)
    cos_prw:        cos(a)
    tan_prw:        tan(a)
    asin_prw:       asin(a)
    acos_prw:       acos(a)
    atan_prw:       atan(a)
    sinh_prw:       sinh(a)
    cosh_prw:       cosh(a)
    tanh_prw:       tanh(a)
    exp_prw:        exp(a)
    log_prw:        log(a)
    log2_prw:       log2(a)
    log10_prw:      log10(a)
    invsqrt_prw:    fast_inv_sqrt(a)
    sqrt_prw:       sqrt(a)
    ceil_prw:       ceil(a)
    floor_prw:      floor(a)
    abs_prw:        fabs(a)
    sign_prw:       SIGN(a)
    negate_prw:     -a
    reciprocal_prw: 1. / a
    relu_prw:       a < 0 ? 0 : a
    leaky_relu_prw: a < 0 ? param * a : a
    binstep_prw:    a < 0 ? 0 : 1
    logistic_prw:   1. / (exp(-a) + 1.)
]]]

#endif //CORE_UNARY_PRW
