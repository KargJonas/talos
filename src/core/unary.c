// pairwise operations on two arrays of identical size

#ifndef CORE_UNARY
#define CORE_UNARY

#include <stddef.h>
#include <math.h>
#include <string.h>
#include "./util.h"
#include "./tensor.h"

#define UNARY_OP(NAME, OP) \
    void NAME(struct tensor_t* a, struct tensor_t* res) { \
        for (size_t i = 0; i < a->nelem; i++) res->data[i] = OP(a->data[i]); }

// "tns" should signify that these are oparations
// on tensors, not scalars

UNARY_OP(negate_tns, -);

UNARY_OP(sin_tns, sin);
UNARY_OP(cos_tns, cos);
UNARY_OP(tan_tns, tan);
UNARY_OP(asin_tns, asin);
UNARY_OP(acos_tns, acos);
UNARY_OP(atan_tns, atan);
UNARY_OP(sinh_tns, sinh);
UNARY_OP(cosh_tns, cosh);
UNARY_OP(tanh_tns, tanh);

UNARY_OP(exp_tns, exp);
UNARY_OP(log_tns, log);
UNARY_OP(log10_tns, log10);
UNARY_OP(log2_tns, log2);

// potentially dangerous if user expects precision of (1. / sqrt())
UNARY_OP(invsqrt_tns, fast_inv_sqrt);
UNARY_OP(sqrt_tns, sqrt);

UNARY_OP(ceil_tns, ceil);
UNARY_OP(floor_tns, floor);
UNARY_OP(abs_tns, fabsf);

UNARY_OP(reciprocal_tns, 1./);

// somewhat unconventional unary op. equivalent to identity function
void identity_tns(struct tensor_t* a, struct tensor_t* res) {
    memcpy(res->data, a->data, a->nelem * sizeof(float));
}

void relu_tns(struct tensor_t* a, struct tensor_t* res) {
    for (size_t i = 0; i < a->nelem; i++) res->data[i] = a->data[i] < 0 ? 0 : a->data[i];
}

void binstep_tns(struct tensor_t* a, struct tensor_t* res) {
    for (size_t i = 0; i < a->nelem; i++) res->data[i] = a->data[i] < 0 ? 0 : 1;
}

void logistic_tns(struct tensor_t* a, struct tensor_t* res) {
    for (size_t i = 0; i < a->nelem; i++) res->data[i] = 1. / (exp(-a->data[i]) + 1.);
}

#endif //CORE_UNARY
