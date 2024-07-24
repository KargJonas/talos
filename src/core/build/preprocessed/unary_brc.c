// pairwise operations on two arrays of identical size

#ifndef CORE_UNARY_BRC
#define CORE_UNARY_BRC

#include <stddef.h>
#include <math.h>
#include <string.h>
#include "./util.h"
#include "./tensor.h"

// NOTE: param is an optional floating point value that may or may not be used
#define BROADCASTING_UNARY_OP(NAME, ASSIGNMENT, RESULT)  \
void NAME(struct tensor_t *_a, struct tensor_t *res, float param) { \
    size_t ia, ires, iaxis, remainder, dim; \
    size_t strides_a[res->rank]; \
 \
    /* // extend stride arrays of a and with zeros to match rank of result tensor */ \
    for (dim = res->rank; dim-- > 0;) { \
        /* // original condition was (res->rank - a->rank > dim) but we cannot safely do */ \
        /* // subtractions here because size_t would underflow so i reformulated the inequality */ \
        /* //               [pad with zeros to the left]     [when shape[dim] is 1 we can't step to the next element, so set stride to 0] */ \
        strides_a[dim] = (res->rank > dim + _a->rank ? 0 : (_a->shape[dim - (res->rank - _a->rank)] == 1 ? 0 : _a->strides[dim - (res->rank - _a->rank)])); \
    } \
 \
    for (size_t i = 0; i < res->nelem; i++) { \
        /* // get indices of a and result */ \
        ia = _a->offset; ires = res->offset; remainder = i; \
 \
        for (dim = res->rank; dim-- > 0;) { \
            /* // index of current element on current axis  */ \
            iaxis = remainder % res->shape[dim]; \
            remainder /= res->shape[dim]; \
            ia += iaxis * strides_a[dim]; \
            ires += iaxis * res->strides[dim]; \
        } \
 \
        float a = _a->data[_a->offset + ia]; \
        res->data[ires] ASSIGNMENT RESULT; \
    } \
} \


BROADCASTING_UNARY_OP(sin_brc, =, sin(a))
BROADCASTING_UNARY_OP(sin_brc_acc, +=, sin(a))

BROADCASTING_UNARY_OP(cos_brc, =, cos(a))
BROADCASTING_UNARY_OP(cos_brc_acc, +=, cos(a))

BROADCASTING_UNARY_OP(tan_brc, =, tan(a))
BROADCASTING_UNARY_OP(tan_brc_acc, +=, tan(a))

BROADCASTING_UNARY_OP(asin_brc, =, asin(a))
BROADCASTING_UNARY_OP(asin_brc_acc, +=, asin(a))

BROADCASTING_UNARY_OP(acos_brc, =, acos(a))
BROADCASTING_UNARY_OP(acos_brc_acc, +=, acos(a))

BROADCASTING_UNARY_OP(atan_brc, =, atan(a))
BROADCASTING_UNARY_OP(atan_brc_acc, +=, atan(a))

BROADCASTING_UNARY_OP(sinh_brc, =, sinh(a))
BROADCASTING_UNARY_OP(sinh_brc_acc, +=, sinh(a))

BROADCASTING_UNARY_OP(cosh_brc, =, cosh(a))
BROADCASTING_UNARY_OP(cosh_brc_acc, +=, cosh(a))

BROADCASTING_UNARY_OP(tanh_brc, =, tanh(a))
BROADCASTING_UNARY_OP(tanh_brc_acc, +=, tanh(a))

BROADCASTING_UNARY_OP(exp_brc, =, exp(a))
BROADCASTING_UNARY_OP(exp_brc_acc, +=, exp(a))

BROADCASTING_UNARY_OP(log_brc, =, log(a))
BROADCASTING_UNARY_OP(log_brc_acc, +=, log(a))

BROADCASTING_UNARY_OP(log2_brc, =, log2(a))
BROADCASTING_UNARY_OP(log2_brc_acc, +=, log2(a))

BROADCASTING_UNARY_OP(log10_brc, =, log10(a))
BROADCASTING_UNARY_OP(log10_brc_acc, +=, log10(a))

BROADCASTING_UNARY_OP(invsqrt_brc, =, fast_inv_sqrt(a))
BROADCASTING_UNARY_OP(invsqrt_brc_acc, +=, fast_inv_sqrt(a))

BROADCASTING_UNARY_OP(sqrt_brc, =, sqrt(a))
BROADCASTING_UNARY_OP(sqrt_brc_acc, +=, sqrt(a))

BROADCASTING_UNARY_OP(ceil_brc, =, ceil(a))
BROADCASTING_UNARY_OP(ceil_brc_acc, +=, ceil(a))

BROADCASTING_UNARY_OP(floor_brc, =, floor(a))
BROADCASTING_UNARY_OP(floor_brc_acc, +=, floor(a))

BROADCASTING_UNARY_OP(abs_brc, =, fabs(a))
BROADCASTING_UNARY_OP(abs_brc_acc, +=, fabs(a))

BROADCASTING_UNARY_OP(sign_brc, =, SIGN(a))
BROADCASTING_UNARY_OP(sign_brc_acc, +=, SIGN(a))

BROADCASTING_UNARY_OP(negate_brc, =, -a)
BROADCASTING_UNARY_OP(negate_brc_acc, +=, -a)

BROADCASTING_UNARY_OP(reciprocal_brc, =, 1. / a)
BROADCASTING_UNARY_OP(reciprocal_brc_acc, +=, 1. / a)

BROADCASTING_UNARY_OP(relu_brc, =, a < 0. ? 0. : a)
BROADCASTING_UNARY_OP(relu_brc_acc, +=, a < 0. ? 0. : a)

BROADCASTING_UNARY_OP(leaky_relu_brc, =, a < 0. ? param * a : a)
BROADCASTING_UNARY_OP(leaky_relu_brc_acc, +=, a < 0. ? param * a : a)

BROADCASTING_UNARY_OP(binstep_brc, =, a < 0. ? 0. : 1.)
BROADCASTING_UNARY_OP(binstep_brc_acc, +=, a < 0. ? 0. : 1.)

BROADCASTING_UNARY_OP(logistic_brc, =, 1. / (exp(-a) + 1.))
BROADCASTING_UNARY_OP(logistic_brc_acc, +=, 1. / (exp(-a) + 1.))

BROADCASTING_UNARY_OP(df_sin_brc, =, cos(a))
BROADCASTING_UNARY_OP(df_sin_brc_acc, +=, cos(a))

BROADCASTING_UNARY_OP(df_cos_brc, =, -sin(a))
BROADCASTING_UNARY_OP(df_cos_brc_acc, +=, -sin(a))

BROADCASTING_UNARY_OP(df_tan_brc, =, 1. / pow(cos(a), 2.))
BROADCASTING_UNARY_OP(df_tan_brc_acc, +=, 1. / pow(cos(a), 2.))

BROADCASTING_UNARY_OP(df_asin_brc, =, 1. / sqrt(1 - pow(a, 2.)))
BROADCASTING_UNARY_OP(df_asin_brc_acc, +=, 1. / sqrt(1 - pow(a, 2.)))

BROADCASTING_UNARY_OP(df_acos_brc, =, -1. / sqrt(1 - pow(a, 2.)))
BROADCASTING_UNARY_OP(df_acos_brc_acc, +=, -1. / sqrt(1 - pow(a, 2.)))

BROADCASTING_UNARY_OP(df_atan_brc, =, 1. / (pow(a, 2.) + 1.))
BROADCASTING_UNARY_OP(df_atan_brc_acc, +=, 1. / (pow(a, 2.) + 1.))

BROADCASTING_UNARY_OP(df_sinh_brc, =, cosh(a))
BROADCASTING_UNARY_OP(df_sinh_brc_acc, +=, cosh(a))

BROADCASTING_UNARY_OP(df_cosh_brc, =, sinh(a))
BROADCASTING_UNARY_OP(df_cosh_brc_acc, +=, sinh(a))

BROADCASTING_UNARY_OP(df_tanh_brc, =, 1. - pow(tanh(a), 2.))
BROADCASTING_UNARY_OP(df_tanh_brc_acc, +=, 1. - pow(tanh(a), 2.))

BROADCASTING_UNARY_OP(df_log_brc, =, 1. / a)
BROADCASTING_UNARY_OP(df_log_brc_acc, +=, 1. / a)

BROADCASTING_UNARY_OP(df_log2_brc, =, 1. / (a * log(2.)))
BROADCASTING_UNARY_OP(df_log2_brc_acc, +=, 1. / (a * log(2.)))

BROADCASTING_UNARY_OP(df_log10_brc, =, 1. / (a * log(10.)))
BROADCASTING_UNARY_OP(df_log10_brc_acc, +=, 1. / (a * log(10.)))

BROADCASTING_UNARY_OP(df_invsqrt_brc, =, -.5 / pow(a, 3. / 2.))
BROADCASTING_UNARY_OP(df_invsqrt_brc_acc, +=, -.5 / pow(a, 3. / 2.))

BROADCASTING_UNARY_OP(df_sqrt_brc, =, .5 / sqrt(a))
BROADCASTING_UNARY_OP(df_sqrt_brc_acc, +=, .5 / sqrt(a))

BROADCASTING_UNARY_OP(df_abs_brc, =, SIGN(a))
BROADCASTING_UNARY_OP(df_abs_brc_acc, +=, SIGN(a))

BROADCASTING_UNARY_OP(df_negate_brc, =, -1)
BROADCASTING_UNARY_OP(df_negate_brc_acc, +=, -1)

BROADCASTING_UNARY_OP(df_reciprocal_brc, =, -1. / pow(a, 2.))
BROADCASTING_UNARY_OP(df_reciprocal_brc_acc, +=, -1. / pow(a, 2.))

BROADCASTING_UNARY_OP(df_relu_brc, =, a < 0. ? 0. : 1.)
BROADCASTING_UNARY_OP(df_relu_brc_acc, +=, a < 0. ? 0. : 1.)

BROADCASTING_UNARY_OP(df_leaky_relu_brc, =, a < 0 ? param : 1)
BROADCASTING_UNARY_OP(df_leaky_relu_brc_acc, +=, a < 0 ? param : 1)

#endif //CORE_UNARY_BRC
