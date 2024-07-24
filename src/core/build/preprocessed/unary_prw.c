// pairwise operations on two arrays of identical size

#ifndef CORE_UNARY_PRW
#define CORE_UNARY_PRW

#include <stddef.h>
#include <math.h>
#include <string.h>
#include "./util.h"
#include "./tensor.h"

#define PAIRWISE_UNARY_OP(NAME, ASSIGNMENT, RESULT)  \
void NAME(struct tensor_t* _a, struct tensor_t* res, float param) { \
    if (_a->isview || res->isview) { \
        for (size_t i = 0; i < _a->nelem; i++) { \
            float a = get_index(_a, i); \
            res->data[get_index(res, i)] ASSIGNMENT RESULT; \
        } \
 \
        return; \
    } \
 \
    for (size_t i = 0; i < _a->nelem; i++) { \
        float a = _a->data[i]; \
        res->data[i] ASSIGNMENT RESULT; \
    } \
} \


PAIRWISE_UNARY_OP(sin_prw, =, sin(a))
PAIRWISE_UNARY_OP(sin_prw_acc, +=, sin(a))

PAIRWISE_UNARY_OP(cos_prw, =, cos(a))
PAIRWISE_UNARY_OP(cos_prw_acc, +=, cos(a))

PAIRWISE_UNARY_OP(tan_prw, =, tan(a))
PAIRWISE_UNARY_OP(tan_prw_acc, +=, tan(a))

PAIRWISE_UNARY_OP(asin_prw, =, asin(a))
PAIRWISE_UNARY_OP(asin_prw_acc, +=, asin(a))

PAIRWISE_UNARY_OP(acos_prw, =, acos(a))
PAIRWISE_UNARY_OP(acos_prw_acc, +=, acos(a))

PAIRWISE_UNARY_OP(atan_prw, =, atan(a))
PAIRWISE_UNARY_OP(atan_prw_acc, +=, atan(a))

PAIRWISE_UNARY_OP(sinh_prw, =, sinh(a))
PAIRWISE_UNARY_OP(sinh_prw_acc, +=, sinh(a))

PAIRWISE_UNARY_OP(cosh_prw, =, cosh(a))
PAIRWISE_UNARY_OP(cosh_prw_acc, +=, cosh(a))

PAIRWISE_UNARY_OP(tanh_prw, =, tanh(a))
PAIRWISE_UNARY_OP(tanh_prw_acc, +=, tanh(a))

PAIRWISE_UNARY_OP(exp_prw, =, exp(a))
PAIRWISE_UNARY_OP(exp_prw_acc, +=, exp(a))

PAIRWISE_UNARY_OP(log_prw, =, log(a))
PAIRWISE_UNARY_OP(log_prw_acc, +=, log(a))

PAIRWISE_UNARY_OP(log2_prw, =, log2(a))
PAIRWISE_UNARY_OP(log2_prw_acc, +=, log2(a))

PAIRWISE_UNARY_OP(log10_prw, =, log10(a))
PAIRWISE_UNARY_OP(log10_prw_acc, +=, log10(a))

PAIRWISE_UNARY_OP(invsqrt_prw, =, fast_inv_sqrt(a))
PAIRWISE_UNARY_OP(invsqrt_prw_acc, +=, fast_inv_sqrt(a))

PAIRWISE_UNARY_OP(sqrt_prw, =, sqrt(a))
PAIRWISE_UNARY_OP(sqrt_prw_acc, +=, sqrt(a))

PAIRWISE_UNARY_OP(ceil_prw, =, ceil(a))
PAIRWISE_UNARY_OP(ceil_prw_acc, +=, ceil(a))

PAIRWISE_UNARY_OP(floor_prw, =, floor(a))
PAIRWISE_UNARY_OP(floor_prw_acc, +=, floor(a))

PAIRWISE_UNARY_OP(abs_prw, =, fabs(a))
PAIRWISE_UNARY_OP(abs_prw_acc, +=, fabs(a))

PAIRWISE_UNARY_OP(sign_prw, =, SIGN(a))
PAIRWISE_UNARY_OP(sign_prw_acc, +=, SIGN(a))

PAIRWISE_UNARY_OP(negate_prw, =, -a)
PAIRWISE_UNARY_OP(negate_prw_acc, +=, -a)

PAIRWISE_UNARY_OP(reciprocal_prw, =, 1. / a)
PAIRWISE_UNARY_OP(reciprocal_prw_acc, +=, 1. / a)

PAIRWISE_UNARY_OP(relu_prw, =, a < 0 ? 0 : a)
PAIRWISE_UNARY_OP(relu_prw_acc, +=, a < 0 ? 0 : a)

PAIRWISE_UNARY_OP(leaky_relu_prw, =, a < 0 ? param * a : a)
PAIRWISE_UNARY_OP(leaky_relu_prw_acc, +=, a < 0 ? param * a : a)

PAIRWISE_UNARY_OP(binstep_prw, =, a < 0 ? 0 : 1)
PAIRWISE_UNARY_OP(binstep_prw_acc, +=, a < 0 ? 0 : 1)

PAIRWISE_UNARY_OP(logistic_prw, =, 1. / (exp(-a) + 1.))
PAIRWISE_UNARY_OP(logistic_prw_acc, +=, 1. / (exp(-a) + 1.))

PAIRWISE_UNARY_OP(df_sin_prw, =, cos(a))
PAIRWISE_UNARY_OP(df_sin_prw_acc, +=, cos(a))

PAIRWISE_UNARY_OP(df_cos_prw, =, -sin(a))
PAIRWISE_UNARY_OP(df_cos_prw_acc, +=, -sin(a))

PAIRWISE_UNARY_OP(df_tan_prw, =, 1. / pow(cos(a), 2.))
PAIRWISE_UNARY_OP(df_tan_prw_acc, +=, 1. / pow(cos(a), 2.))

PAIRWISE_UNARY_OP(df_asin_prw, =, 1. / sqrt(1 - pow(a, 2.)))
PAIRWISE_UNARY_OP(df_asin_prw_acc, +=, 1. / sqrt(1 - pow(a, 2.)))

PAIRWISE_UNARY_OP(df_acos_prw, =, -1. / sqrt(1 - pow(a, 2.)))
PAIRWISE_UNARY_OP(df_acos_prw_acc, +=, -1. / sqrt(1 - pow(a, 2.)))

PAIRWISE_UNARY_OP(df_atan_prw, =, 1. / (pow(a, 2.) + 1.))
PAIRWISE_UNARY_OP(df_atan_prw_acc, +=, 1. / (pow(a, 2.) + 1.))

PAIRWISE_UNARY_OP(df_sinh_prw, =, cosh(a))
PAIRWISE_UNARY_OP(df_sinh_prw_acc, +=, cosh(a))

PAIRWISE_UNARY_OP(df_cosh_prw, =, sinh(a))
PAIRWISE_UNARY_OP(df_cosh_prw_acc, +=, sinh(a))

PAIRWISE_UNARY_OP(df_tanh_prw, =, 1. - pow(tanh(a), 2.))
PAIRWISE_UNARY_OP(df_tanh_prw_acc, +=, 1. - pow(tanh(a), 2.))

PAIRWISE_UNARY_OP(df_log_prw, =, 1. / a)
PAIRWISE_UNARY_OP(df_log_prw_acc, +=, 1. / a)

PAIRWISE_UNARY_OP(df_log2_prw, =, 1. / (a * log(2.)))
PAIRWISE_UNARY_OP(df_log2_prw_acc, +=, 1. / (a * log(2.)))

PAIRWISE_UNARY_OP(df_log10_prw, =, 1. / (a * log(10.)))
PAIRWISE_UNARY_OP(df_log10_prw_acc, +=, 1. / (a * log(10.)))

PAIRWISE_UNARY_OP(df_invsqrt_prw, =, -.5 / pow(a, 3. / 2.))
PAIRWISE_UNARY_OP(df_invsqrt_prw_acc, +=, -.5 / pow(a, 3. / 2.))

PAIRWISE_UNARY_OP(df_sqrt_prw, =, .5 / sqrt(a))
PAIRWISE_UNARY_OP(df_sqrt_prw_acc, +=, .5 / sqrt(a))

PAIRWISE_UNARY_OP(df_abs_prw, =, SIGN(a))
PAIRWISE_UNARY_OP(df_abs_prw_acc, +=, SIGN(a))

PAIRWISE_UNARY_OP(df_negate_prw, =, -1)
PAIRWISE_UNARY_OP(df_negate_prw_acc, +=, -1)

PAIRWISE_UNARY_OP(df_reciprocal_prw, =, -1. / pow(a, 2.))
PAIRWISE_UNARY_OP(df_reciprocal_prw_acc, +=, -1. / pow(a, 2.))

PAIRWISE_UNARY_OP(df_relu_prw, =, a < 0. ? 0. : 1.)
PAIRWISE_UNARY_OP(df_relu_prw_acc, +=, a < 0. ? 0. : 1.)

PAIRWISE_UNARY_OP(df_leaky_relu_prw, =, a < 0 ? param : 1)
PAIRWISE_UNARY_OP(df_leaky_relu_prw_acc, +=, a < 0 ? param : 1)

#endif //CORE_UNARY_PRW
