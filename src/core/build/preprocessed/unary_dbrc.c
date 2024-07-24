// pairwise operations on two arrays of identical size

#ifndef CORE_UNARY_DBRC
#define CORE_UNARY_DBRC

#include <stddef.h>
#include <math.h>
#include <string.h>
#include "./util.h"
#include "./tensor.h"

#define DEBROADCASTING_UNARY_OP(NAME, ASSIGNMENT, RESULT)  \
void NAME(struct tensor_t *_a, struct tensor_t *dest, float param) { \
    size_t diff = dest->nelem == 1 ? _a->rank : _a->rank - dest->rank, n_elem_var_shape = 1; \
 \
    /* // compute number of elements of the source tensor */ \
    /* // that sum up to one element of the dest tensor */ \
    for (size_t i = 0; i < diff; i++) { \
        n_elem_var_shape *= _a->shape[i]; \
    } \
 \
    /* // iterate over all elements in the destination tensor */ \
    for (size_t i = 0; i < dest->nelem; i++) { \
        size_t src_base_coord = _a->offset; \
        size_t remainder = i; \
        size_t dest_coord = dest->offset; \
        float a = 0; \
 \
        /* // find the linear index of the dest component and the base of the linear index */ \
        /* // of each of the elements summing up to this destination component */ \
        for (int dim = dest->rank - 1; dim >= 0; dim--) { \
            size_t iaxis = remainder % dest->shape[dim]; \
            remainder = remainder / dest->shape[dim]; \
            src_base_coord += iaxis * _a->strides[diff + dim]; \
            dest_coord += iaxis * dest->strides[dim]; \
        } \
 \
        float sum = 0; \
 \
        /* // iterate over all elements in the source tensor that contribute to */ \
        /* // the current element in the destination tensor and sum their values together */ \
        /* // this is done by using the fixed base index src_base_coord and varying the */ \
        /* // coordinates that are only present in the larger tensor */ \
        for (size_t j = 0; j < n_elem_var_shape; j++) { \
            size_t src_coord = src_base_coord, remainder = j; \
            for (int dim = diff - 1; dim >= 0; dim--) { \
                size_t iaxis = remainder % _a->shape[dim]; \
                remainder = remainder / _a->shape[dim]; \
                src_coord += iaxis * _a->strides[dim]; \
            } \
            float a = _a->data[src_coord]; \
            sum += RESULT; \
        } \
         \
        dest->data[dest_coord] ASSIGNMENT sum; \
    } \
} \


DEBROADCASTING_UNARY_OP(sin_dbrc, =, sin(a))
DEBROADCASTING_UNARY_OP(sin_dbrc_acc, +=, sin(a))

DEBROADCASTING_UNARY_OP(cos_dbrc, =, cos(a))
DEBROADCASTING_UNARY_OP(cos_dbrc_acc, +=, cos(a))

DEBROADCASTING_UNARY_OP(tan_dbrc, =, tan(a))
DEBROADCASTING_UNARY_OP(tan_dbrc_acc, +=, tan(a))

DEBROADCASTING_UNARY_OP(asin_dbrc, =, asin(a))
DEBROADCASTING_UNARY_OP(asin_dbrc_acc, +=, asin(a))

DEBROADCASTING_UNARY_OP(acos_dbrc, =, acos(a))
DEBROADCASTING_UNARY_OP(acos_dbrc_acc, +=, acos(a))

DEBROADCASTING_UNARY_OP(atan_dbrc, =, atan(a))
DEBROADCASTING_UNARY_OP(atan_dbrc_acc, +=, atan(a))

DEBROADCASTING_UNARY_OP(sinh_dbrc, =, sinh(a))
DEBROADCASTING_UNARY_OP(sinh_dbrc_acc, +=, sinh(a))

DEBROADCASTING_UNARY_OP(cosh_dbrc, =, cosh(a))
DEBROADCASTING_UNARY_OP(cosh_dbrc_acc, +=, cosh(a))

DEBROADCASTING_UNARY_OP(tanh_dbrc, =, tanh(a))
DEBROADCASTING_UNARY_OP(tanh_dbrc_acc, +=, tanh(a))

DEBROADCASTING_UNARY_OP(exp_dbrc, =, exp(a))
DEBROADCASTING_UNARY_OP(exp_dbrc_acc, +=, exp(a))

DEBROADCASTING_UNARY_OP(log_dbrc, =, log(a))
DEBROADCASTING_UNARY_OP(log_dbrc_acc, +=, log(a))

DEBROADCASTING_UNARY_OP(log2_dbrc, =, log2(a))
DEBROADCASTING_UNARY_OP(log2_dbrc_acc, +=, log2(a))

DEBROADCASTING_UNARY_OP(log10_dbrc, =, log10(a))
DEBROADCASTING_UNARY_OP(log10_dbrc_acc, +=, log10(a))

DEBROADCASTING_UNARY_OP(invsqrt_dbrc, =, fast_inv_sqrt(a))
DEBROADCASTING_UNARY_OP(invsqrt_dbrc_acc, +=, fast_inv_sqrt(a))

DEBROADCASTING_UNARY_OP(sqrt_dbrc, =, sqrt(a))
DEBROADCASTING_UNARY_OP(sqrt_dbrc_acc, +=, sqrt(a))

DEBROADCASTING_UNARY_OP(ceil_dbrc, =, ceil(a))
DEBROADCASTING_UNARY_OP(ceil_dbrc_acc, +=, ceil(a))

DEBROADCASTING_UNARY_OP(floor_dbrc, =, floor(a))
DEBROADCASTING_UNARY_OP(floor_dbrc_acc, +=, floor(a))

DEBROADCASTING_UNARY_OP(abs_dbrc, =, fabs(a))
DEBROADCASTING_UNARY_OP(abs_dbrc_acc, +=, fabs(a))

DEBROADCASTING_UNARY_OP(sign_dbrc, =, SIGN(a))
DEBROADCASTING_UNARY_OP(sign_dbrc_acc, +=, SIGN(a))

DEBROADCASTING_UNARY_OP(negate_dbrc, =, -a)
DEBROADCASTING_UNARY_OP(negate_dbrc_acc, +=, -a)

DEBROADCASTING_UNARY_OP(reciprocal_dbrc, =, 1. / a)
DEBROADCASTING_UNARY_OP(reciprocal_dbrc_acc, +=, 1. / a)

DEBROADCASTING_UNARY_OP(relu_dbrc, =, a < 0 ? 0 : a)
DEBROADCASTING_UNARY_OP(relu_dbrc_acc, +=, a < 0 ? 0 : a)

DEBROADCASTING_UNARY_OP(leaky_relu_dbrc, =, a < 0 ? param * a : a)
DEBROADCASTING_UNARY_OP(leaky_relu_dbrc_acc, +=, a < 0 ? param * a : a)

DEBROADCASTING_UNARY_OP(binstep_dbrc, =, a < 0 ? 0 : 1)
DEBROADCASTING_UNARY_OP(binstep_dbrc_acc, +=, a < 0 ? 0 : 1)

DEBROADCASTING_UNARY_OP(logistic_dbrc, =, 1. / (exp(-a) + 1.))
DEBROADCASTING_UNARY_OP(logistic_dbrc_acc, +=, 1. / (exp(-a) + 1.))

DEBROADCASTING_UNARY_OP(df_sin_dbrc, =, cos(a))
DEBROADCASTING_UNARY_OP(df_sin_dbrc_acc, +=, cos(a))

DEBROADCASTING_UNARY_OP(df_cos_dbrc, =, -sin(a))
DEBROADCASTING_UNARY_OP(df_cos_dbrc_acc, +=, -sin(a))

DEBROADCASTING_UNARY_OP(df_tan_dbrc, =, 1. / pow(cos(a), 2.))
DEBROADCASTING_UNARY_OP(df_tan_dbrc_acc, +=, 1. / pow(cos(a), 2.))

DEBROADCASTING_UNARY_OP(df_asin_dbrc, =, 1. / sqrt(1 - pow(a, 2.)))
DEBROADCASTING_UNARY_OP(df_asin_dbrc_acc, +=, 1. / sqrt(1 - pow(a, 2.)))

DEBROADCASTING_UNARY_OP(df_acos_dbrc, =, -1. / sqrt(1 - pow(a, 2.)))
DEBROADCASTING_UNARY_OP(df_acos_dbrc_acc, +=, -1. / sqrt(1 - pow(a, 2.)))

DEBROADCASTING_UNARY_OP(df_atan_dbrc, =, 1. / (pow(a, 2.) + 1.))
DEBROADCASTING_UNARY_OP(df_atan_dbrc_acc, +=, 1. / (pow(a, 2.) + 1.))

DEBROADCASTING_UNARY_OP(df_sinh_dbrc, =, cosh(a))
DEBROADCASTING_UNARY_OP(df_sinh_dbrc_acc, +=, cosh(a))

DEBROADCASTING_UNARY_OP(df_cosh_dbrc, =, sinh(a))
DEBROADCASTING_UNARY_OP(df_cosh_dbrc_acc, +=, sinh(a))

DEBROADCASTING_UNARY_OP(df_tanh_dbrc, =, 1. - pow(tanh(a), 2.))
DEBROADCASTING_UNARY_OP(df_tanh_dbrc_acc, +=, 1. - pow(tanh(a), 2.))

DEBROADCASTING_UNARY_OP(df_log_dbrc, =, 1. / a)
DEBROADCASTING_UNARY_OP(df_log_dbrc_acc, +=, 1. / a)

DEBROADCASTING_UNARY_OP(df_log2_dbrc, =, 1. / (a * log(2.)))
DEBROADCASTING_UNARY_OP(df_log2_dbrc_acc, +=, 1. / (a * log(2.)))

DEBROADCASTING_UNARY_OP(df_log10_dbrc, =, 1. / (a * log(10.)))
DEBROADCASTING_UNARY_OP(df_log10_dbrc_acc, +=, 1. / (a * log(10.)))

DEBROADCASTING_UNARY_OP(df_invsqrt_dbrc, =, -.5 / pow(a, 3. / 2.))
DEBROADCASTING_UNARY_OP(df_invsqrt_dbrc_acc, +=, -.5 / pow(a, 3. / 2.))

DEBROADCASTING_UNARY_OP(df_sqrt_dbrc, =, .5 / sqrt(a))
DEBROADCASTING_UNARY_OP(df_sqrt_dbrc_acc, +=, .5 / sqrt(a))

DEBROADCASTING_UNARY_OP(df_abs_dbrc, =, SIGN(a))
DEBROADCASTING_UNARY_OP(df_abs_dbrc_acc, +=, SIGN(a))

DEBROADCASTING_UNARY_OP(df_negate_dbrc, =, -1)
DEBROADCASTING_UNARY_OP(df_negate_dbrc_acc, +=, -1)

DEBROADCASTING_UNARY_OP(df_reciprocal_dbrc, =, -1. / pow(a, 2.))
DEBROADCASTING_UNARY_OP(df_reciprocal_dbrc_acc, +=, -1. / pow(a, 2.))

DEBROADCASTING_UNARY_OP(df_relu_dbrc, =, a < 0. ? 0. : 1.)
DEBROADCASTING_UNARY_OP(df_relu_dbrc_acc, +=, a < 0. ? 0. : 1.)

DEBROADCASTING_UNARY_OP(df_leaky_relu_dbrc, =, a < 0 ? param : 1)
DEBROADCASTING_UNARY_OP(df_leaky_relu_dbrc_acc, +=, a < 0 ? param : 1)

#endif //CORE_UNARY_DBRC
