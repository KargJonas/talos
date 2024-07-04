// pairwise operations on two arrays of identical size

#ifndef CORE_UNARY_DBRC
#define CORE_UNARY_DBRC

#include <stddef.h>
#include <math.h>
#include <string.h>
#include "./util.h"
#include "./tensor.h"

#define DEBROADCASTING_UNARY_OP(NAME, ASSIGNMENT, RESULT) [[[
void NAME(struct tensor_t *_a, struct tensor_t *dest) {
    size_t diff = dest->nelem == 1 ? _a->rank : _a->rank - dest->rank, n_elem_var_shape = 1;

    // compute number of elements of the source tensor
    // that sum up to one element of the dest tensor
    for (size_t i = 0; i < diff; i++) {
        n_elem_var_shape *= _a->shape[i];
    }

    // iterate over all elements in the destination tensor
    for (size_t i = 0; i < dest->nelem; i++) {
        size_t src_base_coord = _a->offset;
        size_t remainder = i;
        size_t dest_coord = dest->offset;
        float a = 0;

        // find the linear index of the dest component and the base of the linear index
        // of each of the elements summing up to this destination component
        for (int dim = dest->rank - 1; dim >= 0; dim--) {
            size_t iaxis = remainder % dest->shape[dim];
            remainder = remainder / dest->shape[dim];
            src_base_coord += iaxis * _a->strides[diff + dim];
            dest_coord += iaxis * dest->strides[dim];
        }

        float sum = 0;

        // iterate over all elements in the source tensor that contribute to
        // the current element in the destination tensor and sum their values together
        // this is done by using the fixed base index src_base_coord and varying the
        // coordinates that are only present in the larger tensor
        for (size_t j = 0; j < n_elem_var_shape; j++) {
            size_t src_coord = src_base_coord, remainder = j;
            for (int dim = diff - 1; dim >= 0; dim--) {
                size_t iaxis = remainder % _a->shape[dim];
                remainder = remainder / _a->shape[dim];
                src_coord += iaxis * _a->strides[dim];
            }
            float a = _a->data[src_coord];
            sum += RESULT;
        }
        
        dest->data[dest_coord] ASSIGNMENT sum;
    }
}
]]]

@GENERATE (DEBROADCASTING_UNARY_OP) [[[
    sin_dbrc:        sin(a)
    cos_dbrc:        cos(a)
    tan_dbrc:        tan(a)
    asin_dbrc:       asin(a)
    acos_dbrc:       acos(a)
    atan_dbrc:       atan(a)
    sinh_dbrc:       sinh(a)
    cosh_dbrc:       cosh(a)
    tanh_dbrc:       tanh(a)
    exp_dbrc:        exp(a)
    log_dbrc:        log(a)
    log2_dbrc:       log2(a)
    log10_dbrc:      log10(a)
    invsqrt_dbrc:    fast_inv_sqrt(a)
    sqrt_dbrc:       sqrt(a)
    ceil_dbrc:       ceil(a)
    floor_dbrc:      floor(a)
    abs_dbrc:        fabs(a)
    negate_dbrc:     -a
    reciprocal_dbrc: 1. / a
    relu_dbrc:       a < 0 ? 0 : a
    binstep_dbrc:    a < 0 ? 0 : 1
    logistic_dbrc:   1. / (exp(-a) + 1.)
]]]

#endif //CORE_UNARY_DBRC