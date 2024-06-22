#ifndef CORE_DEBROADCAST
#define CORE_DEBROADCAST

#include "float.h"
#include "util.h"

// this function sums along the appropriate axis such that a larger tensor a
// can be "de-broadcasted" into a smaller tensor res.
// e.g. if a is of shape [5, 2, 9] and be of shape [2, 9], then
// sum tensor a along the axis of size 5 such that we get a tensor [2, 9]
#define DEBROADCASTING_BINARY_OP(NAME, ASSIGNMENT, RESULT) \
void NAME(struct tensor_t *_a, struct tensor_t *_b, struct tensor_t *dest) { \
    size_t diff = _b->nelem == 1 ? _b->nelem : _a->rank - dest->rank, n_elem_var_shape = 1; \
    /* compute number of elements of the source tensor */ \
    /* that sum up to one element of the dest tensor */ \
    for (size_t i = 0; i < diff; i++) n_elem_var_shape *= _a->shape[i]; \
    /* iterate over all elements in the destination tensor */ \
    for (size_t i = 0; i < dest->nelem; i++) { \
        size_t src_base_coord = _a->offset, remainder = i, b_coord = _b->offset, dest_coord = dest->offset; \
        float a = 0; \
        /* find the linear index of the dest component and the base of the linear index */ \
        /* of each of the elements summing up to this destination component */ \
        for (int dim = dest->rank - 1; dim >= 0; dim--) { \
            size_t iaxis = remainder % dest->shape[dim]; \
            remainder = remainder / dest->shape[dim]; \
            src_base_coord += iaxis * _a->strides[diff + dim]; \
            b_coord += iaxis * _b->strides[dim]; \
            dest_coord += iaxis * dest->strides[dim]; \
        } \
        /* iterate over all elements in the source tensor that contribute to */ \
        /* the current element in the destination tensor and sum their values together */ \
        /* this is done by using the fixed base index src_base_coord and varying the */ \
        /* coordinates that are only present in the larger tensor */ \
        float sum = 0; \
        for (size_t j = 0; j < n_elem_var_shape; j++) { \
            size_t src_coord = src_base_coord, remainder = j; \
            for (int dim = diff - 1; dim >= 0; dim--) { \
                size_t iaxis = remainder % _a->shape[dim]; \
                remainder = remainder / _a->shape[dim]; \
                src_coord += iaxis * _a->strides[dim]; \
            } \
            float b = _b->data[b_coord]; \
            float a = _a->data[src_coord]; \
            sum += RESULT; \
        } \
        dest->data[dest_coord] ASSIGNMENT sum; \
    } \
}

// Accumulating debroadcasting operations (for grad accumulation)
// These work like so: destination_grad += grad_a <operation> grad_b
DEBROADCASTING_BINARY_OP(add_acc_dbrc, +=, a + b)
DEBROADCASTING_BINARY_OP(sub_acc_dbrc, +=, a - b)
DEBROADCASTING_BINARY_OP(mul_acc_dbrc, +=, a * b)
DEBROADCASTING_BINARY_OP(div_acc_dbrc, +=, a / b)
DEBROADCASTING_BINARY_OP(pow_acc_dbrc, +=, pow(a, b))

#endif//CORE_DEBROADCAST