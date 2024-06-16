// pairwise operations on two arrays of identical size

#ifndef CORE_PAIRWISE
#define CORE_PAIRWISE

#include <stddef.h>
#include <stdio.h>
#include "./util.h"

#define PARIWISE_OP(NAME, RESULT) \
void NAME(struct tensor_t* _a, struct tensor_t* _b, struct tensor_t* res) { \
    for (size_t i = 0; i < _a->nelem; i++) { \
        float a = get_item(_a, i), b = get_item(_b, i); \
        res->data[get_index(res, i)] RESULT; \
    } \
}

// Regular pairwise operations
PARIWISE_OP(add_prw, = a + b)
PARIWISE_OP(sub_prw, = a - b)
PARIWISE_OP(mul_prw, = a * b)
PARIWISE_OP(div_prw, = a / b)
PARIWISE_OP(pow_prw, = pow(a, b))

// Accumulative pairwise operations
PARIWISE_OP(add_prw_acc, += a + b)
PARIWISE_OP(sub_prw_acc, += a - b)
PARIWISE_OP(mul_prw_acc, += a * b)
PARIWISE_OP(div_prw_acc, += a / b)
PARIWISE_OP(pow_prw_acc, += pow(a, b))

#endif //CORE_PAIRWISE
