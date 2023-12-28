// scalar operations on a single array

#ifndef CORE_SCALAR
#define CORE_SCALAR

#include <stddef.h>
#include <math.h>
#include "./tensor.h"

#define SCALAR_OP(NAME, OP) \
    void NAME(struct tensor_t* a, float b, struct tensor_t* res) { \
        for (size_t i = 0; i < a->nelem; i++) res->data[i] = a->data[i] OP b; }

SCALAR_OP(add_scl, +) // add
SCALAR_OP(sub_scl, -) // sub
SCALAR_OP(mul_scl, *) // mul
SCALAR_OP(div_scl, /) // div

void pow_scl(struct tensor_t* a, float b, struct tensor_t* res) {
    for (size_t i = 0; i < a->nelem; i++) res->data[i] = pow(a->data[i], b);
}

#endif //CORE_SCALAR
