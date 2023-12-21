// scalar operations on a single array

#ifndef CORE_SCALAR
#define CORE_SCALAR

#include <stddef.h>
#include <math.h>

#define SCALAR_OP(NAME, OP) \
    void NAME(float* a, float b, float* res, size_t size) { \
        for (size_t i = 0; i < size; i++) res[i] = a[i] OP b; }

SCALAR_OP(add_scl, +) // add
SCALAR_OP(sub_scl, -) // sub
SCALAR_OP(mul_scl, *) // mul
SCALAR_OP(div_scl, /) // div

void pow_scl(float* a, float b, float* res, size_t size) {
    for (size_t i = 0; i < size; i++) res[i] = pow(a[i], b);
}

#endif //CORE_SCALAR
