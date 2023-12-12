// scalar operations on a single array

#ifndef CORE_SCALAR
#define CORE_SCALAR

#include <stddef.h>

#define SCALAR_OP(NAME, OP) \
    void NAME(float* a, float b, size_t size) { \
        for (size_t i = 0; i < size; i++) a[i] OP##= b; }

SCALAR_OP(scl_add, +) // add
SCALAR_OP(scl_sub, -) // sub
SCALAR_OP(scl_mul, *) // mul
SCALAR_OP(scl_div, /) // div

#endif //CORE_SCALAR
