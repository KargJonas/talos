// scalar operations on a single array

#ifndef CORE_SCALAR
#define CORE_SCALAR

#include <stddef.h>

#define SCALAR_OP(NAME, OP) \
    void NAME(float* a, float b, size_t size) { \
        for (size_t i = 0; i < size; i++) a[i] OP##= b; }

SCALAR_OP(add_scl, +) // add
SCALAR_OP(sub_scl, -) // sub
SCALAR_OP(mul_scl, *) // mul
SCALAR_OP(div_scl, /) // div

#endif //CORE_SCALAR
