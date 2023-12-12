// pairwise operations on two arrays of identical size

#ifndef CORE_PAIRWISE
#define CORE_PAIRWISE

#include <stddef.h>

#define PARIWISE_OP(NAME, OP) \
    void NAME(float* a, float* b, size_t size) { \
        for (size_t i = 0; i < size; i++) a[i] OP##= b[i]; }

PARIWISE_OP(add_prw, +) // add
PARIWISE_OP(sub_prw, -) // sub
PARIWISE_OP(mul_prw, *) // mul
PARIWISE_OP(div_prw, /) // div

#endif //CORE_PAIRWISE
