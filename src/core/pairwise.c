// pairwise operations on two arrays of identical size

#ifndef CORE_PAIRWISE
#define CORE_PAIRWISE

#include <stddef.h>

#define PARIWISE_OP(NAME, OP) \
    void NAME(float* a, float* b, size_t size) { \
        for (size_t i = 0; i < size; i++) a[i] OP##= b[i]; }

PARIWISE_OP(prw_add, +) // add
PARIWISE_OP(prw_sub, -) // sub
PARIWISE_OP(prw_mul, *) // mul
PARIWISE_OP(prw_div, /) // div

#endif //CORE_PAIRWISE
