#ifndef CORE_UTIL
#define CORE_UTIL

#include <stddef.h>
#include <stdlib.h>

struct sized_arr {
    size_t size;
    float* data[];
};

float* alloc_farr(size_t size) {
    return (float*)malloc(size * sizeof(float));
}

void free_farr(float* ptr) {
    free(ptr);
}

float fast_inv_sqrt(float number) {
    long i;
    float x2, y;
    const float threehalfs = 1.5F;
    x2 = number * 0.5F;
    y  = number;
    i  = *(long*)&y;                       // evil floating point bit level hacking
    i  = 0x5f3759df - (i >> 1);               // what the fuck?
    y  = *(float*)&i;
    y  = y * (threehalfs - (x2 * y * y));   // 1st iteration
    // y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed
    return y;
}

#endif //CORE_UTIL
