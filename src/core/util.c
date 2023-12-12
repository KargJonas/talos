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

#endif //CORE_UTIL