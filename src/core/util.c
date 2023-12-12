#ifndef CORE_UTIL
#define CORE_UTIL

#include <stddef.h>
#include <stdlib.h>

struct sized_arr {
    size_t size;
    float* data[];
};

EMSCRIPTEN_KEEPALIVE
float* alloc_farr(size_t size) {
    return (float*)malloc(size * sizeof(float));
}

#endif //CORE_UTIL