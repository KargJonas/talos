#include <stddef.h>
#include <stdlib.h>

float* alloc_farr(size_t size) {
    return (float*)malloc(size * sizeof(float));
}

// Function to add two floating-point numbers
float* add(float* a, float b, size_t size) {
    for (size_t i = 0; i < size; i++) a[i] += b;
    return a;
}
