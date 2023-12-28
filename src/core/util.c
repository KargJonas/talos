#include "./util.h"

// allocation functions

float* alloc_farr(size_t size) {
    return (float*)malloc(size * sizeof(float));
}

size_t* alloc_starr(size_t size) {
    return (size_t*)malloc(size * sizeof(size_t));
}


// array copy functions

void copy_farr(float* source, float* dest, size_t nelem) {
    memcpy(dest, source, nelem * sizeof(float));
}

void copy_starr(size_t* source, size_t* dest, size_t nelem) {
    memcpy(dest, source, nelem * sizeof(size_t));
}


// deallocation functions

void free_farr(float* ptr) {
    free(ptr);
}

void free_starr(size_t* ptr) {
    free(ptr);
}

void free_tensor(struct tensor_t* tensor) {
    free_farr(tensor->data);
    free_starr(tensor->shape);
    free_starr(tensor->strides);
    free(tensor);
}


// misc utility functions

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

size_t get_ncols(struct tensor_t* a) {
    return a->shape[a->rank - 1];
}

size_t get_nrows(struct tensor_t* a) {
    if (a->rank < 2) return 1;
    return a->shape[a->rank - 2];
}

size_t get_nmat(struct tensor_t* a) {
    if (a->rank < 3) return 1;
    return a->shape[a->rank - 3];
}
