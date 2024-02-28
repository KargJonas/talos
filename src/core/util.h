#ifndef CORE_UTIL
#define CORE_UTIL

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "./tensor.h"

#include <emscripten.h>

#define MAX(A, B) A > B ? A : B
#define print_js(msg, var) EM_ASM({ console.log(msg, $0); }, var);

// allocation functions

float* alloc_farr(size_t size);
size_t* alloc_starr(size_t size);


// array copy functions

void copy_farr(float* source, float* dest, size_t nelem);
void copy_starr(size_t* source, size_t* dest, size_t nelem);


// deallocation functions

void free_farr(float* ptr);
void free_starr(size_t* ptr);


// misc utility functions

float fast_inv_sqrt(float number);
size_t get_nsubtns(struct tensor_t *a, size_t n);

#endif //CORE_UTIL
