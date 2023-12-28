#ifndef CORE_TENSOR
#define CORE_TENSOR

#include <stddef.h>
#include <stdlib.h>
#include "util.h"

struct tensor_t {
    float* data;
    size_t* shape;
    size_t* strides;
    size_t rank;
    size_t nelem;
};

struct tensor_t* create_tensor();
void copy_tensor(struct tensor_t* a, struct tensor_t* res);

#endif//CORE_TENSOR