#ifndef CORE_TENSOR
#define CORE_TENSOR

#include <stddef.h>
#include <stdlib.h>
#include <stdbool.h>
#include "util.h"

struct tensor_t {
    float* data;
    size_t* shape;
    size_t* strides;
    size_t rank;
    size_t nelem;
    size_t offset;
    bool isview;
};

struct tensor_t* create_tensor(size_t rank, size_t nelem);
void clone_tensor(struct tensor_t* a, struct tensor_t* res);

#endif//CORE_TENSOR