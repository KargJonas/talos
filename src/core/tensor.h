#ifndef CORE_TENSOR
#define CORE_TENSOR

#include <stddef.h>
#include <stdlib.h>
#include <stdbool.h>
#include "util.h"

struct tensor_t {
    float* data;        // array that contains the actual tensor data/values
    size_t* shape;      // tensor shape of the form [..., n_matrices, n_rows, n_cols]
    size_t* strides;    // strides same length as shape, counted in elements, not bytes like NumPy
    size_t rank;        // rank of the tensor. dictates length of shape/strides arrays
    size_t nelem;       // number of elements in the tensor. product of all elements of the shape array
    size_t ndata;       // number of elements in the data array (of topmost parent tensor)
    size_t offset;      // if view: offset of this view inside the parent tensor in number of elements default 0
    bool isview;        // indicates if this tensor is a view of another tensor
};

struct tensor_t* create_tensor(size_t rank, size_t nelem);
struct tensor_t* create_view(struct tensor_t* source, size_t axis, size_t offset);
void free_tensor(struct tensor_t* a);
void clone_tensor(struct tensor_t* a, struct tensor_t* res);

#endif//CORE_TENSOR
