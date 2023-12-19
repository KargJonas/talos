// pairwise operations on two arrays of identical size

#ifndef CORE_PAIRWISE
#define CORE_PAIRWISE

#include <stddef.h>
#include "./util.c"
#include <stdio.h>

#define PARIWISE_OP(NAME, OP) \
    void NAME(float* a, float* b, size_t size) { \
        for (size_t i = 0; i < size; i++) a[i] OP##= b[i]; }

PARIWISE_OP(add_prw, +) // add
PARIWISE_OP(sub_prw, -) // sub
PARIWISE_OP(mul_prw, *) // mul
PARIWISE_OP(div_prw, /) // div

void prw_op_broadcast(
    float* a, float* b, float* res, 
    size_t* strides, size_t rank
) {
    size_t* strides_a = strides;
    size_t* strides_b = strides + rank;
    size_t* shape_res = strides_b + rank;
    size_t location[rank];

    // computet total number of elements of the resuting tensor
    size_t total_elements = 1;
    for (size_t i = 0; i < rank; ++i)
        total_elements *= shape_res[i];

    // iterate over flat result tensor
    for (size_t linear_index = 0; linear_index < total_elements; linear_index++) {

        size_t index_a = 0, index_b = 0;
        size_t stride_a = 1, stride_b = 1;

        // calculate n-dim location from flat index
        size_t remainder = linear_index;
        for (int dim = rank - 1; dim >= 0; --dim) {
            location[dim] = remainder % shape_res[dim];
            remainder /= shape_res[dim];
        }

        // Convert multidimensional indices to linear indices for a and b

        for (int dim = rank - 1; dim >= 0; --dim) {
            index_a += location[dim] * strides_a[dim];
            index_b += location[dim] * strides_b[dim];
            stride_a *= (dim > 0 ? shape_res[dim - 1] : 1); // Correct the stride calculation
            stride_b *= (dim > 0 ? shape_res[dim - 1] : 1);
        }

        // Perform the operation
        res[linear_index] = a[index_a] + b[index_b];
    }
}

#endif //CORE_PAIRWISE
