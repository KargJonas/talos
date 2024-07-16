#ifndef CORE_REDUCE
#define CORE_REDUCE

#include "float.h"
#include "util.h"

// these functions return scalar values directly
// the in-place reduce operations are implemented below

float max_red_scl(struct tensor_t* a) {
    register float val, max = FLT_MIN;

    for (size_t ires = 0; ires < a->nelem; ires++) {
        val = get_item(a, ires);
        if (val > max) max = val;
    }

    return max;
}

float min_red_scl(struct tensor_t* a) {
    register float val, min = FLT_MAX;

    for (size_t ires = 0; ires < a->nelem; ires++) {
        val = get_item(a, ires);
        if (val < min) min = val;
    }

    return min;
}

float sum_red_scl(struct tensor_t* a) {
    register float sum = 0;

    for (size_t ires = 0; ires < a->nelem; ires++) {
        sum += get_item(a, ires);
    }

    return sum;
}

// doing this in a way that prevents overflow of float32 and also
// reduces precision losses but is slightly inefficient
float mean_red_scl(struct tensor_t* a) {
    register float mean = 0;

    for (size_t ires = 0; ires < a->nelem; ires++) {
        mean = (mean * ires + get_item(a, ires)) / (ires + 1);
    }

    return mean;
}

/**
 * @brief Finds the largest element in a tensor.
 * The destination must be a scalar view of the source.
 * It will reference the largest element in the source tensor.
 * @param src Source tensor.
 * @param dest Scalar destination tensor.
 *             Must either be a scalar view of the source.
 */
// void max_red_tns(struct tensor_t* src, struct tensor_t* dest) {
//     register size_t index, max_index = src->offset;

//     for (size_t i = 0; i < src->nelem; i++) {
//         index = get_index(src, i);
//         if (src->data[index] > src->data[max_index]) max_index = index;
//     }

//     dest->offset = max_index;    
// }

// void max_red_tns(struct tensor_t* src, struct tensor_t* dest) {
//     register size_t index, max_index = src->offset;

//     for (size_t linear_index = 0; linear_index < src->nelem; linear_index++) {
//         size_t remainder = linear_index;
//         size_t iaxis;
//         index = src->offset;

//         for (size_t dim = src->rank; dim-- > 0;) {
//             iaxis = remainder % src->shape[dim];
//             index += iaxis * src->strides[dim];
//             remainder /= src->shape[dim];
//             dest->pos[dim] = iaxis - 1;
//         }

//         if (src->data[index] > src->data[max_index]) max_index = index;
//     }

//     dest->offset = max_index;    
// }

// -- NOTE --
// for tensors with gradients, we need to update the gradient mask during the
// the forward pass. this means, we need access to the gradient during the actual
// max operation.
// currently this is done by passing in a reference to the gradient explicitly,
// in the future, it might make sense to add a reference directly in the tensor
// struct

/**
 * @brief Takes a source tensor, finds it's largest element and sets a destination
 *        tensor (scalar view of source) to point to that largest element.
 *        At the same time, sets the gradient of the source (a scalar view of the
 *        gradient of the destination) to point to the same location the max element
 *        resides at (in the source tensor), but instead of referencing the primal,
 *        we reference the gradient of the source.
 * 
 * @param src Arbitrary input tensor, that we want to find the smallest element of
 * @param dest Scalar view of src.
 * @param grad_src Gradient of input tensor. Must have the same shape as the input tensor.
 * @param grad_dest Scalar view of grad_src.
 */
void max_red_tns(struct tensor_t* src, struct tensor_t* dest, struct tensor_t* grad_src, struct tensor_t* grad_dest) {
    register size_t index, max_index = src->offset, grad_index = grad_src->offset, grad_max_index;

    for (size_t linear_index = 0; linear_index < src->nelem; linear_index++) {
        size_t remainder = linear_index, remainder_grad = linear_index;
        size_t iaxis, iaxis_grad;
        index = src->offset;
        grad_index = grad_src->offset;

        for (size_t dim = src->rank; dim-- > 0;) {
            iaxis = remainder % src->shape[dim];
            iaxis_grad = remainder_grad % grad_src->shape[dim];
            index += iaxis * src->strides[dim];
            grad_index += iaxis_grad * grad_src->strides[dim];
            remainder /= src->shape[dim];
            remainder_grad /= grad_src->shape[dim];
        }

        if (src->data[index] > src->data[max_index]) {
            max_index = index;
            grad_max_index = grad_index;
        }
    }

    dest->offset = max_index;
    grad_dest->offset = grad_max_index;
}

void min_red_tns(struct tensor_t* src, struct tensor_t* dest) {
    register size_t index, min_index = src->offset;

    for (size_t i = 0; i < src->nelem; i++) {
        index = get_index(src, i);
        if (src->data[index] < src->data[min_index]) min_index = index;
    }

    dest->offset = min_index;    
}

void sum_red_tns(struct tensor_t* src, struct tensor_t* dest) {
    register float sum = 0;

    for (size_t ires = 0; ires < src->nelem; ires++) {
        sum += get_item(src, ires);
    }

    dest->data[get_index(dest, 0)] = sum;
}

// doing this in a way that prevents overflow of float32 and also
// reduces precision losses but is slightly inefficient
void mean_red_tns(struct tensor_t* src, struct tensor_t* dest) {
    register float mean = 0;

    for (size_t ires = 0; ires < src->nelem; ires++) {
        mean = (mean * ires + get_item(src, ires)) / (ires + 1);
    }

    dest->data[get_index(dest, 0)] = mean;
}

#endif//CORE_REDUCE
