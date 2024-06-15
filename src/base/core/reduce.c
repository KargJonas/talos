#ifndef CORE_REDUCE
#define CORE_REDUCE

#include "float.h"
#include "util.h"

#include <emscripten.h>

// this function sums along the appropriate axis such that a larger tensor a
// can be "de-broadcasted" into a smaller tensor res.
// e.g. if a is of shape [5, 2, 9] and be of shape [2, 9], then
// sum tensor a along the axis of size 5 such that we get a tensor [2, 9]

void debroadcast(struct tensor_t *src, struct tensor_t *dest) {
    size_t diff = src->rank - dest->rank;

    // todo use fast pairwise op when shapes match.
    // todo check that diff is not negative (cant broadcast)
    // todo check that rightmost dimensions match

    // compute number of elements of the source tensor
    // that sum up to one element of the dest tensor
    size_t n_elem_var_shape = 1;
    for (int i = 0; i < diff; i++) n_elem_var_shape *= src->shape[i];

    // iterate over all elements in the destination tensor
    for (int i = 0; i < dest->nelem; i++) {
        size_t src_base_coord = src->offset, remainder = i;
        size_t dest_coord = dest->offset;

        for (int dim = dest->rank - 1; dim >= 0; dim--) {
            size_t iaxis = remainder % dest->shape[dim];
            remainder = remainder / dest->shape[dim];
            src_base_coord += iaxis * src->strides[diff + dim];
            dest_coord += iaxis * dest->strides[dim];
        }

        float sum = 0;

        // iterate over all elements in the source tensor that contribute to
        // the current element in the destination tensor and sum their values together
        for (int j = 0; j < n_elem_var_shape; j++) {
            size_t src_coord = src_base_coord, remainder = j;

            for (int dim = diff - 1; dim >= 0; dim--) {
                size_t iaxis = remainder % src->shape[dim];
                remainder = remainder / src->shape[dim];
                src_coord += iaxis * src->strides[dim];
            }

            sum += src->data[src_coord];
        }

        dest->data[dest_coord] = sum;
    }
}


// void debroadcast(struct tensor_t *a, struct tensor_t *b, struct tensor_t *res) {
//     size_t i, ia, ires, remainder, dim;
//     size_t *strides_a, *strides_res;
//
//     // Use b's rank to allocate on stack if variable-length arrays are supported:
//     size_t tmp_strides_a[b->rank], tmp_strides_res[b->rank];
//     strides_a = tmp_strides_a;
//     strides_res = tmp_strides_res;
//
//     // Initialize result tensor data to zero
//     for (i = 0; i < res->nelem; i++) {
//         res->data[i] = 0;
//     }
//
//     // Compute strides in the original tensor `a` corresponding to the axes in `b`
//     for (dim = 0; dim < b->rank; dim++) {
//         if (dim < a->rank) {
//             strides_a[dim] = (a->shape[dim] == 1 ? 0 : a->strides[dim]);  // Skip stride if broadcasting
//         } else {
//             strides_a[dim] = 0; // No stride for non-existent dimensions in `a`
//         }
//         strides_res[dim] = (b->shape[dim] == 1 ? 0 : b->strides[dim]); // Stride for `res` is determined by `b`
//     }
//
//     // Sum along the axes of `a` to match the shape of `b`
//     for (i = 0; i < a->nelem; i++) {
//         ia = a->offset;
//         ires = res->offset;
//         remainder = i;
//
//         // Compute indices for both tensors
//         for (dim = 0; dim < b->rank; dim++) {
//             size_t iaxis = remainder % (dim < a->rank ? a->shape[dim] : 1);  // Use 1 for non-existent dimensions in `a`
//             remainder /= (dim < a->rank ? a->shape[dim] : 1);
//
//             ia += iaxis * strides_a[dim];
//             if (strides_res[dim] != 0) {
//                 ires += iaxis * strides_res[dim];
//             }
//         }
//
//         // Aggregate the result into the reduced tensor
//         res->data[ires] += a->data[ia];
//     }
// }


// void debroadcast(struct tensor_t *a, struct tensor_t *b, struct tensor_t *res) {
//     size_t i, ia, ires, remainder, dim;
//     size_t strides_a[b->rank], strides_res[b->rank];
//
//     // Initialize result tensor data to zero
//     for (i = 0; i < res->nelem; i++) {
//         res->data[i] = 0;
//     }
//
//     // Compute strides in the original tensor `a` corresponding to the axes in `b`
//     for (dim = 0; dim < b->rank; dim++) {
//         // Stride for `a` and `res` need to be computed from the original broadcasting logic
//         if (a->rank > dim) {
//             strides_a[dim] = (a->shape[dim] == 1 ? 0 : a->strides[dim]);
//         } else {
//             strides_a[dim] = 0; // Extra axes in `a` that were broadcasted from scalar values in `b`
//         }
//         strides_res[dim] = (b->shape[dim] == 1 ? 0 : b->strides[dim]); // Only step through dimensions in `b`
//     }
//
//     // Sum along the axes of `a` to match the shape of `b`
//     for (i = 0; i < a->nelem; i++) {
//         ia = a->offset;
//         ires = res->offset;
//         remainder = i;
//
//         // Compute indices for both tensors
//         for (dim = 0; dim < b->rank; dim++) {
//             size_t iaxis = remainder % a->shape[dim];
//             remainder /= a->shape[dim];
//
//             ia += iaxis * strides_a[dim];
//             if (strides_res[dim] != 0) {
//                 ires += iaxis * strides_res[dim];
//             }
//         }
//
//         // Aggregate the result into the reduced tensor
//         res->data[ires] += a->data[ia];
//     }
// }



// GOOD contender:
void sum_red_brc(struct tensor_t *a, struct tensor_t *res) {
    size_t ia, ires, remainder, dim, res_dim = 0;
    size_t res_strides[a->rank];

    // set up the corresponding strides in res for each dimension in a
    for (dim = 0; dim < a->rank; dim++) {
        res_strides[dim] = res_dim < res->rank && a->shape[dim] == res->shape[res_dim] ? res->strides[res_dim++] : 0;
    }

    // iterate over all elements in tensor a
    for (size_t i = 0; i < a->nelem; i++) {
        ia = a->offset; ires = res->offset; remainder = i;

        // calculate indices for the original and result tensors
        for (dim = 0; dim < a->rank; dim++) {
            size_t iaxis = remainder % a->shape[dim]; // Index along the current dimension
            remainder /= a->shape[dim];
            ia += iaxis * a->strides[dim];
            ires += iaxis * res_strides[dim];
        }

        // accumulate the value in the result tensor
        res->data[ires] += a->data[ia];
    }
}


//
// void debroadcast(struct tensor_t *a, struct tensor_t *b, struct tensor_t *res) {
//     // Initialize result tensor with zeros
//     for (size_t i = 0; i < res->nelem; i++) {
//         res->data[get_index(res, i)] = 0;
//     }
//
//     // Check if b is a scalar (rank 0)
//     if (b->rank == 0) {
//         // If b is scalar, sum all elements of a into the single element of res (if res is also scalar)
//         // or distribute across all elements of res according to res's shape
//         size_t ires;
//         for (size_t i = 0; i < a->nelem; i++) {
//             // Determine where to add a's element in res
//             ires = (res->nelem == 1) ? 0 : (i % res->nelem);
//             res->data[get_index(res, ires)] += a->data[get_index(a, i)];
//         }
//     } else {
//         // More general case: Handle when b is not a scalar
//         // Map reduction over b's shape onto a, then aggregate into res accordingly
//         size_t ia, ires, remainder, dim;
//         for (size_t i = 0; i < a->nelem; i++) {
//             ia = a->offset;
//             ires = res->offset;
//             remainder = i;
//             for (dim = 0; dim < a->rank; dim++) {
//                 size_t iaxis = remainder % a->shape[dim];
//                 remainder /= a->shape[dim];
//                 ia += iaxis * a->strides[dim];
//                 // Use b's dimensionality to determine res stride mapping
//                 if (dim < b->rank && b->shape[dim] != 1 && dim < res->rank) {
//                     ires += iaxis * res->strides[dim];
//                 }
//             }
//             res->data[ires] += a->data[ia];
//         }
//     }
// }


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


// these functions operate in-place on a source and a destination tensor tensor.
// the destination tensor should be a scalar tensor (only one component)

void max_red_tns(struct tensor_t* src, struct tensor_t* dest) {
    register float val, max = FLT_MIN;

    for (size_t ires = 0; ires < src->nelem; ires++) {
        val = get_item(src, ires);
        if (val > max) max = val;
    }

    dest->data[get_index(dest, 0)] = max;
}

void min_red_tns(struct tensor_t* src, struct tensor_t* dest) {
    register float val, min = FLT_MAX;

    for (size_t ires = 0; ires < src->nelem; ires++) {
        val = get_item(src, ires);
        if (val < min) min = val;
    }

    dest->data[get_index(dest, 0)] = min;
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
