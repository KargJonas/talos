#include "./tensor.h"
#include "./util.h"

struct tensor_t* create_tensor() {
    // allocate memory for struct itself
    struct tensor_t* new_tensor = (struct tensor_t*)malloc(sizeof(struct tensor_t));    
    return new_tensor;
}

// todo
// struct tensor_t* create_view(struct tensor_t* source, size_t axis, size_t offset) {
//     struct tensor_t* new_tensor = create_tensor();
//     new_tensor->data = source->data;
//     new_tensor->offset = source->offset + offset;

//     // the new shape/strides will be the source shape/strides but without
//     // the first n (=axis) elements

//     // assertion: axis may not be larger than rank - 1

//     new_tensor->shape = alloc_starr();
// }

void copy_tensor_metadata(struct tensor_t* source, struct tensor_t* dest) {
    copy_starr(source->shape, dest->shape, source->rank);
    copy_starr(source->strides, dest->strides, source->rank);
    dest->rank = source->rank;
    dest->nelem = source->nelem;
    dest->offset = source->offset;
    dest->isview = source->isview; // todo: validate
}

/**
 * with tensor views we can no longer just copy the data segment
 * naively because the elements of a subtensor may be distributed
 * sparsely throughout the supertensor. currently we are just
 * copying the first nelem elements from the data array.
 * 
 * to do this properly, we need to move through the tensor
 * using strides and insert the elements into the new tensor
 * contiguously. we could/should do this in column major order
 * and then update the strides of the resulting tensor accordingly.
 *
 * in most/many cases we do not need to do this whole debacle,
 * so we should mark tensors as view-tensors and only do this
 * when necessary
 */

void copy_tensor(struct tensor_t* source, struct tensor_t* dest) {
    copy_starr(source->shape, dest->shape, source->rank);
    copy_starr(source->strides, dest->strides, source->rank);
    dest->rank = source->rank;
    dest->nelem = source->nelem;
    dest->offset = source->offset;
    dest->isview = false; // todo: validate

    // source tensor is not a view, so we can naively copy the data
    if (!source->isview) {
        copy_farr(source->data, dest->data, source->nelem);
        return;
    }

    // cloning tensor from view

    size_t dim, ia, remainder, iaxis;

    // iterating over every element in the destination tensor
    for (size_t ires = 0; ires < source->nelem; ires++) {
        ia = source->offset; // todo validate
        remainder = ires;

        // iterating over every dimension to obtain the index
        // of the current element in the source tensor
        for (dim = source->rank; dim-- > 0;) {
            // index of current element on current axis
            iaxis = remainder % source->shape[dim];
            // ia += iaxis * strides_a[dim];
            ia += iaxis * source->strides[dim];
            remainder /= dest->shape[dim];
        }

        // populate destination tensor
        dest->data[ires] = source->data[ia];
    }

    dest->offset = 0;
    set_row_major(dest);
}

