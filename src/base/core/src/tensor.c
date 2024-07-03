#include "./tensor.h"
#include "./util.h"
#include "./mgmt.h"

// creates a tensor and allocates all necessary memory
// this is how "base-tensors" are created, this means data will be allocated.
struct tensor_t* create_tensor(size_t rank, size_t nelem) {
    // allocate memory for the struct
    struct tensor_t* new_tensor = (struct tensor_t*)malloc(sizeof(struct tensor_t));

    // allocate memory for data/shape/strides
    new_tensor->data = alloc_farr(nelem);
    new_tensor->shape = alloc_starr(rank);
    new_tensor->strides = alloc_starr(rank);

    // set default values for other metadata
    new_tensor->rank = rank;
    new_tensor->nelem = nelem;
    new_tensor->ndata = nelem;
    new_tensor->offset = 0;
    new_tensor->isview = false;
    new_tensor->size = sizeof(struct tensor_t) + sizeof(size_t) * rank * 2 + sizeof(float) * nelem;

    mgmt.allocated += new_tensor->size;
    mgmt.ntensors++;

    return new_tensor;
}

// creates a new tensor that references the data of another tensor
// nice feature of this function - automatically resizes view tensor to size of elements
// of the desired axis so you can use this to e.g. iterate over larger tensors
struct tensor_t* create_view(struct tensor_t* source, size_t axis, size_t offset) {
    // allocate memory for the struct
    struct tensor_t* new_tensor = (struct tensor_t*)malloc(sizeof(struct tensor_t));

    // assertion: axis may not be larger than rank - 1

    // reference data of source tensor
    new_tensor->data    = source->data;

    // allocate memory for shape/strides
    new_tensor->shape   = alloc_starr(source->rank - axis);
    new_tensor->strides = alloc_starr(source->rank - axis);

    // copy shape/strides from source but omit first n (=axis) elements
    size_t new_rank = source->rank - axis;
    memcpy(new_tensor->shape,   &source->shape[axis],   new_rank * sizeof(size_t));
    memcpy(new_tensor->strides, &source->strides[axis], new_rank * sizeof(size_t));

    // set default values for view
    new_tensor->rank = source->rank - axis;
    new_tensor->nelem = get_nelem_of_axis_elements(source, axis);
    new_tensor->ndata = source->ndata;
    new_tensor->offset = source->offset + offset;
    new_tensor->isview = true;
    new_tensor->size = sizeof(struct tensor_t) + sizeof(size_t) * new_rank * 2;

    mgmt.allocated += new_tensor->size;
    mgmt.ntensors++;

    return new_tensor;
}

// creates a (potentially differently-ranked) view of a source tensor for use in reshape operations
// the shape and strides will not be set here as it is expected that these will be set from js
struct tensor_t* create_reshape_view(struct tensor_t* source, size_t rank) {
    // allocate memory for the struct
    struct tensor_t* new_tensor = (struct tensor_t*)malloc(sizeof(struct tensor_t));
    
    // reference data of source tensor
    new_tensor->data    = source->data;

    // allocate memory for shape/strides
    new_tensor->shape   = alloc_starr(rank);
    new_tensor->strides = alloc_starr(rank);

    // set default values for view
    new_tensor->rank = rank;
    new_tensor->nelem = source->nelem;
    new_tensor->ndata = source->ndata;
    new_tensor->offset = source->offset;
    new_tensor->isview = true;
    new_tensor->size = sizeof(struct tensor_t) + sizeof(size_t) * rank * 2;

    mgmt.allocated += new_tensor->size;
    mgmt.ntensors++;

    return new_tensor;
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
 // todo: rename this to copy_tensor
void clone_tensor(struct tensor_t* source, struct tensor_t* dest) {
    copy_starr(source->shape, dest->shape, source->rank);
    copy_starr(source->strides, dest->strides, source->rank);
    dest->rank = source->rank;
    dest->nelem = source->nelem;
    dest->ndata = source->ndata;
    dest->offset = source->offset;
    dest->isview = false;

    // source tensor is not a view, so we can naively copy the data
    if (!source->isview) {
        copy_farr(source->data, dest->data, source->nelem);
        return;
    }

    // cloning tensor from view

    size_t dim, ia, remainder, iaxis;

    // iterating over every element in the destination tensor
    for (size_t ires = 0; ires < source->nelem; ires++) {
        ia = source->offset;
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

    mgmt.allocated += dest->size;
    mgmt.ntensors++;
}

void free_tensor(struct tensor_t* a) {
    if (!a->isview) free(a->data);
    free(a->shape);
    free(a->strides);
    free(a);

    mgmt.allocated -= a->size;
    mgmt.ntensors--;
}
