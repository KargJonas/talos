#include "./tensor.h"

struct tensor_t* create_tensor() {
    // allocate memory for struct itself
    struct tensor_t* new_tensor = (struct tensor_t*)malloc(sizeof(struct tensor_t));    
    return new_tensor;
}

void copy_tensor(struct tensor_t* source, struct tensor_t* dest) {
    copy_farr(source->data, dest->data, source->nelem);
    copy_starr(source->shape, dest->shape, source->rank);
    copy_starr(source->strides, dest->strides, source->rank);
    dest->rank = source->rank;
    dest->nelem = source->nelem;
}
