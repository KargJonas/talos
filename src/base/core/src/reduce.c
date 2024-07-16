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

// finds the linear index where the largest element resides 
size_t max_red_idx(struct tensor_t* src) {
    register float cur, max = FLT_MIN;
    register size_t max_idx = 0;

    for (size_t i = 0; i < src->nelem; i++) {
        cur = get_item(src, i);
        if (cur > max) {
            max = cur;
            max_idx = i;
        }
    }

    return max_idx;  
}

// finds the linear index where the smallest element resides 
size_t min_red_idx(struct tensor_t* src) {
    register float cur, min = FLT_MAX;
    register size_t min_idx = 0;

    for (size_t i = 0; i < src->nelem; i++) {
        cur = get_item(src, i);
        if (cur < min) {
            min = cur;
            min_idx = i;
        }
    }

    return min_idx;  
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
