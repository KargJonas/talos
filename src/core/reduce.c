#ifndef CORE_REDUCE
#define CORE_REDUCE

#include "float.h"
#include "util.h"

float max_red(struct tensor_t* a) {
    float val, max = FLT_MIN;
    
    for (size_t ires = 0; ires < a->nelem; ires++) {
        val = a->data[get_index(a, ires)];
        if (val > max) max = val;
    }

    return max;
}

float min_red(struct tensor_t* a) {
    float val, min = FLT_MAX;
    
    for (size_t ires = 0; ires < a->nelem; ires++) {
        val = a->data[get_index(a, ires)];
        if (val < min) min = val;
    }

    return min;
}

float sum_red(struct tensor_t* a) {
    float sum = 0;
    
    for (size_t ires = 0; ires < a->nelem; ires++) {
        sum += a->data[get_index(a, ires)];
    }

    return sum;
}

#endif//CORE_REDUCE
