#ifndef CORE_REDUCE
#define CORE_REDUCE

#include "float.h"
#include "util.h"

float max_red(struct tensor_t* a) {
    register float val, max = FLT_MIN;
    
    for (size_t ires = 0; ires < a->nelem; ires++) {
        val = get_item(a, ires);
        if (val > max) max = val;
    }

    return max;
}

float min_red(struct tensor_t* a) {
    register float val, min = FLT_MAX;
    
    for (size_t ires = 0; ires < a->nelem; ires++) {
        val = get_item(a, ires);
        if (val < min) min = val;
    }

    return min;
}

float sum_red(struct tensor_t* a) {
    register float sum = 0;
    
    for (size_t ires = 0; ires < a->nelem; ires++) {
        sum += get_item(a, ires);
    }

    return sum;
}

// doing this in a way that prevents overflow of float32 and also
// reduces precision losses but is slightly inefficient
float mean_red(struct tensor_t* a) {
    register float mean = 0;
    
    for (size_t ires = 0; ires < a->nelem; ires++) {
        mean = (mean * ires + get_item(a, ires)) / (ires + 1);
    }

    return mean;
}

#endif//CORE_REDUCE
