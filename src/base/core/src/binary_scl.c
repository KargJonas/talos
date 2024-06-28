// scalar operations on a single array

#ifndef CORE_SCALAR
#define CORE_SCALAR

#include <stddef.h>
#include <math.h>
#include "./tensor.h"

#define SCALAR_OP(NAME, ASSIGNMENT, RESULT) [[[
void NAME(struct tensor_t* _a, float b, struct tensor_t* res) {
    // use expensive indexing if the tensor is non-contiguous
    if (_a->isview || res->isview) {
        for (size_t i = 0; i < _a->nelem; i++) {
            float a = _a->data[get_index(_a, i)];
            res->data[get_index(res, i)] ASSIGNMENT RESULT;
            return;
        }
    }

    // use faster indexing if tensor is contiguous
    for (size_t i = 0; i < _a->nelem; i++) {
        float a = _a->data[i];
        res->data[i] ASSIGNMENT RESULT;
    }
}
]]]

@GENERATE (SCALAR_OP) [[[
    add_scl: a + b
    sub_scl: a - b
    mul_scl: a * b
    div_scl: a / b
    pow_scl: pow(a, b)
]]]

#endif //CORE_SCALAR
