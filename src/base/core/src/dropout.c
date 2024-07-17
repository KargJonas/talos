#ifndef CORE_DROPOUT
#define CORE_DROPOUT

#include <stdlib.h>
#include "./tensor.h"

#define DROPOUT_OP(NAME, ASSIGNMENT, RESULT) [[[
void NAME(struct tensor_t* _a, struct tensor_t* res, float _p, unsigned int seed) {
    // scale dropout probability from range [0, 1] to [0, RAND_MAX]
    int p = _p * (float)RAND_MAX;

    // scale inputs that were not set to 0 up by 1 / (1-p)
    float scale = 1. / (1. - _p);

    if (_a->isview || res->isview) {
        for (size_t i = 0; i < _a->nelem; i++) {
            float a = get_index(_a, i);
            res->data[get_index(res, i)] ASSIGNMENT RESULT;
        }

        return;
    }

    for (size_t i = 0; i < _a->nelem; i++) {
        float a = _a->data[i];
        res->data[i] ASSIGNMENT RESULT;
    }
}
]]]

@GENERATE (DROPOUT_OP) [[[
    dropout: rand_r(&seed) < p ? 0 : (a * scale)
]]] 

#endif //CORE_DROPOUT
