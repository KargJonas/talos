#ifndef CORE_RAND
#define CORE_RAND

#include <stddef.h>
#include <stdlib.h>

unsigned int global_seed = 0;

void init_uniform(struct tensor_t* a, float min, float max, unsigned int seed) {
    float range = max - min;

    if (a->isview) for (size_t i = 0; i < a->nelem; i++) {
        a->data[get_index(a, i)] = rand_r(&seed) / (float)RAND_MAX * range + min;
    }

    else for (size_t i = 0; i < a->nelem; i++) {
        a->data[i] = rand_r(&seed) / (float)RAND_MAX * range + min;
    }
}

void init_normal(struct tensor_t* a, float mean, float std_dev, unsigned int seed) {
    if (a->isview) for (size_t i = 0; i < a->nelem; i++) {
        a->data[get_index(a, i)] = normal(mean, std_dev, &seed);
    }

    else for (size_t i = 0; i < a->nelem; i++) {
        a->data[i] = normal(mean, std_dev, &seed);
    }
}

void init_fill(struct tensor_t* a, float value) {
    if (a->isview) for (size_t i = 0; i < a->nelem; i++) {
        a->data[get_index(a, i)] = value;
    }

    else for (size_t i = 0; i < a->nelem; i++) {
        a->data[i] = value;
    }
}

#endif //CORE_RAND
