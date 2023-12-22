#ifndef CORE_RAND
#define CORE_RAND

#include <stddef.h>
#include <stdlib.h>

void rand_seed(int seed) {
    srand(seed);
}

void rand_f(float* a, size_t size, float min, float max) {
    float range = max - min;

    for (size_t i = 0; i < size; i++)
        a[i] = rand() / (float)RAND_MAX * range + min;
}

void rand_i(float* a, size_t size, int min, int max) {
    int range = max - min + 1;

    for (size_t i = 0; i < size; i++)
        a[i] = (rand() % range) + min;
}

void fill(float* a, size_t size, float value) {
    for (size_t i = 0; i < size; i++)
        a[i] = value;
}

#endif //CORE_RAND