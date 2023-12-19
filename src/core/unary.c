// pairwise operations on two arrays of identical size

#ifndef CORE_UNARY
#define CORE_UNARY

#include <stddef.h>
#include <math.h>
#include <wasm_simd128.h>

#define UNARY_OP(NAME, OP) \
    void NAME(float* a, size_t size) { \
        for (size_t i = 0; i < size; i++) OP(a[i]); }

void act_relu(float* a, size_t size) {
    for (size_t i = 0; i < size; i++) a[i] = a[i] < 0 ? 0 : a[i];
}

// void act_relu_simd(float* a, size_t size) {
//     // Process four elements at a time with SIMD
//     size_t i;
//     for (i = 0; i < size / 4 * 4; i += 4) {
//         v128_t vec = wasm_v128_load(a + i);                 // Load vector from array
//         v128_t zero_vec = wasm_f32x4_splat(0.0f);          // Create a vector of zeros
//         v128_t result = wasm_f32x4_max(vec, zero_vec);     // ReLU: max(vec, 0)
//         wasm_v128_store(a + i, result);                    // Store the result back into the array
//     }

//     // Process the remaining elements
//     for (; i < size; ++i) {
//         a[i] = a[i] < 0 ? 0 : a[i];
//     }
// }

void act_tanh(float* a, size_t size) {
    for (size_t i = 0; i < size; i++) a[i] = tanh(a[i]); 
}

#endif //CORE_UNARY
