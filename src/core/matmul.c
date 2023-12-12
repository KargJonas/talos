#ifndef CORE_MATMUL
#define CORE_MATMUL

#include <stddef.h>

void mul_mat(
    float* a, size_t nrow_a, size_t ncol_a,
    float* b, size_t nrow_b, size_t ncol_b,
    float* result
) {
    size_t r, c, i, index, r_offset, ib;

    for (r = 0; r < nrow_a; r++) {
        for (c = 0; c < ncol_b; c++) {
            index = r * ncol_b + c;
            r_offset = r * ncol_a;

            for (i = 0; i < ncol_a; i++) {
                ib = i * ncol_b + c;
                result[index] += a[r_offset + i] * b[ib];
            }
        }
    }
}

#endif //CORE_MATMUL