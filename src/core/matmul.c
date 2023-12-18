#ifndef CORE_MATMUL
#define CORE_MATMUL

#include <stddef.h>

#define MAX(A, B) A > B ? A : B;

// performs standard matrix multiplication on a matrix
// that is located somewhere inside a larger tensor
void mul_mat(
    float* a, size_t nrow_a, size_t ncol_a, size_t base_a,
    float* b, size_t nrow_b, size_t ncol_b, size_t base_b,
    float* result, size_t base_result
) {
    register size_t r, c, i, ires, offset_row, ib;

    for (r = 0; r < nrow_a; r++) {
        for (c = 0; c < ncol_b; c++) {
            ires       = base_result + r * ncol_b + c;
            offset_row = base_a      + r * ncol_a;

            for (i = 0; i < ncol_a; i++) {
                ib = base_b + i * ncol_b + c;
                result[ires] += a[offset_row + i] * b[ib];
            }
        }
    }
}


// pairwise multiplication of the matrices in two tensors
void mul_tns(
    float* a, size_t nrow_a, size_t ncol_a, size_t nmat_a,
    float* b, size_t nrow_b, size_t ncol_b, size_t nmat_b,
    float* result
) {
    size_t stride_res = nrow_a * ncol_b;
    size_t stride_a = nmat_a > 1 ? nrow_a * ncol_a : 0;
    size_t stride_b = nmat_b > 1 ? nrow_b * ncol_b : 0;
    size_t ia = 0, ib = 0, ires = 0;
    size_t nmat_max = MAX(nmat_a, nmat_b);

    for (size_t i = 0; i < nmat_max; i++) {
        mul_mat(
            a, nrow_a, ncol_a, ia,
            b, nrow_b, ncol_b, ib,
            result, ires);

        ia += stride_a; // todo: minimal optimization potential (if stride = 0, add useless)
        ib += stride_b;
        ires += stride_res;
    }
}

#endif //CORE_MATMUL