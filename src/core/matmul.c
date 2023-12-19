#ifndef CORE_MATMUL
#define CORE_MATMUL

#include <stddef.h>
#include "./util.c"

// performs standard matrix multiplication on a matrix
// that is located somewhere inside a larger tensor
void mul_mat(
    float* a, size_t nrow_a, size_t ncol_a,
    float* b, size_t nrow_b, size_t ncol_b,
    float* result
) {
    register size_t r, c, i, ires, offset_row, ib;

    // todo optimization potential: replace multiplications by looped increments

    for (r = 0; r < nrow_a; r++) {
        for (c = 0; c < ncol_b; c++) {
            ires       = r * ncol_b + c;
            offset_row = r * ncol_a;

            for (i = 0; i < ncol_a; i++) {
                ib = i * ncol_b + c;
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
    size_t nmat_max = MAX(nmat_a, nmat_b);
    size_t ia, ib, ires;

    for (size_t i = 0; i < nmat_max; i++) {
        mul_mat(
            a + ia, nrow_a, ncol_a,
            b + ib, nrow_b, ncol_b,
            result + ires);

        ia += stride_a; // todo: minimal optimization potential (if stride = 0, add useless)
        ib += stride_b;
        ires += stride_res;
    }
}

void dot_tns(
    float* a, size_t ncol_a, size_t nvec_a,
    float* b, size_t nrow_b, size_t ncol_b, size_t nmat_b,
    float* result
) {
    size_t stride_b = nrow_b * ncol_b; // number of elements in one matrix of b
    size_t iv, im, ia, ib, ires;

    for (iv = 0; iv < nvec_a; iv++) {
        for (im = 0; im < nmat_b; im++) {
            mul_mat(
                a + ia, 1, ncol_a,
                b + ib, nrow_b, ncol_b,
                result + ires);

            ib += stride_b; // step over to next matrix
            ires += ncol_b;
        }

        ia += ncol_a; // step over to next vector
        ib = 0;
    }
}

#endif //CORE_MATMUL