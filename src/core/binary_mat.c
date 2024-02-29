#ifndef CORE_MATMUL
#define CORE_MATMUL

#include <stddef.h>
#include "./util.h"
#include "./tensor.h"

#define get_shape_bwd(a, i) a->shape[a->rank - i - 1]
#define get_strides_bwd(a, i) a->strides[a->rank - i - 1]
#define get_ncols(a) get_shape_bwd(a, 0)
#define get_nrows(a) get_shape_bwd(a, 1)
#define get_colstride(a) get_strides_bwd(a, 0)
#define get_rowstride(a) get_strides_bwd(a, 1)

// we are tearing open a healed wound here
void mul_mat(struct tensor_t* a, struct tensor_t* b, struct tensor_t* res, bool dot) {
    size_t nrow_a = dot ? 1 : get_nrows(a); // todo remove redundancy
    size_t ncol_a = get_ncols(a);
    size_t nrow_b = get_nrows(b);
    size_t ncol_b = get_ncols(b);

    register size_t r, c, i, ires, offset_row, ia, ib;

    for (r = 0; r < nrow_a; r++) {
        for (c = 0; c < ncol_b; c++) {
            ires =  res->offset +
                    r * get_rowstride(res) +
                    c * get_colstride(res);

            // this loop could be easily parallelized because
            // of the inherent associativity of the summation
            for (i = 0; i < ncol_a; i++) {
                ia = a->offset +
                    r * get_rowstride(a) +
                    i * get_colstride(a);

                ib = b->offset +
                    i * get_rowstride(b) +
                    c * get_colstride(b);

                res->data[ires] += a->data[ia] * b->data[ib];
            }
        }
    }

    // todo optimization potential: replace multiplications by looped increments

    // for (size_t r = 0; r < nrow_a; r++) {
    //     for (c = 0; c < ncol_b; c++) {
    //         ires       = r * ncol_b + c;
    //         offset_row = r * ncol_a;

    //         for (i = 0; i < ncol_a; i++) {
    //             ib = i * ncol_b + c;
    //             // result[ires] += a[offset_row + i] * b[ib];
                
    //             res->data[ires] +=
    //                 a->data[a->offset + offset_row + i] *
    //                 b->data[b->offset + ib];
    //         }
    //     }
    // }
}

// pairwise multiplication of the matrices in two tensors
void mul_tns(struct tensor_t* a, struct tensor_t* b, struct tensor_t* result) {
    size_t nrow_a = get_nrows(a);
    size_t ncol_a = get_ncols(a);
    size_t nrow_b = get_nrows(b);
    size_t ncol_b = get_ncols(b);

    size_t nmat_a = get_nsubtns(a, 2);
    size_t nmat_b = get_nsubtns(b, 2);

    size_t stride_res = nrow_a * ncol_b; // todo: think i could replace this by using result->strides    // size_t stride_a = nmat_a > 1 ? nrow_a * ncol_a : 0;
    size_t stride_b = nmat_b > 1 ? nrow_b * ncol_b : 0;
    size_t stride_a = nmat_a > 1 ? nrow_a * ncol_a : 0;

    size_t nmat_max = MAX(nmat_a, nmat_b);
    register size_t ia = 0, ib = 0, ires = 0;

    fill(result->data, result->nelem, 0);

    struct tensor_t* view_a   = create_view(a, a->rank - 2, 0);
    struct tensor_t* view_b   = create_view(b, b->rank - 2, 0);
    struct tensor_t* view_res = create_view(result, result->rank - 2, 0);

    for (size_t i = 0; i < nmat_max; i++) {
         mul_mat(view_a, view_b, view_res, false);

        view_a->offset += stride_a;
        view_b->offset += stride_b;
        view_res->offset += stride_res;
    }

    free_tensor(view_a);
    free_tensor(view_b);
    free_tensor(view_res);
}

// numpy-style tensor multiplication
void dot_tns(struct tensor_t* a, struct tensor_t* b, struct tensor_t* result) {
    size_t ncol_a = get_ncols(a);
    size_t nvec_a = get_nsubtns(a, 1);

    size_t nrow_b = get_nrows(b);
    size_t ncol_b = get_ncols(b);
    size_t nmat_b = get_nsubtns(b, 2);

    size_t stride_b = nrow_b * ncol_b; // number of elements in one matrix of b
    size_t iv, im, ires = 0;

    fill(result->data, result->nelem, 0);

    // Correctly setting the initial view for a and b
    struct tensor_t* view_a = create_view(a, a->rank - 1, 0);
    struct tensor_t* view_b = create_view(b, b->rank - 2, 0);
    struct tensor_t* view_res = create_view(result, result->rank - 2, 0);

    size_t offset_a = 0;
    size_t offset_b = 0;

    for (iv = 0; iv < nvec_a; iv++) {
        offset_b = 0; // Reset offset_b for each new vector in a
        for (im = 0; im < nmat_b; im++) {
            // Adjusting offsets for views based on current matrix/vector
            view_a->offset = offset_a;
            view_b->offset = offset_b;
            view_res->offset = ires;

            // Perform matrix multiplication with adjusted views
            mul_mat(view_a, view_b, view_res, true);

            offset_b += stride_b; // Move to the next matrix in b
            ires += ncol_b; // Move to the next result slot
        }
        offset_a += ncol_a; // Move to the next vector in a
    }

    free_tensor(view_a);
    free_tensor(view_b);
    free_tensor(view_res);
}

#endif //CORE_MATMUL
