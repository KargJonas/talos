#ifndef CORE_MATMUL
#define CORE_MATMUL

#include <stddef.h>
#include "./util.h"
#include "./tensor.h"

#include <emscripten.h>

// todo:
//         YOU LEFT OFF HERE
// 
//   here i need to work on different parts of the supertensor individually
//   maybe i should introduce a get_view(offset) method
//
//   this is basically the same issue that get_axis_iterable() solves
//   in the frontend...

// performs standard matrix multiplication on a matrix
// that is located somewhere inside a larger tensor
// void mul_mat(
//     float* a, size_t nrow_a, size_t ncol_a,
//     float* b, size_t nrow_b, size_t ncol_b,
//     float* result
// ) {
//     register size_t r, c, i, ires, offset_row, ib;

//     // todo optimization potential: replace multiplications by looped increments

//     for (r = 0; r < nrow_a; r++) {
//         for (c = 0; c < ncol_b; c++) {
//             ires       = r * ncol_b + c;
//             offset_row = r * ncol_a;

//             for (i = 0; i < ncol_a; i++) {
//                 ib = i * ncol_b + c;
//                 result[ires] += a[offset_row + i] * b[ib];
//             }
//         }
//     }
// }

// todo make a macro for all of these dumb functions

size_t get_ncols(struct tensor_t* a) {
    return a->shape[a->rank - 1];
}

size_t get_nrows(struct tensor_t* a) {
    if (a->rank < 2) return 1;
    return a->shape[a->rank - 2];
}

size_t get_row_stride(struct tensor_t* a) {
    // todo: do we need a special case for vectors
    return a->strides[a->rank - 2];
}

#define get_strides_bwd(a, i) a->strides[a->rank - i - 1]

// we are tearing open a healed wound here
void mul_mat(struct tensor_t* a, struct tensor_t* b, struct tensor_t* res, bool dot) {
    size_t nrow_a = dot ? 1 : get_nrows(a);
    size_t ncol_a = get_ncols(a);
    size_t nrow_b = get_nrows(b);
    size_t ncol_b = get_ncols(b);

    register size_t r, c, i, ires, offset_row, ia, ib;

    for (r = 0; r < nrow_a; r++) {
        for (c = 0; c < ncol_b; c++) {
            ires =  res->offset +
                    r * get_strides_bwd(res, 1) +
                    c * get_strides_bwd(res, 0);


            EM_ASM({
                console.log('row: ', $0, " col: ", $1, " ires: ", $2, " ia: ", $3, " ib: ", $3);
            }, r, c, ires, ia, ib);

            // this loop could be easily parallelized because
            // of the inherent associativity of the summation
            for (i = 0; i < ncol_a; i++) {

                ia = a->offset +
                    r * get_strides_bwd(a, 1) +
                    i * get_strides_bwd(a, 0);

                ib = b->offset +
                    i * get_strides_bwd(b, 1) +
                    c * get_strides_bwd(b, 0);

                res->data[ires] += a->data[ia] * a->data[ib];
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

    size_t stride_res = nrow_a * ncol_b; // todo: think i could replace this by using result->strides
    size_t stride_a = nmat_a > 1 ? nrow_a * ncol_a : 0;
    size_t stride_b = nmat_b > 1 ? nrow_b * ncol_b : 0;
    size_t nmat_max = MAX(nmat_a, nmat_b);
    register size_t ia = 0, ib = 0, ires = 0;

    fill(result->data, result->nelem, 0);

    struct tensor_t* view_a   = create_view(a, a->rank - 2, 0);
    struct tensor_t* view_b   = create_view(b, b->rank - 2, 0);
    struct tensor_t* view_res = create_view(result, result->rank - 2, 0);

    for (size_t i = 0; i < nmat_max; i++) {
        // mul_mat(
        //     a, ia, nrow_a, ncol_a,
        //     b, ib, nrow_b, ncol_b,
        //     result, ires);

        mul_mat(view_a, view_b, view_res, false);

        view_a->offset += stride_a;
        view_b->offset += stride_b;
        view_res->offset += stride_res;

        // ia += stride_a; // todo: minimal optimization potential (if stride = 0, add useless)
        // ib += stride_b;
        // ires += stride_res;
    }

    free_tensor(view_a);
    free_tensor(view_b);
    free_tensor(view_res);

    // todo: dispose of view !!!!!!!!
}

void dot_tns(struct tensor_t* a, struct tensor_t* b, struct tensor_t* result) {
    size_t ncol_a = get_ncols(a);
    size_t nvec_a = get_nsubtns(a, 1);

    size_t nrow_b = get_nrows(b);
    size_t ncol_b = get_ncols(b);
    size_t nmat_b = get_nsubtns(b, 2);

    size_t stride_b = nrow_b * ncol_b; // number of elements in one matrix of b
    size_t iv, im, ia = 0, ib = 0, ires = 0;

    fill(result->data, result->nelem, 0);

    for (iv = 0; iv < nvec_a; iv++) {
        for (im = 0; im < nmat_b; im++) {
            // mul_mat(
            //     a->data + ia, 1, ncol_a,
            //     b->data + ib, nrow_b, ncol_b,
            //     result->data + ires);

            mul_mat(a, b, result, true);

            ib += stride_b; // step over to next matrix
            ires += ncol_b;
        }

        ia += ncol_a; // step over to next vector
        ib = 0;
    }
}

#endif //CORE_MATMUL