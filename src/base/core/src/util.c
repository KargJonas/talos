#include "./util.h"
#include <math.h>

#define PI 3.1415926535

// allocation functions

float* alloc_farr(size_t size) {
    return (float*)malloc(size * sizeof(float));
}

size_t* alloc_starr(size_t size) {
    return (size_t*)malloc(size * sizeof(size_t));
}


// array copy functions

void copy_farr(float* source, float* dest, size_t nelem) {
    memcpy(dest, source, nelem * sizeof(float));
}

void copy_starr(size_t* source, size_t* dest, size_t nelem) {
    memcpy(dest, source, nelem * sizeof(size_t));
}


// deallocation functions

void free_farr(float* ptr) {
    free(ptr);
}

void free_starr(size_t* ptr) {
    free(ptr);
}


// misc utility functions

float fast_inv_sqrt(float number) {
    long i;
    float x2, y;
    const float threehalfs = 1.5F;
    x2 = number * 0.5F;
    y  = number;
    i  = *(long*)&y;                       // evil floating point bit level hacking
    i  = 0x5f3759df - (i >> 1);               // what the fuck?
    y  = *(float*)&i;
    y  = y * (threehalfs - (x2 * y * y));   // 1st iteration
    // y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed
    return y;
}

// get number of subtensors at a certain level
//   e.g.: n=1 will return the number of vectors in the tensor
//         n=2 will return the number of matrices in the tensor
size_t get_nsubtns(struct tensor_t* a, size_t n)
{
    size_t end = a->rank - n;
    size_t nsubtns = 1;

    for (size_t dim = 0; dim < end; dim++) {
        nsubtns *= a->shape[dim];
    }

    return nsubtns;
}

void set_row_major(struct tensor_t* a) {
    size_t stride = 1;

    for (size_t i = 0; i < a->rank; i++) {
        a->strides[i] = 1;
    }

    if (a->rank < 2) return;
    if (a->rank == 2) {
        a->strides[1] = 1;
        a->strides[0] = a->shape[1];
        return;
    }

    for (size_t i = a->rank - 1; i-- > 0;) {
        a->strides[i] = stride *= a->shape[i + 1];
    }
}

// get number of elements of subtensors of the specified axis
size_t get_nelem_of_axis_elements(struct tensor_t* a, size_t axis) {
    if (a->rank == 0) return 0;
    size_t nelem = 1;

    for (size_t dim = axis; dim < a->rank; dim++) {
        nelem *= a->shape[dim];
    }

    return nelem;
}

// get index of element in data array from linear index
size_t get_index(struct tensor_t* a, size_t linear_index) {
    size_t ia = a->offset;
    size_t remainder = linear_index;
    size_t iaxis;

    for (size_t dim = a->rank; dim-- > 0;) {
        iaxis = remainder % a->shape[dim];
        ia += iaxis * a->strides[dim];
        remainder /= a->shape[dim];
    }

    return ia;
}

float get_item(struct tensor_t* a, size_t linear_index) {
    size_t ia = a->offset;
    size_t remainder = linear_index;
    size_t iaxis;

    for (size_t dim = a->rank; dim-- > 0;) {
        iaxis = remainder % a->shape[dim];
        ia += iaxis * a->strides[dim];
        remainder /= a->shape[dim];
    }

    return a->data[ia];
}

// normal distribution based on box-muller transform
float normal(float mean, float std_dev, unsigned int* seed) {
    static int has_spare = 0;
    static double spare;

    if (has_spare) {
        has_spare = 0;
        return mean + std_dev * spare;
    }

    has_spare = 1;
    double u, v, s;
    do {
        u = (rand_r(seed) / ((double) RAND_MAX)) * 2.0 - 1.0;
        v = (rand_r(seed) / ((double) RAND_MAX)) * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0);

    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return mean + std_dev * u * s;
}

// // higher precision version of the above code
// float normal(float mean, float sigma) {
//     static int has_spare = 0;
//     static double spare;

//     if (has_spare) {
//         has_spare = 0;
//         return mean + sigma * spare;
//     }

//     has_spare = 1;
//     double u, v, s;
//     do {
//         u = drand48() * 2.0 - 1.0;
//         v = drand48() * 2.0 - 1.0;
//         s = u * u + v * v;
//     } while (s >= 1.0 || s == 0);

//     s = sqrt(-2.0 * log(s) / s);
//     spare = v * s;
//     return mean + sigma * u * s;
// }

// the problem at hand:
//   given two equally shaped, but potentially differently strided tensors A, B,
//   the strides S_A, S_B of said tensors and a coordinate a = [i,j,k, ...], that
//   locates a component of tensor A;
//   how can we compute a coordinate b = [v, w, x, ...] that locates the element
//   in tensor B that is "in the same location" as the element identified by a?
//   with in the same location, we mean this:
//   if A and B were of rank 2, then we could place the two matrices on top of one
//   another and the components identified by a and b respectively would lie directly
//   on top of one another.
// in other words, we want to perform a kind of "basis transformation" from the
// coordinate system of A to the coordinate system of B
// solution:
// - find coordinate of the component (that the scalar tensor a is referencing)
//   within the parent tensor of a
// - find linear index of scalar a within it's parent tensor from the coordinates
//   (as if the parent were contiguous)
// - dissect linear index into B coords
// - find actual index of b within its parent from B coords

// void sync_scl_views(struct tensor_t* a, struct tensor_t* b) {
//     // Assertion 1: a->isview && b->isview
//     // Assertion 2: a->viewsrc->rank == b->viewsrc->rank

//     // step 1: find linear index of `a` within it's parent tensor
    
//     size_t ia = a->offset;
//     size_t remainder = linear_index;
//     size_t iaxis;

//     for (size_t dim = a->rank; dim-- > 0;) {
//         iaxis = remainder % a->shape[dim];
//         ia += iaxis * a->strides[dim];
//         remainder /= a->shape[dim];
//     }

//     return ia;
// }
