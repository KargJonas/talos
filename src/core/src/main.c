#include "./util.c"
#include "./init.c"

// unary operations
#include "./unary_prw.c"
#include "./unary_brc.c"
#include "./unary_dbrc.c"

// binary operations
#include "./binary_brc.c"
#include "./binary_dbrc.c"
#include "./binary_mat.c"

// reduce operations
#include "./reduce.c"

// misc operations
#include "./dropout.c"

#include "./tensor.c"
#include "./mgmt.c"

int main() {
    init_mgmt();
    return 0;
}
