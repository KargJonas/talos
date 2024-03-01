#include "./mgmt.h"

struct mgmt_t* get_mgmt_ptr() {
    return &mgmt;
}

void init_mgmt() {
    mgmt.allocated = 0;
    mgmt.ntensors = 0;
}
