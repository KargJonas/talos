#ifndef CORE_MGMT
#define CORE_MGMT

struct mgmt_t {
    size_t allocated;  // amount of bytes allocated (only takes tensors into account)
    uint64_t ntensors; // number registered tensors (should take around 584942 years to overflow at 1Mio increments/s)
} mgmt;

void init_mgmt();
struct mgmt_t* get_mgmt_ptr();


#endif//CORE_MGMT
