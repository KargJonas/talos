include .env
include $(CORE_OP_NAME_DEFINITIONS)

EERM		= [ccall, cwrap, getValue, setValue]

# exported functions
EF = [ \
	_alloc_farr, _alloc_starr, _create_tensor, \
	_free_farr, _free_starr, _free_tensor, \
	_copy_farr, _copy_starr, \
	_clone_tensor, _create_view, \
	\
	_rand_seed, _rand_f, _rand_i, _fill, \
	\
	_get_mgmt_ptr, \
	\
	_mul_tns, _dot_tns,\
	_max_red_scl, _min_red_scl, _sum_red_scl, _mean_red_scl, \
	_max_red_tns, _min_red_tns, _sum_red_tns, _mean_red_tns, \
	\
	$(EXPORTED_OPS) \
]

##  WARNING  ##
# experimenting with -msimd128 flag
# can cause compat issues

## Initial memory flag ##
# -s INITIAL_MEMORY=256MB

clean: $(CORE_OUT_DIR)
	-rm -rf $(CORE_OUT_DIR)

main: $(CORE_SRC_DIR)/main.c
	@echo Building WASM executables from $(CORE_SRC_DIR)
	-mkdir -p $(CORE_OUT_DIR)
	-emcc -s "EXPORTED_RUNTIME_METHODS=$(EERM)" \
				-s "EXPORTED_FUNCTIONS=$(EF)" \
				-s WASM=1 \
				-s ALLOW_MEMORY_GROWTH=1 \
				-s SINGLE_FILE=1 \
				-O3 $(CORE_PREPROC_OUT_DIR)/main.c -o $(CORE_OUT_DIR)/index.js
