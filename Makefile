include .env
include $(CORE_OP_NAME_DEFINITIONS)

# NOTE: see .env for preprocessor/compiler/bundler configuration

EERM		= [ccall, cwrap, getValue, setValue]

# exported functions
EF = [ \
	_create_tensor, _free_tensor, \
	_clone_tensor, _create_view, _create_reshape_view, _shift_view, \
	\
	_init_uniform, _init_normal, _init_fill, \
	\
	_matmul, _matmul_acc, _dot, _dot_acc, \
	_max_red_idx, _min_red_idx, \
	_max_red_scl, _min_red_scl, _sum_red_scl, _mean_red_scl, \
	_sum_red_tns, _mean_red_tns, \
	\
	_get_mgmt_ptr, \
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
