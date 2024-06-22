include .env

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
	_add_scl, _sub_scl, _mul_scl, _div_scl, _pow_scl, \
	_add_brc, _sub_brc, _mul_brc, _div_brc, _pow_brc, \
	\
	_add_scl_acc, _sub_scl_acc, _mul_scl_acc, _div_scl_acc, _pow_scl_acc, \
	_add_brc_acc, _sub_brc_acc, _mul_brc_acc, _div_brc_acc, _pow_brc_acc, \
	\
	_mul_tns, _dot_tns, \
	\
	_add_acc_dbrc, _sub_acc_dbrc, _mul_acc_dbrc, _div_acc_dbrc, _pow_acc_dbrc, \
	\
	_max_red_scl, _min_red_scl, _sum_red_scl, _mean_red_scl, \
	_max_red_tns, _min_red_tns, _sum_red_tns, _mean_red_tns, \
	\
	_negate_tns, _reciprocal_tns, \
	_sin_tns, _cos_tns, _tan_tns, \
	_asin_tns, _acos_tns, _atan_tns, \
	_sinh_tns, _cosh_tns, _tanh_tns, \
	_exp_tns, \
	_log_tns, _log10_tns, _log2_tns, \
	_invsqrt_tns, _sqrt_tns, \
	_ceil_tns, _floor_tns, _abs_tns, \
	\
	_relu_tns, _binstep_tns, _logistic_tns, \
	\
	_get_mgmt_ptr \
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
