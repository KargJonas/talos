EERM		= [ccall, cwrap, getValue, setValue]
OUT_DIR 	= ./src/core/build
SRC_DIR 	= ./src/core

# exported functions
EF = [ \
	_alloc_farr, _alloc_starr, _create_tensor, \
	_free_farr, _free_starr, _free_tensor, \
	_copy_farr, _copy_starr, _copy_tensor, _copy_tensor_metadata, \
	\
	_rand_seed, _rand_f, _rand_i, _fill, \
	\
	_add_scl, _sub_scl, _mul_scl, _div_scl, _pow_scl, \
	_add_prw, _sub_prw, _mul_prw, _div_prw, \
	_add_prw_brc, _sub_prw_brc, _mul_prw_brc, _div_prw_brc, \
	_mul_tns, _dot_tns, \
	\
	_max_red, _min_red, _sum_red, \
	\
	_negate_tns, _reciprocal_tns, \
	_sin_tns, _cos_tns, _tan_tns, \
	_asin_tns, _acos_tns, _atan_tns, \
	_sinh_tns, _cosh_tns, _tanh_tns, \
	_exp_tns, \
	_log_tns, _log10_tns, _log2_tns, \
	_invsqrt_tns, _sqrt_tns, \
	_ceil_tns, _floor_tns, _abs_tns, \
	_identity_tns, \
	\
	_relu_tns, _binstep_tns, _logistic_tns \
]

##  WARNING  ##
# experimenting with -msimd128 flag
# can cause compat issues

## Initial memory flag ##
# -s INITIAL_MEMORY=256MB

clean: $(OUT_DIR)
	-rm -rf $(OUT_DIR)

main: $(SRC_DIR)/main.c
	@echo Building WASM executables from $(SRC_DIR)
	-mkdir -p $(OUT_DIR)
	-emcc -s "EXPORTED_RUNTIME_METHODS=$(EERM)" \
				-s "EXPORTED_FUNCTIONS=$(EF)" \
				-s WASM=1 \
				-s ALLOW_MEMORY_GROWTH=1 \
				-s SINGLE_FILE=1\
				-O3 $(SRC_DIR)/main.c -o $(OUT_DIR)/index.js
