EERM		= ['ccall', 'cwrap', 'getValue', 'setValue']
OUT_DIR 	= ./src/core/build
SRC_DIR 	= ./src/core

# exported functions
EF = [ \
	'_alloc_farr', '_alloc_starr', '_free',  \
	'_rand_seed', '_rand_f', '_rand_i', \
	'_add_scl', '_sub_scl', '_mul_scl', '_div_scl', \
	'_prw_op_broadcast', \
	'_add_prw', '_sub_prw', '_mul_prw', '_div_prw', \
	'_mul_tns', '_dot_tns', \
	'_act_relu', '_act_tanh' \
]

##  WARNING  ##
# experimenting with -msimd128 flag
# can cause compat issues

clean: $(OUT_DIR)
	-rm -rf $(OUT_DIR)

main: $(SRC_DIR)/main.c $(SRC_DIR)/util.c $(SRC_DIR)/scalar.c $(SRC_DIR)/pairwise.c $(SRC_DIR)/rand.c $(SRC_DIR)/matmul.c
	@echo Building WASM executables from $(SRC_DIR)
	-mkdir -p $(OUT_DIR)
	-emcc -s "EXPORTED_RUNTIME_METHODS=$(EERM)" \
				-s "EXPORTED_FUNCTIONS=$(EF)" \
				-s WASM=1 \
				-s ALLOW_MEMORY_GROWTH=1 \
				-msimd128 \
				-O3 $(SRC_DIR)/main.c -o $(OUT_DIR)/index.js
