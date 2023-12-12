EERM		= ['ccall', 'cwrap', 'getValue', 'setValue']
OUT_DIR 	= ./src/core/build
SRC_DIR 	= ./src/core

# exported functions
EF = [ \
	'_scl_add', '_scl_sub', '_scl_mul', '_scl_div', \
	'_prw_add', '_prw_sub', '_prw_mul', '_prw_div', \
	'_rand_seed', '_rand_f', '_rand_i' \
]

clean: $(OUT_DIR)
	-rm -rf $(OUT_DIR)

main: $(SRC_DIR)/main.c $(SRC_DIR)/util.c $(SRC_DIR)/scalar.c $(SRC_DIR)/pairwise.c $(SRC_DIR)/rand.c
	@echo Building WASM executables from $(SRC_DIR)
	-mkdir -p $(OUT_DIR)
	-emcc -s "EXPORTED_RUNTIME_METHODS=$(EERM)" \
				-s "EXPORTED_FUNCTIONS=$(EF)" \
				-s WASM=1 \
				$(SRC_DIR)/main.c -o $(OUT_DIR)/index.js
