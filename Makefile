EF		= ['_alloc_farr', '_add']
EERM	= ['ccall', 'cwrap', 'getValue', 'setValue']

OUT_DIR 		= ./src/wasm/build
SRC_DIR 		= ./src/wasm

clean: $(OUT_DIR)
	-rm -rf $(OUT_DIR)

main: $(SRC_DIR)/add.c
	@echo Building WASM executables from $(SRC_DIR)
	-mkdir -p $(OUT_DIR)
	-emcc -s "EXPORTED_RUNTIME_METHODS=$(EERM)" \
				-s "EXPORTED_FUNCTIONS=$(EF)" \
				-s WASM=1 \
				$(SRC_DIR)/add.c -o $(OUT_DIR)/compiled.js
