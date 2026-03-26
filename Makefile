INPUT   ?= input.md   # Chat log to parse into training data.

WINDOW  ?= 6          # Number of preceding messages included as context for
                      # each training example. Larger values give the model
                      # more conversational context to learn from, but produce
                      # longer examples which use more memory and slow training.

ITERS   ?= 1000       # Number of training steps. More iterations means longer
                      # training and potentially better results, but with
                      # diminishing returns and risk of overfitting (where the
                      # model memorizes the training data instead of learning
                      # general patterns).

BATCH   ?= 1          # Number of training examples processed per step. Higher
                      # values use more memory but produce more stable gradient
                      # updates. On Apple Silicon, 1 is typical due to memory
                      # constraints.

MAX_SEQ ?= 650        # Maximum number of tokens per training example. Examples
                      # longer than this are truncated. Lower values reduce
                      # memory usage but may cut off context that the model
                      # needs to learn from.

.PHONY: all run clean FORCE

all: build/.ollama
parse: build/data/train.jsonl
train: build/.train
fuse: build/.fuse
gguf: build/.gguf

build/.venv: requirements.txt
	mkdir -p build
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	touch build/.venv

build/.submodules:
	mkdir -p build
	git submodule update --init
	touch build/.submodules

build/.venv-llama: build/.submodules llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
	mkdir -p build
	python3.10 -m venv llama.cpp/.venv
	llama.cpp/.venv/bin/pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
	touch build/.venv-llama

build/data/train.jsonl: build/.venv build/.parse-params $(INPUT) scripts/parse.py
	mkdir -p build/data
	. .venv/bin/activate && python3 scripts/parse.py $(INPUT) build/data/train.jsonl --window $(WINDOW)

build/.train: build/data/train.jsonl build/.train-params scripts/train.sh
	mkdir -p build/adapters
	bash scripts/train.sh $(ITERS) $(BATCH) $(MAX_SEQ)
	touch build/.train

build/.fuse: build/.train scripts/fuse.sh
	mkdir -p build/model
	bash scripts/fuse.sh
	touch build/.fuse

build/.gguf: build/.fuse build/.venv-llama scripts/gguf.sh
	mkdir -p build/gguf
	bash scripts/gguf.sh
	touch build/.gguf

build/.ollama: build/.gguf Modelfile
	ollama create bantz -f Modelfile
	touch build/.ollama

run: build/.ollama build/.venv
	. .venv/bin/activate && python3 scripts/chat.py $(PROMPT)

clean:
	rm -rf build .venv llama.cpp/.venv

# Parameter stamp files: each file stores the current values of the parameters
# for its step. FORCE ensures the recipe always runs to check the values, but
# the file is only written (and thus made newer than its dependents) when the
# values have actually changed. This triggers a rebuild of the affected step.

build/.parse-params: FORCE
	@mkdir -p build
	@printf 'WINDOW=%s\n' '$(WINDOW)' | cmp -s - $@ \
		|| printf 'WINDOW=%s\n' '$(WINDOW)' > $@

build/.train-params: FORCE
	@mkdir -p build
	@printf 'ITERS=%s\nBATCH=%s\nMAX_SEQ=%s\n' '$(ITERS)' '$(BATCH)' '$(MAX_SEQ)' | cmp -s - $@ \
		|| printf 'ITERS=%s\nBATCH=%s\nMAX_SEQ=%s\n' '$(ITERS)' '$(BATCH)' '$(MAX_SEQ)' > $@
