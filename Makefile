INPUT ?= input.md   # Chat log to parse into training data.

BASE_MODEL ?= mlx-community/Llama-3.2-3B-Instruct  # HuggingFace model to fine-tune. This must
                                                   # be a HuggingFace model ID in the format
                                                   # "org/model-name". MLX-compatible models can
                                                   # be found at huggingface.co/mlx-community.

ANALYSIS_MODEL ?= llama3.1:8b                      # Ollama model used for chat analysis. This
                                                   # must be an Ollama model name in the format
                                                   # "name:tag". Available models can be found
                                                   # at ollama.com/library.

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

WIDTH   ?= 60         # Word-wrap width for chat output. Set to your terminal
                      # width for best results, or 0 to disable wrapping.

MAX_PHRASES ?= 100    # Max number of frequent multi-word phrases to consider
                      # when analyzing group references.

MAX_TOPICS  ?= 50     # Max number of high-anomaly words to consider when
                      # analyzing group references.

.PHONY: all run analyze clean FORCE

all: build/.ollama
corpus: build/data/train.jsonl
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

build/data/train.jsonl: build/.venv build/.corpus-params $(INPUT) scripts/corpus.py
	mkdir -p build/data
	. .venv/bin/activate && python3 scripts/corpus.py $(INPUT) build/data/train.jsonl --window $(WINDOW)

build/.train: build/data/train.jsonl build/.train-params scripts/train.sh
	mkdir -p build/adapters
	start=$$(date +%s); \
	bash scripts/train.sh $(BASE_MODEL) $(ITERS) $(BATCH) $(MAX_SEQ) && \
	end=$$(date +%s) && \
	elapsed=$$((end - start)) && \
	printf '%s  train    %dm %ds\n' "$$(date '+%Y-%m-%d %H:%M:%S')" $$((elapsed / 60)) $$((elapsed % 60)) >> build/timings.log
	touch build/.train

build/.fuse: build/.train scripts/fuse.sh
	mkdir -p build/model
	bash scripts/fuse.sh $(BASE_MODEL)
	touch build/.fuse

build/.gguf: build/.fuse build/.venv-llama scripts/gguf.sh
	mkdir -p build/gguf
	bash scripts/gguf.sh
	touch build/.gguf

build/Modelfile: Modelfile.tmpl build/analysis.md
	@mkdir -p build
	awk 'FNR==NR{content=content $$0 "\n"; next} /\{\{ANALYSIS\}\}/{printf "%s", content; next} 1' \
		build/analysis.md Modelfile.tmpl > build/Modelfile

build/analysis.md: build/.venv $(INPUT)
	@mkdir -p build
	ollama pull $(ANALYSIS_MODEL)
	start=$$(date +%s); \
	. .venv/bin/activate && python3 scripts/analyze.py $(INPUT) --model $(ANALYSIS_MODEL) --verbose \
		--max-phrases $(MAX_PHRASES) --max-topics $(MAX_TOPICS) --output build/analysis.md && \
	end=$$(date +%s) && \
	elapsed=$$((end - start)) && \
	printf '%s  analyze  %dm %ds\n' "$$(date '+%Y-%m-%d %H:%M:%S')" $$((elapsed / 60)) $$((elapsed % 60)) >> build/timings.log

build/.ollama: build/.gguf build/Modelfile
	ollama create bantz -f build/Modelfile
	touch build/.ollama

run: build/.ollama build/.venv
	. .venv/bin/activate && python3 scripts/chat.py --width $(WIDTH) $(PROMPT)

analyze: build/analysis.md

clean:
	rm -rf build .venv llama.cpp/.venv

# Parameter stamp files: each file stores the current values of the parameters
# for its step. FORCE ensures the recipe always runs to check the values, but
# the file is only written (and thus made newer than its dependents) when the
# values have actually changed. This triggers a rebuild of the affected step.

build/.corpus-params: FORCE
	@mkdir -p build
	@printf 'WINDOW=%s\n' '$(WINDOW)' | cmp -s - $@ \
		|| printf 'WINDOW=%s\n' '$(WINDOW)' > $@

build/.train-params: FORCE
	@mkdir -p build
	@printf 'BASE_MODEL=%s\nITERS=%s\nBATCH=%s\nMAX_SEQ=%s\n' '$(BASE_MODEL)' '$(ITERS)' '$(BATCH)' '$(MAX_SEQ)' | cmp -s - $@ \
		|| printf 'BASE_MODEL=%s\nITERS=%s\nBATCH=%s\nMAX_SEQ=%s\n' '$(BASE_MODEL)' '$(ITERS)' '$(BATCH)' '$(MAX_SEQ)' > $@
