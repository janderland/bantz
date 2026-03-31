INPUT ?= input.md   # Chat log to parse into training data.

BASE_MODEL ?= mlx-community/Llama-3.2-3B-Instruct  # HuggingFace model to fine-tune. This must
                                                   # be a HuggingFace model ID in the format
                                                   # "org/model-name". MLX-compatible models can
                                                   # be found at huggingface.co/mlx-community.

PROMPT_MODEL ?= llama3.1:8b                        # Ollama model used for prompt generation. This
                                                   # must be an Ollama model name in the format
                                                   # "name:tag". Available models can be found
                                                   # at ollama.com/library.

WINDOW  ?= 6          # Number of preceding messages included as context for
                      # each training example. Larger values give the model
                      # more conversational context to learn from, but produce
                      # longer examples which use more memory and slow training.

VALID_SPLIT ?= 10     # Percentage of corpus examples held out for validation
                      # (0-100). The validation set is used during training to
                      # detect overfitting. Must be greater than 0.

ITERS   ?= 1000       # Number of training steps. More iterations means longer
                      # training and potentially better results, but with
                      # diminishing returns and risk of overfitting (where the
                      # model memorizes the training data instead of learning
                      # general patterns).

BATCH   ?= 4          # Number of training examples processed per step. Higher
                      # values use more memory but produce more stable gradient
                      # updates. Gradient checkpointing (enabled below) reduces
                      # memory enough to use larger batches on Apple Silicon.

MAX_SEQ ?= 650        # Maximum number of tokens per training example. Examples
                      # longer than this are truncated. Lower values reduce
                      # memory usage but may cut off context that the model
                      # needs to learn from.

WIDTH   ?= 60         # Word-wrap width for chat output. Set to your terminal
                      # width for best results, or 0 to disable wrapping.

MAX_PHRASES ?= 100    # Max number of frequent multi-word phrases to consider
                      # when generating the prompt.

MAX_TOPICS  ?= 50     # Max number of high-anomaly words to consider when
                      # generating the prompt.

.PHONY: help all deps corpus train fuse gguf run prompt clean FORCE

help:
	@printf 'Targets:\n'
	@printf '  all      Build everything and register the model with Ollama\n'
	@printf '  deps     Check system dependencies and install missing ones via brew\n'
	@printf '  corpus   Parse the chat log into training data (JSONL)\n'
	@printf '  prompt   Analyze the chat log for personality notes and group references\n'
	@printf '  train    Fine-tune the base model on the training corpus\n'
	@printf '  fuse     Merge the LoRA adapters into the base model weights\n'
	@printf '  gguf     Convert the fused model to GGUF format\n'
	@printf '  run      Chat with the trained model\n'
	@printf '  clean    Remove all build artifacts\n'

deps:
	@missing=""; \
	command -v ollama     >/dev/null 2>&1 || missing="$$missing ollama"; \
	command -v python3.10 >/dev/null 2>&1 || missing="$$missing python@3.10"; \
	command -v jq         >/dev/null 2>&1 || missing="$$missing jq"; \
	if [ -z "$$missing" ]; then \
		printf 'All dependencies are installed.\n'; \
	else \
		printf 'Missing:%s\n' "$$missing"; \
		printf 'Install with brew? [y/N] '; \
		read ans </dev/tty; \
		case "$$ans" in \
			[yY]*) brew install $$missing ;; \
			*) printf 'Skipped.\n' ;; \
		esac; \
	fi

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
	. .venv/bin/activate && python3 scripts/corpus.py $(INPUT) build/data/train.jsonl --window $(WINDOW) --valid-split $(VALID_SPLIT)

build/.train: build/data/train.jsonl build/.train-params
	mkdir -p build/adapters
	@date +%s > build/.train-start
	. .venv/bin/activate && mlx_lm.lora \
	  --model $(BASE_MODEL) \
	  --data build/data \
	  --adapter-path build/adapters \
	  --train \
	  --iters $(ITERS) \
	  --batch-size $(BATCH) \
	  --max-seq-length $(MAX_SEQ) \
	  --grad-checkpoint \
	  --lr-schedule cosine_decay \
	  --warmup-steps 100
	@elapsed=$$(($$(date +%s) - $$(cat build/.train-start))); \
	printf '%s  train    %dm %ds\n' "$$(date '+%Y-%m-%d %H:%M:%S')" $$((elapsed / 60)) $$((elapsed % 60)) >> build/timings.log
	touch build/.train

build/.fuse: build/.train
	mkdir -p build/model
	. .venv/bin/activate && mlx_lm.fuse \
	  --model $(BASE_MODEL) \
	  --adapter-path build/adapters \
	  --save-path build/model
	jq '.tokenizer_class = "PreTrainedTokenizerFast"' build/model/tokenizer_config.json \
	  > build/model/tokenizer_config.tmp.json
	mv build/model/tokenizer_config.tmp.json build/model/tokenizer_config.json
	touch build/.fuse

build/.gguf: build/.fuse build/.venv-llama
	mkdir -p build/gguf
	. llama.cpp/.venv/bin/activate && python llama.cpp/convert_hf_to_gguf.py \
	  build/model --outfile build/gguf/bantz-model.gguf
	touch build/.gguf

build/Modelfile: Modelfile.tmpl build/analysis.md
	@mkdir -p build
	awk 'FNR==NR{content=content $$0 "\n"; next} /\{\{ANALYSIS\}\}/{printf "%s", content; next} 1' \
		build/analysis.md Modelfile.tmpl | \
	sed 's|FROM build/|FROM $(CURDIR)/build/|' > build/Modelfile

build/analysis.md: build/.venv $(INPUT)
	@mkdir -p build
	ollama pull $(PROMPT_MODEL)
	@date +%s > build/.prompt-start
	. .venv/bin/activate && python3 scripts/prompt.py $(INPUT) --model $(PROMPT_MODEL) --verbose \
		--max-phrases $(MAX_PHRASES) --max-topics $(MAX_TOPICS) --output build/analysis.md
	@elapsed=$$(($$(date +%s) - $$(cat build/.prompt-start))); \
	printf '%s  prompt   %dm %ds\n' "$$(date '+%Y-%m-%d %H:%M:%S')" $$((elapsed / 60)) $$((elapsed % 60)) >> build/timings.log

build/.ollama: build/.gguf build/Modelfile
	ollama create bantz -f build/Modelfile
	touch build/.ollama

run: build/.ollama build/.venv
	. .venv/bin/activate && python3 scripts/chat.py --width $(WIDTH) "$(PROMPT)"

prompt: build/analysis.md

clean:
	rm -rf build .venv llama.cpp/.venv

# Parameter stamp files: each file stores the current values of the parameters
# for its step. FORCE ensures the recipe always runs to check the values, but
# the file is only written (and thus made newer than its dependents) when the
# values have actually changed. This triggers a rebuild of the affected step.

build/.corpus-params: FORCE
	@mkdir -p build
	@printf 'WINDOW=%s\nVALID_SPLIT=%s\n' '$(WINDOW)' '$(VALID_SPLIT)' | cmp -s - $@ \
		|| printf 'WINDOW=%s\nVALID_SPLIT=%s\n' '$(WINDOW)' '$(VALID_SPLIT)' > $@

build/.train-params: FORCE
	@mkdir -p build
	@printf 'BASE_MODEL=%s\nITERS=%s\nBATCH=%s\nMAX_SEQ=%s\n' '$(BASE_MODEL)' '$(ITERS)' '$(BATCH)' '$(MAX_SEQ)' | cmp -s - $@ \
		|| printf 'BASE_MODEL=%s\nITERS=%s\nBATCH=%s\nMAX_SEQ=%s\n' '$(BASE_MODEL)' '$(ITERS)' '$(BATCH)' '$(MAX_SEQ)' > $@
