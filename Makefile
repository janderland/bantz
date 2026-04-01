# Name of the bot version to build. Each version gets its own build directory
# and Ollama model name (bantz-VERSION). Version-specific parameters are loaded
# from versions/$(VERSION).mk if it exists.
VERSION ?= default

-include versions/$(VERSION).mk

# Scripts used for each pipeline stage. Override in versions/$(VERSION).mk to
# use version-specific scripts.
CORPUS_SCRIPT ?= scripts/corpus.py
CHAT_SCRIPT   ?= scripts/chat.py
PROMPT_SCRIPT ?= scripts/prompt.py

INPUT ?= input.md   # Chat log to parse into training data.

BASE_MODEL ?= mlx-community/Llama-3.2-3B-Instruct  # HuggingFace model to fine-tune. This must
                                                   # be a HuggingFace model ID in the format
                                                   # "org/model-name". MLX-compatible models can
                                                   # be found at huggingface.co/mlx-community.

PROMPT_MODEL ?= llama3.1:8b                        # Ollama model used for prompt generation. This
                                                   # must be an Ollama model name in the format
                                                   # "name:tag". Available models can be found
                                                   # at ollama.com/library.

WINDOW ?= 6           # Number of preceding messages included as context for
                      # each training example. Larger values give the model
                      # more conversational context to learn from, but produce
                      # longer examples which use more memory and slow training.

VALID_SPLIT ?= 10     # Percentage of corpus examples held out for validation
                      # (0-100). The validation set is used during training to
                      # detect overfitting. Must be greater than 0.

ITERS ?= 1000         # Number of training steps. More iterations means longer
                      # training and potentially better results, but with
                      # diminishing returns and risk of overfitting (where the
                      # model memorizes the training data instead of learning
                      # general patterns).

BATCH ?= 4            # Number of training examples processed per step. Higher
                      # values use more memory but produce more stable gradient
                      # updates. Gradient checkpointing (enabled below) reduces
                      # memory enough to use larger batches on Apple Silicon.

MAX_SEQ ?= 1024       # Maximum number of tokens per training example. Examples
                      # longer than this are truncated. Lower values reduce
                      # memory usage but may cut off context that the model
                      # needs to learn from.

WIDTH   ?= 60         # Word-wrap width for chat output. Set to your terminal
                      # width for best results, or 0 to disable wrapping.

MAX_PHRASES ?= 100    # Max number of frequent multi-word phrases to consider
                      # when generating the prompt.

MAX_TOPICS ?= 50      # Max number of high-anomaly words to consider when
                      # generating the prompt.

BUILD_DIR = build/$(strip $(VERSION))

.PHONY: help all deps corpus train fuse gguf run prompt test versions clean clean-all FORCE

help:
	@printf 'Targets:\n'
	@printf '  all       Build everything and register the model with Ollama\n'
	@printf '  deps      Check system dependencies and install missing ones via brew\n'
	@printf '  corpus    Parse the chat log into training data (JSONL)\n'
	@printf '  prompt    Analyze the chat log for personality notes and group references\n'
	@printf '  train     Fine-tune the base model on the training corpus\n'
	@printf '  fuse      Merge the LoRA adapters into the base model weights\n'
	@printf '  gguf      Convert the fused model to GGUF format\n'
	@printf '  run       Chat with the trained model\n'
	@printf '  test      Run the test suite against the trained model\n'
	@printf '  versions  List available versions\n'
	@printf '  clean     Remove build artifacts for the current version\n'
	@printf '  clean-all Remove all build artifacts\n'
	@printf '\nParameters:\n'
	@printf '  VERSION=name  Version to build (default: default)\n'

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

all: $(BUILD_DIR)/.ollama
corpus: $(BUILD_DIR)/data/train.jsonl
train: $(BUILD_DIR)/.train
fuse: $(BUILD_DIR)/.fuse
gguf: $(BUILD_DIR)/.gguf

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

$(BUILD_DIR)/data/train.jsonl: build/.venv $(BUILD_DIR)/.corpus-params $(INPUT) $(CORPUS_SCRIPT)
	mkdir -p $(BUILD_DIR)/data
	. .venv/bin/activate && python3 $(CORPUS_SCRIPT) $(INPUT) $(BUILD_DIR)/data/train.jsonl --window $(WINDOW) --valid-split $(VALID_SPLIT)

$(BUILD_DIR)/.train: $(BUILD_DIR)/data/train.jsonl $(BUILD_DIR)/.train-params
	mkdir -p $(BUILD_DIR)/adapters
	@date +%s > $(BUILD_DIR)/.train-start
	. .venv/bin/activate && mlx_lm.lora \
	  --model $(BASE_MODEL) \
	  --data $(BUILD_DIR)/data \
	  --adapter-path $(BUILD_DIR)/adapters \
	  --train \
	  --iters $(ITERS) \
	  --batch-size $(BATCH) \
	  --max-seq-length $(MAX_SEQ) \
	  --grad-checkpoint
	@elapsed=$$(($$(date +%s) - $$(cat $(BUILD_DIR)/.train-start))); \
	printf '%s  train    %dm %ds\n' "$$(date '+%Y-%m-%d %H:%M:%S')" $$((elapsed / 60)) $$((elapsed % 60)) >> $(BUILD_DIR)/timings.log
	touch $(BUILD_DIR)/.train

$(BUILD_DIR)/.fuse: $(BUILD_DIR)/.train
	mkdir -p $(BUILD_DIR)/model
	. .venv/bin/activate && mlx_lm.fuse \
	  --model $(BASE_MODEL) \
	  --adapter-path $(BUILD_DIR)/adapters \
	  --save-path $(BUILD_DIR)/model
	jq '.tokenizer_class = "PreTrainedTokenizerFast"' $(BUILD_DIR)/model/tokenizer_config.json \
	  > $(BUILD_DIR)/model/tokenizer_config.tmp.json
	mv $(BUILD_DIR)/model/tokenizer_config.tmp.json $(BUILD_DIR)/model/tokenizer_config.json
	touch $(BUILD_DIR)/.fuse

$(BUILD_DIR)/.gguf: $(BUILD_DIR)/.fuse build/.venv-llama
	mkdir -p $(BUILD_DIR)/gguf
	. llama.cpp/.venv/bin/activate && python llama.cpp/convert_hf_to_gguf.py \
	  $(BUILD_DIR)/model --outfile $(BUILD_DIR)/gguf/bantz-model.gguf
	touch $(BUILD_DIR)/.gguf

$(BUILD_DIR)/Modelfile: Modelfile.tmpl $(BUILD_DIR)/analysis.md
	@mkdir -p $(BUILD_DIR)
	awk 'FNR==NR{content=content $$0 "\n"; next} /\{\{ANALYSIS\}\}/{printf "%s", content; next} 1' \
		$(BUILD_DIR)/analysis.md Modelfile.tmpl | \
	sed 's|FROM build/|FROM $(CURDIR)/$(BUILD_DIR)/|' > $(BUILD_DIR)/Modelfile

$(BUILD_DIR)/analysis.md: build/.venv $(INPUT)
	@mkdir -p $(BUILD_DIR)
	ollama pull $(PROMPT_MODEL)
	@date +%s > $(BUILD_DIR)/.prompt-start
	. .venv/bin/activate && python3 $(PROMPT_SCRIPT) $(INPUT) --model $(PROMPT_MODEL) --verbose \
		--max-phrases $(MAX_PHRASES) --max-topics $(MAX_TOPICS) --output $(BUILD_DIR)/analysis.md
	@elapsed=$$(($$(date +%s) - $$(cat $(BUILD_DIR)/.prompt-start))); \
	printf '%s  prompt   %dm %ds\n' "$$(date '+%Y-%m-%d %H:%M:%S')" $$((elapsed / 60)) $$((elapsed % 60)) >> $(BUILD_DIR)/timings.log

$(BUILD_DIR)/.ollama: $(BUILD_DIR)/.gguf $(BUILD_DIR)/Modelfile
	ollama create bantz-$(VERSION) -f $(BUILD_DIR)/Modelfile
	touch $(BUILD_DIR)/.ollama

run: $(BUILD_DIR)/.ollama build/.venv
	. .venv/bin/activate && python3 $(CHAT_SCRIPT) --model bantz-$(VERSION) --width $(WIDTH) "$(PROMPT)"

prompt: $(BUILD_DIR)/analysis.md

test: $(BUILD_DIR)/.ollama build/.venv
	. .venv/bin/activate && python3 tests/run_tests.py --model bantz-$(VERSION)

versions:
	@ls versions/*.mk 2>/dev/null | sed 's|versions/||;s|\.mk$$||' || printf '(no versions defined)\n'

clean:
	rm -rf $(BUILD_DIR) .venv llama.cpp/.venv

clean-all:
	rm -rf build .venv llama.cpp/.venv

# Parameter stamp files: each file stores the current values of the parameters
# for its step. FORCE ensures the recipe always runs to check the values, but
# the file is only written (and thus made newer than its dependents) when the
# values have actually changed. This triggers a rebuild of the affected step.

$(BUILD_DIR)/.corpus-params: FORCE
	@mkdir -p $(BUILD_DIR)
	@printf 'WINDOW=%s\nVALID_SPLIT=%s\n' '$(WINDOW)' '$(VALID_SPLIT)' | cmp -s - $@ \
		|| printf 'WINDOW=%s\nVALID_SPLIT=%s\n' '$(WINDOW)' '$(VALID_SPLIT)' > $@

$(BUILD_DIR)/.train-params: FORCE
	@mkdir -p $(BUILD_DIR)
	@printf 'BASE_MODEL=%s\nITERS=%s\nBATCH=%s\nMAX_SEQ=%s\n' '$(BASE_MODEL)' '$(ITERS)' '$(BATCH)' '$(MAX_SEQ)' | cmp -s - $@ \
		|| printf 'BASE_MODEL=%s\nITERS=%s\nBATCH=%s\nMAX_SEQ=%s\n' '$(BASE_MODEL)' '$(ITERS)' '$(BATCH)' '$(MAX_SEQ)' > $@
