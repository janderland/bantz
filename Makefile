INPUT ?= input.md

.PHONY: all run clean

all: build/.ollama

build/.venv: requirements.txt
	mkdir -p build
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	touch build/.venv

build/.venv-llama: llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
	mkdir -p build
	python3.10 -m venv llama.cpp/.venv
	llama.cpp/.venv/bin/pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
	touch build/.venv-llama

build/data/train.jsonl: build/.venv $(INPUT) scripts/parse.py
	mkdir -p build/data
	. .venv/bin/activate && python3 scripts/parse.py $(INPUT) build/data/train.jsonl $(MAP)

build/.train: build/data/train.jsonl scripts/train.sh
	mkdir -p build/adapters
	bash scripts/train.sh
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

parse: build/data/train.jsonl
train: build/.train
fuse: build/.fuse
gguf: build/.gguf

run: build/.ollama build/.venv
	. .venv/bin/activate && python3 scripts/chat.py $(PROMPT)

clean:
	rm -rf build .venv llama.cpp/.venv
