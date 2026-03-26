# bantz

Fine-tune a Llama 3.2 3B model on a group chat log and use it to generate
realistic group chat conversations.

## Overview

The pipeline has four stages:

1. **Parse** — Convert a chat log into JSONL training examples
2. **Train** — LoRA fine-tune `mlx-community/Llama-3.2-3B-Instruct` via MLX-LM
3. **Fuse** — Merge the LoRA adapters into the base model
4. **GGUF** — Convert the fused model to GGUF format and load it into Ollama

Once built, `scripts/chat.py` streams responses from Ollama and formats them
as a styled terminal chat transcript.

## Prerequisites

- macOS with Apple Silicon (MLX requirement)
- Python 3 and Python 3.10 (`brew install python@3.10`)
- [Ollama](https://ollama.com) running locally
- ~15 GB of free disk space

## Quick Start

```sh
make run PROMPT="what should we do tonight"
```

This builds the full pipeline if needed, then streams a generated conversation.

## Preparing Your Data

### Chat log (`input.md`)

Place your chat log at `input.md` in the repo root (it is gitignored), or
pass a custom path:

```sh
make INPUT=path/to/export.md
```

Each message begins with a header line, followed by zero or more content lines:

```
[YYYY-MM-DD HH:MM:SS] Username: optional first line
body line
> quoted text from a previous message
(-👍 😂 ❤️-)
```

**Header** (required, starts a new message):
```
[YYYY-MM-DD HH:MM:SS] Username: optional text
```
Any text after the colon becomes the first body line of the message.

**Body lines**: Any non-blank line that isn't a reaction or quote. Added to
the message body in order.

**Quote lines**: Must start with `> `. Added to the message body. The first
quote line is also used to link the message as a reply to the original.
Multiple quote lines are allowed.

**Reaction lines**: Must match `(-content-)` at the start of the line.
The content between `(-` and `-)` is captured as-is — multiple reactions can
be included within one pair, e.g. `(-👍 😂-)` or `(-👍, 😂, ❤️-)`. Only
one reaction line per message is kept; if multiple appear, the last one wins.
Reactions are stored separately from the body and are not included in the
message text.

**Blank lines**: Ignored anywhere in the input.

Lines before the first header are ignored. URLs, image markdown, and the
Unicode object replacement character (U+FFFC) are stripped from the body
before the training data is written. Messages whose body is empty after
stripping are excluded entirely.

### Username mapping (`usermap`)

Create a `usermap` file (gitignored) to rename users before training data is
written. One mapping per line:

```
Me=Alice
RealName=Bob
# lines starting with # are ignored
```

## Build Pipeline

All output goes into `build/`, which is gitignored. Steps only re-run when
their inputs change.

| Target | Description |
|--------|-------------|
| `make` | Run the full pipeline |
| `make parse` | Parse the chat log into `build/data/train.jsonl` |
| `make train` | LoRA fine-tune the base model |
| `make fuse` | Merge adapters into a full model |
| `make gguf` | Convert to GGUF and register with Ollama |
| `make run PROMPT="..."` | Generate a conversation |
| `make clean` | Delete `build/`, `.venv/`, and `llama.cpp/.venv/` |

## Inference

```sh
make run PROMPT="did you see the game"
```

Or directly:

```sh
source .venv/bin/activate
python3 scripts/chat.py [-w WIDTH] [-r FILE] <prompt>
```

| Flag | Description |
|------|-------------|
| `-w WIDTH` | Word-wrap output to this column width |
| `-r FILE` | Also write raw model tokens to FILE |

The formatter strips special tokens, detects speaker lines, quotes, and
reactions, and renders them as a structured terminal transcript. See
[SPEC.md](SPEC.md) for the full formatting specification.

