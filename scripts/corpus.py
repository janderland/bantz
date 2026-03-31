#!/usr/bin/env python3
"""Parse a chat log into JSONL training data for mlx-lm."""

import argparse
import json
import random
from pathlib import Path

from common import Message, load_usermap
from parse_signal import parse_messages


def make_arg_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Parse chat log into JSONL training data.")
    parser.add_argument("input", type=Path, help="Input chat log file")
    parser.add_argument("output", type=Path, nargs="?", default=Path("train.jsonl"), help="Output JSONL file")
    parser.add_argument("-f", "--map-file", metavar="FILE", type=Path, default=Path("usermap"),
                        help="File of OLD=NEW mappings, one per line (default: usermap)")
    parser.add_argument("-w", "--window", type=int, default=6, metavar="N",
                        help="Number of preceding messages used as context (default: 6)")
    parser.add_argument("-s", "--valid-split", type=int, default=10, metavar="PCT",
                        help="Percentage of examples held out for validation (0-100, default: 10)")
    return parser


def apply_usermap(messages: list[Message], usermap: dict[str, str]) -> list[Message]:
    """Rename users and filter messages according to *usermap*.

    For each message, if the sender's name has an entry in *usermap* the name
    is replaced with the mapped value.  An empty mapped value (``""``) means
    the message is dropped entirely.  Messages whose sender has no entry are
    kept with their original name.
    """
    result = []
    for msg in messages:
        name = usermap.get(msg.user, msg.user)
        if not name:
            continue
        msg.user = name
        result.append(msg)
    return result


def resolve_replies(messages: list[Message]) -> int:
    """Set ``reply_to`` on each message that quotes another message.

    Searches backwards from each quoting message for the most recent message
    whose body contains the quoted text.  Sets ``msg.reply_to`` to that
    message's index and returns the total number of resolved replies.
    """
    resolved = 0
    for i, msg in enumerate(messages):
        if not msg.quote_text:
            continue
        for j in range(i - 1, -1, -1):
            candidate = messages[j]
            body = "\n".join(candidate.lines)
            if msg.quote_text in body:
                msg.reply_to = j
                resolved += 1
                break
    return resolved


def make_corpus(messages: list[Message], window: int) -> list[dict]:
    """Build a list of prompt/completion training examples from *messages*.

    Each example is a dict with ``"prompt"`` and ``"completion"`` keys
    formatted for MLX-LM LoRA fine-tuning using Llama 3 chat tokens.

    The prompt ends with ``Speaker:`` so the model learns to generate in a
    specific person's voice.  For reply messages, the context window includes
    messages around the original message *and* messages leading up to the
    reply; standalone messages use a sliding window of the preceding messages.
    Messages with no non-empty body are skipped.
    """
    formatted = [msg.format() for msg in messages]

    corpus = []
    for i, msg in enumerate(messages):
        completion = formatted[i]
        if not completion:
            continue

        if msg.reply_to >= 0:
            # Reply example: a window of context around the original message
            # plus a window of context leading up to the reply.
            orig = msg.reply_to
            orig_ctx = [t for t in formatted[max(0, orig - window + 1) : orig + 1] if t]
            reply_ctx = [t for t in formatted[max(0, i - window) : i] if t]
            ctx = orig_ctx + reply_ctx
            if not ctx:
                continue
            context = "\n".join(ctx)
        else:
            # Standalone example: sliding window of prior messages.
            if i < window:
                continue
            ctx = [t for t in formatted[i - window : i] if t]
            if not ctx:
                continue
            context = "\n".join(ctx)

        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{context}\n{msg.user}:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        corpus.append({"prompt": prompt, "completion": f"{completion}<|eot_id|>"})

    return corpus


def write_corpus(corpus: list[dict], path: Path) -> None:
    """Write *corpus* to *path* as a JSONL file (one JSON object per line)."""
    path.write_text("\n".join(json.dumps(e) for e in corpus), encoding="utf-8")


def main() -> None:
    """Parse arguments, build the training corpus, and write it to disk."""
    args = make_arg_parser().parse_args()

    usermap = load_usermap(args.map_file)

    print(f"Parsing {args.input}...")
    messages = parse_messages(args.input)
    messages = apply_usermap(messages, usermap)
    print(f"  Found {len(messages)} messages")

    replies = resolve_replies(messages)
    print(f"  Resolved {replies} replies")

    corpus = make_corpus(messages, args.window)
    print(f"  Generated {len(corpus)} examples")

    if args.valid_split > 0:
        random.shuffle(corpus)
        n_valid = int(len(corpus) * args.valid_split / 100)
        valid_path = args.output.parent / "valid.jsonl"
        write_corpus(corpus[:n_valid], valid_path)
        print(f"  Wrote {n_valid} validation examples to {valid_path}")
        write_corpus(corpus[n_valid:], args.output)
        print(f"  Wrote {len(corpus) - n_valid} training examples to {args.output}")
    else:
        write_corpus(corpus, args.output)
        print(f"  Wrote {len(corpus)} examples to {args.output}")


if __name__ == "__main__":
    main()
