#!/usr/bin/env python3
"""Parse a chat log into JSONL training data for mlx-lm."""

import argparse
import json
from pathlib import Path

from common import Message, load_usermap
from parse_signal import parse_messages


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse chat log into JSONL training data.")
    parser.add_argument("input", type=Path, help="Input chat log file")
    parser.add_argument("output", type=Path, nargs="?", default=Path("train.jsonl"), help="Output JSONL file")
    parser.add_argument("-f", "--map-file", metavar="FILE", type=Path, default=Path("usermap"),
                        help="File of OLD=NEW mappings, one per line (default: usermap)")
    parser.add_argument("-w", "--window", type=int, default=6, metavar="N",
                        help="Number of preceding messages used as context (default: 6)")
    return parser


def resolve_replies(messages: list[Message]) -> int:
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
    path.write_text("\n".join(json.dumps(e) for e in corpus), encoding="utf-8")


def main() -> None:
    args = make_arg_parser().parse_args()

    usermap = load_usermap(args.map_file)

    print(f"Parsing {args.input}...")
    messages = parse_messages(args.input, usermap)
    print(f"  Found {len(messages)} messages")

    replies = resolve_replies(messages)
    print(f"  Resolved {replies} replies")

    corpus = make_corpus(messages, args.window)
    print(f"  Generated {len(corpus)} examples")

    write_corpus(corpus, args.output)
    print(f"  Wrote {args.output}")


if __name__ == "__main__":
    main()
