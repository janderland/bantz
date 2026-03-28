#!/usr/bin/env python3
"""Parse a chat log into JSONL training data for mlx-lm."""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

HEADER = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] ([^:]+): ?(.*)")
REACTION = re.compile(r"^\(-(.+?)-\)")
QUOTE = re.compile(r"^> (.+)")
URL = re.compile(r"https?://\S+")
IMG = re.compile(r"!?\[.*?\]\(.*?\)")
MEDIA = "\ufffc"  # object replacement character



@dataclass
class Message:
    timestamp: str
    user: str
    lines: list[str] = field(default_factory=list)
    reaction: str = ""
    quote_text: str = ""
    reply_to: int = -1


def parse_messages(path: Path, usermap: dict[str, str]) -> list[Message]:
    messages = []
    current = None

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()

        m = HEADER.match(line)
        if m:
            user = m.group(2)
            user = usermap.get(user, user)
            if not user:
                current = None
                continue
            current = Message(timestamp=m.group(1), user=user)
            if m.group(3):
                current.lines.append(m.group(3))
            messages.append(current)
            continue

        if current is None:
            continue

        # Skip blank lines.
        if not line:
            continue

        # Parse reaction lines and attach to the current message.
        m = REACTION.match(line)
        if m:
            current.reaction = f"( {m.group(1).strip()} )"
            continue

        # Extract quote text from the first quote line found.
        if not current.quote_text:
            m = QUOTE.match(line)
            if m:
                current.quote_text = m.group(1).strip()

        current.lines.append(line)

    return messages


def resolve_replies(messages: list[Message]) -> None:
    for i, msg in enumerate(messages):
        if not msg.quote_text:
            continue
        for j in range(i - 1, -1, -1):
            candidate = messages[j]
            body = "\n".join(candidate.lines)
            if msg.quote_text in body:
                msg.reply_to = j
                break


def format_message(msg: Message) -> str:
    body = "\n".join(msg.lines).strip()
    body = URL.sub("", body)
    body = IMG.sub("", body)
    body = body.replace(MEDIA, "").strip()
    # Drop messages that are empty or media-only.
    if not body:
        return ""
    result = f"{msg.user}: {body}"
    if msg.reaction:
        result += f"\n{msg.reaction}"
    return result


def make_training_examples(messages: list[Message], window: int) -> list[dict]:
    formatted = [format_message(msg) for msg in messages]

    examples = []
    for i, msg in enumerate(messages):
        completion = formatted[i]
        if not completion:
            continue

        if msg.reply_to >= 0:
            # Reply example: context window ending with the original message.
            orig = msg.reply_to
            start = max(0, orig - window + 1)
            ctx = [t for t in formatted[start : orig + 1] if t]
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

        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        examples.append({"prompt": prompt, "completion": f"{completion}<|eot_id|>"})

    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse chat log into JSONL training data.")
    parser.add_argument("input", type=Path, help="Input chat log file")
    parser.add_argument("output", type=Path, nargs="?", default=Path("train.jsonl"), help="Output JSONL file")
    parser.add_argument("-f", "--map-file", metavar="FILE", type=Path, default=Path("usermap"),
                        help="File of OLD=NEW mappings, one per line (default: usermap)")
    parser.add_argument("-w", "--window", type=int, default=6, metavar="N",
                        help="Number of preceding messages used as context (default: 6)")
    args = parser.parse_args()

    usermap = {}
    map_file: Path = args.map_file
    if map_file.exists():
        for line in map_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                print(f"Error: invalid mapping {line!r} in {map_file}, expected OLD=NEW")
                sys.exit(1)
            old, new = line.split("=", 1)
            usermap[old.strip()] = new.strip()

    chat_path = args.input
    out_path = args.output

    print(f"Parsing {chat_path}...")
    messages = parse_messages(chat_path, usermap)
    print(f"  Found {len(messages)} messages")

    resolve_replies(messages)
    replies = sum(1 for m in messages if m.reply_to >= 0)
    print(f"  Resolved {replies} replies")

    examples = make_training_examples(messages, args.window)
    print(f"  Generated {len(examples)} training examples")

    out_path.write_text("\n".join(json.dumps(e) for e in examples), encoding="utf-8")
    print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
