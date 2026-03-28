"""Parse chat log dumps from github.com/carderne/signal-export."""

import re
from pathlib import Path

from message import Message

HEADER = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] ([^:]+): ?(.*)")
REACTION = re.compile(r"^\(-(.+?)-\)")
QUOTE = re.compile(r"^> (.+)")
URL = re.compile(r"https?://\S+")
IMG = re.compile(r"!?\[.*?\]\(.*?\)")
MEDIA = "\ufffc"  # object replacement character


def _clean(line: str) -> str:
    line = URL.sub("", line)
    line = IMG.sub("", line)
    line = line.replace(MEDIA, "")
    return line.strip()


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
                cleaned = _clean(m.group(3))
                if cleaned:
                    current.lines.append(cleaned)
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

        cleaned = _clean(line)
        if cleaned:
            current.lines.append(cleaned)

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
