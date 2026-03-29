"""Parse chat log dumps from github.com/carderne/signal-export.

Each exported log is a Markdown file where each message starts with a header
line of the form ``[YYYY-MM-DD HH:MM:SS] Name: text``, optionally followed by
continuation lines, a quoted reply line (``> ...``), and a reaction line
(``(-emoji-)``).
"""

import re
from pathlib import Path

from common import Message

HEADER = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] ([^:]+): ?(.*)")
REACTION = re.compile(r"^\(-(.+?)-\)")
QUOTE = re.compile(r"^> (.+)")
URL = re.compile(r"https?://\S+")
IMG = re.compile(r"!?\[.*?\]\(.*?\)")
MEDIA = "\ufffc"  # object replacement character


def _clean(line: str) -> str:
    """Strip URLs, Markdown image syntax, and the Unicode object-replacement character from a line."""
    line = URL.sub("", line)
    line = IMG.sub("", line)
    line = line.replace(MEDIA, "")
    return line.strip()


def parse_messages(path: Path) -> list[Message]:
    """Parse a signal-export markdown file into a list of Messages."""
    messages = []
    current = None

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()

        m = HEADER.match(line)
        if m:
            current = Message(timestamp=m.group(1), user=m.group(2))
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
