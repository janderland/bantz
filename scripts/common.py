"""Shared data structures and utilities used across the pipeline."""

import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Message:
    timestamp: str
    user: str
    lines: list[str] = field(default_factory=list)
    reaction: str = ""
    quote_text: str = ""
    reply_to: int = -1

    def format(self) -> str:
        body = "\n".join(self.lines).strip()
        if not body:
            return ""
        result = f"{self.user}: {body}"
        if self.reaction:
            result += f"\n{self.reaction}"
        return result


def load_usermap(path: Path) -> dict[str, str]:
    usermap = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                print(f"Error: invalid mapping {line!r} in {path}, expected OLD=NEW")
                sys.exit(1)
            old, new = line.split("=", 1)
            usermap[old.strip()] = new.strip()
    return usermap
