"""Shared message data structure used across parsers."""

from dataclasses import dataclass, field


@dataclass
class Message:
    timestamp: str
    user: str
    lines: list[str] = field(default_factory=list)
    reaction: str = ""
    quote_text: str = ""
    reply_to: int = -1
