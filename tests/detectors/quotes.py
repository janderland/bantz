import re
from .base import Detector, Issue

# Matches a quote line: "> Speaker: text" or just "> text"
QUOTE_LINE = re.compile(r'^>\s*(.+)$')
# Extracts the body after "> Speaker: " if present, otherwise the whole thing
QUOTE_BODY = re.compile(r'^>\s*\w+:\s*(.+)$')


def _quote_body(line: str) -> str:
    """Extract the quoted text from a quote line, stripping the "> Speaker: " prefix."""
    m = QUOTE_BODY.match(line)
    if m:
        return m.group(1).strip()
    m = QUOTE_LINE.match(line)
    if m:
        return m.group(1).strip()
    return line.strip()


class QuoteDetector(Detector):
    name = "quotes"

    def analyze(self, prompt: str, messages: list) -> list:
        issues = []

        for i, msg in enumerate(messages):
            lines = [l for l in msg["lines"] if l.strip()]
            if not lines:
                continue

            quote_lines = [l for l in lines if QUOTE_LINE.match(l)]
            non_quote_lines = [l for l in lines if not QUOTE_LINE.match(l)]

            # Detector 1: quote_only_message
            # The speaker's entire turn is nothing but quote lines — no original text.
            if quote_lines and not non_quote_lines:
                issues.append(Issue(
                    detector="quote_only_message",
                    speaker=msg["speaker"],
                    context="\n".join(lines[:3]),
                ))

            # Detector 2: ungrounded_quote
            # A quoted text that doesn't appear in the prompt or any earlier message.
            for qline in quote_lines:
                body = _quote_body(qline)
                if not body:
                    continue
                # Build the search corpus: prompt + all earlier messages' lines
                earlier_text = prompt
                for j in range(i):
                    earlier_text += " " + " ".join(messages[j]["lines"])
                if body.lower() not in earlier_text.lower():
                    issues.append(Issue(
                        detector="ungrounded_quote",
                        speaker=msg["speaker"],
                        context=qline,
                    ))

        return issues
