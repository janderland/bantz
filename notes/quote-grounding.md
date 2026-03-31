# Fix: Ground Quotes to Context Window

## Problem

The model hallucinates quotes — it generates `> Speaker: text` lines referencing
messages that aren't present in the current conversation context. It learned the
quote *format* from training data but not the constraint that you can only quote
what's currently visible. Many training examples contain quotes referencing messages
from much earlier in the chat log that fall outside the context window.

## Solution

When building training examples in `corpus.py`, validate every `> Speaker: text`
line in the target message against the context window portion of that example. If
the quoted text isn't found in the context, strip the quote line before writing the
example to the JSONL corpus.

This teaches the model the constraint directly — "quotes only come from what you
can see" — without changing the output format or adding post-processing complexity.

## Notes

- Pair this with a post-processing validation step in `chat.py` as a safety net:
  after generation, strip any quote whose text can't be found in the conversation
  history passed to the model.
- An alternative approach (ID-based references) was considered but rejected: the
  model still needs to decide *which* message to reference, so it can still
  hallucinate IDs, and it requires learning an artificial format not present in the
  original chat data.
