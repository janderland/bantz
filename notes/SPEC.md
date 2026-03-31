# bantz.py — Output Transformation Specification

This document describes all transformations `bantz.py` applies to the raw model
output so that an alternate implementation can reproduce identical behavior.

---

## Overview

`bantz.py` takes a user prompt, feeds it to the `bantz` model via the Ollama
API, and reformats the streaming output into a styled chat transcript for the
terminal.

**Raw model format** (what the model produces):

```
Alice: Did you see the game?
( Bob: 😂 )
Carol: > Alice: Did you see the game?
Yeah what a match
```

**Formatted output** (what the user sees):

```
# Alice
Did you see the game?
( Bob: 😂 )

# Carol
> Alice: Did you see the game?

Yeah what a match
```

---

## Input

The user prompt is passed verbatim to the model as the `prompt` field. The
model continues the conversation from where the prompt ends. The prompt is also
processed through the same formatting pipeline before streaming begins, so any
pre-existing conversation in the prompt is rendered the same way.

---

## Step 1 — Sanitize special tokens

As tokens accumulate in the stream buffer, any `<|...|>` sequences are stripped
before further processing:

```
<\|[^|]*\|>  →  (removed)
```

This handles model format tokens such as `<|im_startitem|>` and `<|im_enditem|>`
that the model may emit as structural markers. Stripping is applied to the
accumulated buffer (not per-token) so that tokens split across multiple stream
chunks are handled correctly.

---

## Step 2 — Line splitting

The stream buffer is split on `\n`. Complete lines are processed one at a time;
any partial line at the end of the buffer is held until more tokens arrive.
Each line is stripped of leading and trailing whitespace before processing.

---

## Step 3 — Per-line classification and transformation

Each line falls into one of four categories, checked in order:

### 3a. Reaction lines

A reaction matches this pattern:

```
\( \w+: \S+(?:, \w+: \S+)* \)
```

Examples: `( Bob: 😂 )`, `( Bob: 😂, Alice: 🙌 )`

A line may contain a reaction mixed with other content (e.g. after stripping a
`<|im_startitem|>` token, a reaction and a speaker line may be fused). In that
case the line is split on reaction tokens, and each piece is processed
independently — reactions are handled as reactions, non-empty remainder pieces
are re-processed as new lines.

When a reaction is encountered:
- Any accumulated paragraph is flushed first.
- The reaction string is printed as-is on its own line, with no blank line
  before or after it (reactions are considered attached to the message above).

### 3b. Speaker lines

A line is a speaker line if it contains `": "` and the text before the first
`": "` is entirely alphabetic (i.e. `prefix.isalpha()` is true).

```
Name: message content here
```

When a speaker line is encountered:
- Any accumulated paragraph is flushed.
- If a previous speaker was already confirmed (i.e. at least one non-quote
  message had been printed for them), a blank line is printed to separate the
  two speaker blocks.
- If the previous speaker had not yet been confirmed (e.g. the model started a
  speaker block but only produced quotes, which are held back), their buffered
  content is discarded.
- A `pending_header` is set to `# Name` but **not yet printed**.
- The content after `": "` is added to the paragraph buffer.

The header is held back ("unconfirmed") until a non-quote paragraph is ready to
print for this speaker. This prevents orphaned headers if the model abandons a
speaker mid-generation.

### 3c. Blank / empty lines

A line that is empty or contains only `>` signals a paragraph break. The
current paragraph buffer is flushed (see Step 4).

### 3d. Continuation lines

Any other line is appended to the paragraph buffer. If the new line's
quote-status (starts with `>` or not) differs from the existing buffer, the
buffer is flushed first, starting a new paragraph.

---

## Step 4 — Paragraph flushing

The paragraph buffer holds the raw token fragments of a single paragraph. When
flushed:

1. All fragments are joined with spaces and the result is collapsed: multiple
   consecutive whitespace characters become a single space.
2. If the merged text is empty, nothing is emitted.
3. The paragraph is classified as a **quote** if it starts with `>`, or a
   **regular message** otherwise.
4. If it is a regular message, the speaker is confirmed at this point:
   - The pending `# Name` header is printed.
   - Any previously buffered lines for this speaker (e.g. reactions or quotes
     that arrived before the first non-quote message) are printed.
5. The previous pending line (if any) is printed before this paragraph is
   stored as the new pending line. A blank line is inserted between the two if
   one was a quote and the other was not (transition between quote and
   non-quote paragraphs within the same speaker block).

The last pending line is printed at the very end of the stream, followed by a
final blank line if a speaker was confirmed.

---

## Step 5 — Line wrapping (optional)

If a `--width` argument is provided, each line is word-wrapped to that column
width before printing. Quote lines (starting with `> `) use `> ` as the
continuation indent so wrapped lines remain visually indented.

---

## Output format summary

```
# SpeakerName
message paragraph one

> QuotedSpeaker: quoted text

message paragraph two (after quote, blank line separating them)
( Reactor: emoji, Reactor2: emoji )

# NextSpeaker
...
```

Key formatting rules:
- Each speaker block begins with `# Name`.
- A blank line separates consecutive speaker blocks.
- A blank line separates a quote paragraph from a non-quote paragraph within
  the same speaker block.
- Reactions are printed immediately after the line they follow, with no blank
  line before or after.
- Quotes are held back until a non-quote message confirms the speaker; then
  both the header and the buffered quotes are printed together.

---

## CLI

```
python3 bantz.py [-w WIDTH] [-r FILE] <prompt words...>
```

- `-w WIDTH` / `--width WIDTH` — wrap output to this column width
- `-r FILE` / `--raw FILE` — write the raw (unformatted) model tokens to FILE
  in addition to printing formatted output
- `<prompt words...>` — joined with spaces to form the prompt
