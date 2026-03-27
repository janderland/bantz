#!/usr/bin/env python3
"""Analyze a chat log to extract personality notes and inside jokes."""

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from parse import Message, format_message, parse_messages


def ollama(model: str, prompt: str) -> str:
    data = json.dumps({"model": model, "prompt": prompt, "stream": True}).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        headers={"Content-Type": "application/json"},
        data=data,
    )
    parts: list[str] = []
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            obj = json.loads(line)
            parts.append(obj.get("response", ""))
            if obj.get("done"):
                break
    return "".join(parts).strip()


def sample_user_messages(messages: list[Message], user: str, n: int) -> list[str]:
    user_msgs = [format_message(m) for m in messages if m.user == user]
    user_msgs = [t for t in user_msgs if t]
    if not user_msgs:
        return []
    # Strip "USER: " prefix — the prompt already names the user.
    bodies = [t.split(": ", 1)[1] if ": " in t else t for t in user_msgs]
    if len(bodies) <= n:
        return bodies
    step = len(bodies) // n
    return bodies[::step][:n]


def analyze_personality(
    model: str,
    user: str,
    messages: list[Message],
    sample: int,
    verbose: bool,
) -> str | None:
    sampled = sample_user_messages(messages, user, sample)
    if len(sampled) < 5:
        if verbose:
            print(f"  Skipping {user} (fewer than 5 messages)", file=sys.stderr)
        return None
    if verbose:
        print(f"  Analyzing {user} ({len(sampled)} messages sampled)...", file=sys.stderr)
    prompt = f"""You are analyzing a group chat to help characterize each participant.

Below are messages written by {user}. Based only on these messages, write exactly 4 bullet points \
describing this person's communication style. Cover: tone and humor, topics they care about, how \
they interact with others, and any distinctive habits or phrases.

Be specific. Use present tense. Do not invent things not in the messages. Output only the 4 bullet \
points, nothing else.

Messages:
{chr(10).join(sampled)}"""
    return ollama(model, prompt)


def chunk_messages(messages: list[Message], chunk_size: int) -> list[list[str]]:
    formatted = [format_message(m) for m in messages]
    formatted = [t for t in formatted if t]
    return [formatted[i : i + chunk_size] for i in range(0, len(formatted), chunk_size)]


def analyze_jokes(
    model: str,
    messages: list[Message],
    chunk_size: int,
    verbose: bool,
) -> str:
    chunks = chunk_messages(messages, chunk_size)
    candidates: list[str] = []

    for i, chunk in enumerate(chunks, 1):
        if verbose:
            print(f"  Scanning chunk {i}/{len(chunks)} for inside jokes...", file=sys.stderr)
        prompt = f"""Look for inside jokes in this group chat segment: recurring nicknames, \
callback phrases, references that assume shared history, or running gags. Ignore generic slang \
or common internet phrases.

For each one, output a line in this format:
- PHRASE: brief note on why it seems like an inside joke

List at most 8 candidates. If nothing notable, output: NONE

{chr(10).join(chunk)}"""
        result = ollama(model, prompt)
        if result.strip().upper() != "NONE":
            candidates.append(result)

    if not candidates:
        return "No clear inside jokes identified."

    # Deduplicate by lowercased first word of each bullet line to reduce synthesis prompt size.
    seen: set[str] = set()
    deduped: list[str] = []
    for block in candidates:
        for line in block.splitlines():
            line = line.strip()
            if not line or not line.startswith("-"):
                continue
            key = line.lstrip("- ").split()[0].lower().rstrip(",:") if line.lstrip("- ") else ""
            if key and key not in seen:
                seen.add(key)
                deduped.append(line)

    if verbose:
        print(f"  Synthesizing {len(deduped)} joke candidates...", file=sys.stderr)

    synthesis_prompt = f"""Consolidate these inside joke candidates found in a group chat. \
Merge duplicates, discard one-offs that are unlikely to be real group in-jokes. For each \
genuine inside joke write one bullet point: the phrase or reference, followed by a plain-English \
explanation of what it is and how the group uses it. Aim for 5–10 total.

{chr(10).join(deduped)}"""
    return ollama(model, synthesis_prompt)


def format_output(personalities: dict[str, str], jokes: str) -> str:
    parts: list[str] = []
    if personalities:
        parts.append("## Personality Notes")
        for user in sorted(personalities):
            parts.append(f"\n### {user}\n{personalities[user].strip()}")
    if jokes:
        if parts:
            parts.append("")
        parts.append("## Inside Jokes")
        parts.append(jokes.strip())
    return "\n".join(parts) + "\n"


def load_usermap(map_file: Path) -> dict[str, str]:
    usermap: dict[str, str] = {}
    if not map_file.exists():
        return usermap
    for line in map_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            print(f"Warning: invalid mapping {line!r} in {map_file}", file=sys.stderr)
            continue
        old, new = line.split("=", 1)
        usermap[old.strip()] = new.strip()
    return usermap


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze chat log for personalities and inside jokes.")
    parser.add_argument("input", type=Path, help="Input chat log file")
    parser.add_argument("-o", "--output", type=Path, metavar="FILE", help="Also write output to FILE")
    parser.add_argument("-m", "--model", default="llama3.1:8b", metavar="NAME",
                        help="Ollama model name (default: llama3.1:8b)")
    parser.add_argument("-f", "--map-file", type=Path, default=Path("usermap"), metavar="FILE",
                        help="Usermap file (default: usermap)")
    parser.add_argument("-s", "--sample", type=int, default=150, metavar="N",
                        help="Max messages sampled per user for personality (default: 150)")
    parser.add_argument("-c", "--chunk", type=int, default=300, metavar="N",
                        help="Messages per chunk for jokes analysis (default: 300)")
    parser.add_argument("--jokes-only", action="store_true", help="Skip personality analysis")
    parser.add_argument("--personality-only", action="store_true", help="Skip inside jokes analysis")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print progress to stderr")
    args = parser.parse_args()

    usermap = load_usermap(args.map_file)

    if args.verbose:
        print(f"Parsing {args.input}...", file=sys.stderr)
    messages = parse_messages(args.input, usermap)
    if args.verbose:
        print(f"  {len(messages)} messages", file=sys.stderr)

    # Preserve order of first appearance.
    seen_users: list[str] = []
    for m in messages:
        if m.user not in seen_users:
            seen_users.append(m.user)

    personalities: dict[str, str] = {}
    if not args.jokes_only:
        if args.verbose:
            print("\nAnalyzing personalities...", file=sys.stderr)
        for user in seen_users:
            result = analyze_personality(args.model, user, messages, args.sample, args.verbose)
            if result:
                personalities[user] = result

    jokes = ""
    if not args.personality_only:
        if args.verbose:
            print("\nAnalyzing inside jokes...", file=sys.stderr)
        try:
            jokes = analyze_jokes(args.model, messages, args.chunk, args.verbose)
        except urllib.error.URLError as e:
            if "Connection refused" in str(e):
                print("Error: cannot connect to Ollama at localhost:11434. Is it running?",
                      file=sys.stderr)
                sys.exit(1)
            raise

    output = format_output(personalities, jokes)
    print(output, end="")

    if args.output:
        args.output.write_text(output, encoding="utf-8")
        print(f"\nWrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as e:
        if "Connection refused" in str(e):
            print("Error: cannot connect to Ollama at localhost:11434. Is it running?",
                  file=sys.stderr)
            sys.exit(1)
        raise
