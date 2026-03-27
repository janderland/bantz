#!/usr/bin/env python3
"""Analyze a chat log to extract personality notes and group references.

Group references cover in-group phrases/jokes, unusual vocabulary, and
recurring topics (people, places, restaurants, etc.) that characterize
this particular group.
"""

import argparse
import atexit
import json
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from parse import Message, format_message, parse_messages

from wordfreq import word_frequency, zipf_frequency


def unload_model(model: str) -> None:
    try:
        data = json.dumps({"model": model, "keep_alive": 0}).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            headers={"Content-Type": "application/json"},
            data=data,
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


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


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "have", "has", "had", "do", "did", "does", "will", "would", "could",
    "should", "may", "might", "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "its", "our",
    "their", "this", "that", "these", "those", "what", "which", "who",
    "when", "where", "how", "why", "if", "so", "not", "no", "up", "out",
    "just", "like", "get", "got", "go", "going", "know", "think", "yeah",
    "ok", "okay", "lol", "haha", "oh", "ah", "hey", "hi", "there", "about",
    "also", "one", "all", "some", "more", "than", "then", "now", "can",
    "want", "need", "really", "actually", "literally", "basically", "still",
    "even", "back", "here", "way", "time", "thing", "things", "people",
    "good", "make", "see", "look", "come", "too", "very", "much", "well",
    # Contractions with apostrophe stripped (normalized before tokenizing).
    "im", "dont", "cant", "ill", "thats", "ive", "youre", "weve",
    "theyre", "isnt", "wasnt", "wouldnt", "couldnt", "shouldnt", "didnt",
    "doesnt", "hasnt", "havent", "id", "hed", "shed", "wed", "theyd",
    "wont", "lets", "hes", "shes", "whats", "theres", "hows", "whos",
}


def extract_ngrams(
    messages: list[Message], users: list[str], min_count: int = 3, top_n: int = 50
) -> list[tuple[str, int, list[str]]]:
    # Build a set of individual words from all usernames to filter out name n-grams.
    name_words = {w for user in users for w in re.findall(r"[a-z]+", user.lower())}
    counts: Counter = Counter()
    phrase_users: dict[str, set[str]] = {}
    for msg in messages:
        body = format_message(msg)
        if not body or ": " not in body:
            continue
        body = body.split(": ", 1)[1]
        # Strip apostrophes so contractions tokenize as single words (e.g. "dont", "cant").
        body = body.replace("\u2019", "").replace("\u2018", "").replace("'", "")
        words = re.findall(r"[a-z]+", body.lower())
        for n in (2, 3, 4):
            for i in range(len(words) - n + 1):
                gram = tuple(words[i : i + n])
                if any(w in name_words for w in gram):
                    continue
                content_words = sum(1 for w in gram if w not in STOPWORDS)
                if content_words < 2:
                    continue
                phrase = " ".join(gram)
                counts[phrase] += 1
                phrase_users.setdefault(phrase, set()).add(msg.user)
    return [
        (p, c, sorted(phrase_users[p]))
        for p, c in counts.most_common(top_n)
        if c >= min_count
    ]


def extract_frequent_topics(
    messages: list[Message],
    users: list[str],
    min_count: int = 5,
    min_ratio: float = 10.0,
    top_n: int = 25,
) -> list[tuple[str, int, float, list[str]]]:
    """Find words used far more than their general English frequency would predict.

    The anomaly ratio is actual_frequency / expected_frequency. A high ratio
    means the group talks about this word much more than average — a signal for
    group-specific topics, proper nouns, and recurring subjects regardless of
    capitalization.
    """
    name_words = {w for user in users for w in re.findall(r"[a-z]+", user.lower())}
    counts: Counter = Counter()
    word_users: dict[str, set[str]] = {}
    total_words = 0

    for msg in messages:
        body = format_message(msg)
        if not body or ": " not in body:
            continue
        body = body.split(": ", 1)[1]
        body = body.replace("\u2019", "").replace("\u2018", "").replace("'", "")
        words = re.findall(r"[a-z]+", body.lower())
        total_words += len(words)
        for word in words:
            if word in STOPWORDS or word in name_words or len(word) < 3:
                continue
            counts[word] += 1
            word_users.setdefault(word, set()).add(msg.user)

    if total_words == 0:
        return []

    results = []
    for word, count in counts.items():
        if count < min_count:
            continue
        expected = word_frequency(word, "en")
        if expected == 0:
            ratio = float("inf")  # Not in general English at all.
        else:
            ratio = (count / total_words) / expected
        if ratio >= min_ratio:
            results.append((word, count, ratio, sorted(word_users[word])))

    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_n]


def analyze_references(model: str, messages: list[Message], users: list[str], verbose: bool) -> str:
    if verbose:
        print("  Extracting frequent phrases...", file=sys.stderr)
    phrases = extract_ngrams(messages, users)

    if verbose:
        print("  Extracting frequent topics...", file=sys.stderr)
    topics = extract_frequent_topics(messages, users)

    if not phrases and not topics:
        return "No recurring phrases or notable topics found."

    if verbose:
        print(
            f"  Found {len(phrases)} phrases and {len(topics)} notable words, "
            "asking LLM to interpret...",
            file=sys.stderr,
        )

    def user_note(u: list[str]) -> str:
        return f"only used by {u[0]}" if len(u) == 1 else f"used by {', '.join(u)}"

    phrase_lines = [
        f'- "{phrase}" (appears {count}x, {user_note(u)})'
        for phrase, count, u in phrases
    ]
    topic_lines = []
    for word, count, ratio, u in topics:
        ratio_str = "not in general English" if ratio == float("inf") else f"{ratio:.0f}x more than expected"
        topic_lines.append(f'- "{word}" (appears {count}x, {ratio_str}, {user_note(u)})')

    prompt = f"""The following data was extracted from a group chat. Use it to identify \
in-group phrases, recurring jokes, catchphrases, and common subjects of discussion \
(people, places, restaurants, events, etc.) that characterize this particular group.

**Frequent multi-word phrases** (may be catchphrases, running gags, or recurring references):
{chr(10).join(phrase_lines) if phrase_lines else "None found."}

**Words used far more than expected** (high anomaly ratio means the group talks about \
this much more than average English text; "not in general English" means the word is \
unknown outside this group):
{chr(10).join(topic_lines) if topic_lines else "None found."}

For each genuine group reference, write one bullet point explaining what it likely is \
and how the group uses it. Include both in-jokes and recurring topics. If something is \
only used by one person, note that. Aim for 5–10 total."""
    return ollama(model, prompt)


def format_output(personalities: dict[str, str], references: str) -> str:
    parts: list[str] = []
    if personalities:
        parts.append("## Personality Notes")
        for user in sorted(personalities):
            parts.append(f"\n### {user}\n{personalities[user].strip()}")
    if references:
        if parts:
            parts.append("")
        parts.append("## Group References")
        parts.append(references.strip())
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
    parser = argparse.ArgumentParser(description="Analyze chat log for personality notes and group references.")
    parser.add_argument("input", type=Path, help="Input chat log file")
    parser.add_argument("-o", "--output", type=Path, metavar="FILE", help="Also write output to FILE")
    parser.add_argument("-m", "--model", default="llama3.1:8b", metavar="NAME",
                        help="Ollama model name (default: llama3.1:8b)")
    parser.add_argument("-f", "--map-file", type=Path, default=Path("usermap"), metavar="FILE",
                        help="Usermap file (default: usermap)")
    parser.add_argument("-s", "--sample", type=int, default=150, metavar="N",
                        help="Max messages sampled per user for personality (default: 150)")
    parser.add_argument("--references-only", action="store_true", help="Skip personality analysis")
    parser.add_argument("--personality-only", action="store_true", help="Skip group references analysis")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print progress to stderr")
    args = parser.parse_args()
    atexit.register(unload_model, args.model)

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
    if not args.references_only:
        if args.verbose:
            print("\nAnalyzing personalities...", file=sys.stderr)
        for user in seen_users:
            result = analyze_personality(args.model, user, messages, args.sample, args.verbose)
            if result:
                personalities[user] = result

    references = ""
    if not args.personality_only:
        if args.verbose:
            print("\nAnalyzing group references...", file=sys.stderr)
        try:
            references = analyze_references(args.model, messages, seen_users, args.verbose)
        except urllib.error.URLError as e:
            if "Connection refused" in str(e):
                print("Error: cannot connect to Ollama at localhost:11434. Is it running?",
                      file=sys.stderr)
                sys.exit(1)
            raise

    output = format_output(personalities, references)

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
