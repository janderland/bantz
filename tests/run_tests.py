"""Test runner for bantz bot quality evaluation.

Usage:
    python tests/run_tests.py --model bantz-VERSION [--prompts FILE] [--output FILE] [--limit N]
"""

import re
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

# Allow importing from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import chat  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from detectors import ALL_DETECTORS  # noqa: E402

DEFAULT_PROMPTS = Path(__file__).parent / "prompts.txt"

SPEAKER_LINE = re.compile(r'^([A-Za-z]+):\s*(.*)')


def load_prompts(path: Path) -> list:
    prompts = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            prompts.append(line)
    return prompts


def query_model(prompt: str, model: str) -> str:
    """Send prompt to Ollama and return the full raw response."""
    tokens = chat.from_ollama(prompt, model)
    lines = list(chat.tokenize_lines(tokens))
    return "\n".join(lines)


def parse_messages(raw: str) -> list:
    """Parse raw model output into a list of speaker turns.

    Returns a list of dicts: [{"speaker": str, "lines": [str]}, ...]
    Lines within a turn are in order; quote lines start with "> ".
    """
    messages = []
    current = None

    for line in raw.splitlines():
        line = line.strip()
        m = SPEAKER_LINE.match(line)
        if m:
            if current is not None:
                messages.append(current)
            speaker = m.group(1)
            rest = m.group(2).strip()
            current = {"speaker": speaker, "lines": []}
            if rest:
                current["lines"].append(rest)
        elif current is not None and line:
            current["lines"].append(line)

    if current is not None:
        messages.append(current)

    return messages


def run_tests(model: str, prompts: list, verbose: bool = False) -> dict:
    """Run all detectors over each prompt and return aggregated results."""
    results = []
    issue_counts = defaultdict(int)
    prompts_with_issues = defaultdict(set)

    for i, prompt in enumerate(prompts):
        if verbose:
            print(f"[{i+1}/{len(prompts)}] {prompt[:60]}", file=sys.stderr)

        try:
            raw = query_model(prompt, model)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            results.append({"prompt": prompt, "error": str(e), "issues": []})
            continue

        messages = parse_messages(raw)
        all_issues = []
        for detector in ALL_DETECTORS:
            found = detector.analyze(prompt, messages)
            all_issues.extend(found)

        for issue in all_issues:
            issue_counts[issue.detector] += 1
            prompts_with_issues[issue.detector].add(i)

        results.append({
            "prompt": prompt,
            "raw": raw,
            "messages": messages,
            "issues": [
                {"detector": iss.detector, "speaker": iss.speaker, "context": iss.context}
                for iss in all_issues
            ],
        })

    return {
        "model": model,
        "total_prompts": len(prompts),
        "issue_counts": dict(issue_counts),
        "prompts_affected": {k: len(v) for k, v in prompts_with_issues.items()},
        "results": results,
    }


def print_summary(report: dict):
    total = report["total_prompts"]
    print(f"\nModel: {report['model']}")
    print(f"Prompts tested: {total}")

    detectors = sorted(set(list(report["issue_counts"]) + list(report["prompts_affected"])))
    if not detectors:
        print("\nNo issues detected.")
        return

    print(f"\n{'Detector':<30} {'Issues':>8} {'Prompts affected':>18} {'Rate':>8}")
    print("-" * 68)
    for name in detectors:
        count = report["issue_counts"].get(name, 0)
        affected = report["prompts_affected"].get(name, 0)
        rate = affected / total * 100 if total else 0
        print(f"{name:<30} {count:>8} {affected:>18} {rate:>7.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Run bantz bot quality tests.")
    parser.add_argument("--model", required=True, help="Ollama model name to test")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS,
                        help="Prompts file (one per line, # for comments)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write full JSON report to this file")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only test the first N prompts")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    if args.limit:
        prompts = prompts[:args.limit]

    if not prompts:
        print("No prompts found.", file=sys.stderr)
        sys.exit(1)

    report = run_tests(args.model, prompts, verbose=args.verbose)
    print_summary(report)

    if args.output:
        # Strip raw output from JSON to keep file size reasonable
        slim = {k: v for k, v in report.items() if k != "results"}
        slim["results"] = [
            {k: v for k, v in r.items() if k != "raw"}
            for r in report["results"]
        ]
        args.output.write_text(json.dumps(slim, indent=2))
        print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
