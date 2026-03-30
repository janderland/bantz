import re
import sys
import json
import itertools
import textwrap
import argparse
import urllib.request

REACTION = re.compile(r'(\( \w+: \S+(?:, \w+: \S+)* \))')
SPECIAL_TOKEN = re.compile(r'<\|[^|]*\|>')


def make_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--width", type=int, default=None)
    parser.add_argument("-r", "--raw", metavar="FILE", default=None)
    parser.add_argument("query")
    return parser


def from_ollama(prompt):
    """Stream raw tokens from the Ollama API."""
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"model": "bantz", "prompt": prompt, "stream": True}).encode(),
    )
    with urllib.request.urlopen(req) as resp:
        for raw_line in resp:
            obj = json.loads(raw_line)
            yield obj["response"]
            if obj.get("done"):
                break


def tee_raw_output(tokens, file):
    """Pass raw tokens through while writing them to a file."""
    for token in tokens:
        file.write(token)
        file.flush()
        yield token


def tokenize_lines(tokens):
    """Accumulate tokens, strip special tokens, yield complete lines."""
    buf = ""
    for token in tokens:
        buf += token
        buf = SPECIAL_TOKEN.sub("", buf)
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            yield line.strip()
    if buf.strip():
        yield buf.strip()


def prefix_with_query(prompt, lines):
    """Prepend the prompt's own lines before the token stream lines."""
    return itertools.chain(
        (line.strip() for line in prompt.split("\n")),
        lines,
    )


def format_chat(lines):
    """Parse chat lines and yield output events.

    Yields str for text lines and None for blank lines.
    """
    output_queue = []

    current_name = None
    speaker_confirmed = False
    pending_header = None
    pre_msg_buf = []
    para_buf = []
    pending_line = None
    prev_was_quote = False

    def enqueue(item):
        output_queue.append(item)

    def confirm_speaker():
        nonlocal speaker_confirmed, pending_header
        if speaker_confirmed:
            return
        if pending_header is not None:
            enqueue(pending_header)
            pending_header = None
        for item in pre_msg_buf:
            enqueue(item)
        pre_msg_buf.clear()
        speaker_confirmed = True

    def flush_pending(next_is_quote=False, next_is_reaction=False):
        nonlocal pending_line
        if pending_line is None:
            return
        spacing = not next_is_reaction and (prev_was_quote != next_is_quote)
        if speaker_confirmed:
            enqueue(pending_line)
            if spacing:
                enqueue(None)
        else:
            pre_msg_buf.append(pending_line)
            if spacing:
                pre_msg_buf.append(None)
        pending_line = None

    def emit_para():
        nonlocal para_buf, pending_line, prev_was_quote
        if not para_buf:
            return
        merged = re.sub(r'\s+', ' ', ' '.join(para_buf)).strip()
        para_buf.clear()
        if not merged:
            return
        is_quote = merged.startswith(">")
        if not is_quote:
            confirm_speaker()
        flush_pending(next_is_quote=is_quote)
        pending_line = merged
        prev_was_quote = is_quote

    def process_line(line):
        nonlocal current_name, speaker_confirmed, pending_header, pending_line, prev_was_quote

        line = line.strip()

        # Reaction detection: split on reaction tokens and re-process pieces.
        parts = REACTION.split(line)
        if len(parts) > 1:
            for part in parts:
                if REACTION.fullmatch(part):
                    emit_para()
                    flush_pending(next_is_reaction=True)
                    if speaker_confirmed:
                        enqueue(part)
                    else:
                        pre_msg_buf.append(part)
                    prev_was_quote = False
                elif part:
                    process_line(part)
            return

        # Speaker line: alphabetic prefix followed by ": ".
        if ": " in line:
            prefix, rest = line.split(": ", 1)
            if prefix.isalpha():
                emit_para()
                if speaker_confirmed:
                    enqueue(None)
                else:
                    pending_line = None
                    para_buf.clear()
                    pre_msg_buf.clear()
                current_name = prefix
                pending_header = f"# {prefix}"
                speaker_confirmed = False
                prev_was_quote = rest.startswith(">") if rest else False
                if rest:
                    para_buf.append(rest)
                return

        if current_name is None:
            return

        # Blank / paragraph-break line.
        if line == "" or line == ">":
            emit_para()
            return

        # Continuation line.
        new_is_quote = line.startswith(">")
        if para_buf and para_buf[0].startswith(">") != new_is_quote:
            emit_para()
        para_buf.append(line)

    for line in lines:
        process_line(line)
        yield from output_queue
        output_queue.clear()

    # Finalize.
    emit_para()
    flush_pending()
    yield from output_queue
    if speaker_confirmed:
        yield None


def wrap_output(events, width):
    """Word-wrap text lines to the given column width."""
    for event in events:
        if event is None:
            yield None
        elif event.startswith("> "):
            yield textwrap.fill(event, width, subsequent_indent="> ")
        else:
            yield textwrap.fill(event, width)


def print_output(events):
    """Print events to stdout."""
    for event in events:
        print("" if event is None else event)
        sys.stdout.flush()


def main():
    args = make_arg_parser().parse_args()

    raw_file = open(args.raw, "w") if args.raw else None

    pipeline = from_ollama(args.query)
    if raw_file: pipeline = tee_raw_output(pipeline, raw_file)
    pipeline = tokenize_lines(pipeline)
    pipeline = prefix_with_query(args.query, pipeline)
    pipeline = format_chat(pipeline)
    if args.width: pipeline = wrap_output(pipeline, args.width)
    print_output(pipeline)

    if raw_file: raw_file.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        sys.exit(0)
