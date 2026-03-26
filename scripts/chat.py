import re
import sys
import json
import textwrap
import argparse
import urllib.request


def stream_ollama(prompt):
    data = json.dumps({"model": "bantz", "prompt": prompt, "stream": True}).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        headers={"Content-Type": "application/json"},
        data=data,
    )
    with urllib.request.urlopen(req) as resp:
        for raw_line in resp:
            obj = json.loads(raw_line)
            yield obj["response"]
            if obj.get("done"):
                break


def stream_formatted(prompt, width=None, raw_file=None):
    text_buffer = ""
    current_name = None
    pending_line = None
    prev_was_quote = False
    para_buffer = []
    speaker_confirmed = False
    pending_header = None
    pre_message_lines = []

    def wrap_line(line):
        if width is None:
            return line
        if line.startswith("> "):
            return textwrap.fill(line, width, subsequent_indent="> ")
        return textwrap.fill(line, width)

    def confirm_speaker():
        nonlocal speaker_confirmed, pending_header, pre_message_lines
        if not speaker_confirmed:
            if pending_header is not None:
                print(pending_header)
                sys.stdout.flush()
                pending_header = None
            for buffered in pre_message_lines:
                print(buffered)
            pre_message_lines.clear()
            sys.stdout.flush()
            speaker_confirmed = True

    def emit_para():
        nonlocal para_buffer, pending_line, prev_was_quote
        if not para_buffer:
            return
        merged = re.sub(r'\s+', ' ', ' '.join(para_buffer)).strip()
        para_buffer.clear()
        if not merged:
            return
        is_quote = merged.startswith(">")
        if not is_quote:
            confirm_speaker()
        flush_pending(next_is_quote=is_quote)
        pending_line = merged
        prev_was_quote = is_quote

    def flush_pending(next_is_quote=False, next_is_reaction=False):
        nonlocal pending_line, prev_was_quote
        if pending_line is not None:
            line_str = wrap_line(pending_line)
            spacing = not next_is_reaction and prev_was_quote != next_is_quote
            if speaker_confirmed:
                print(line_str)
                if spacing:
                    print()
                sys.stdout.flush()
            else:
                pre_message_lines.append(line_str)
                if spacing:
                    pre_message_lines.append("")
            pending_line = None

    def process_line(line):
        nonlocal current_name, pending_line, prev_was_quote, speaker_confirmed, pending_header, pre_message_lines

        line = line.strip()
        parts = re.split(r'(\( \w+: \S+(?:, \w+: \S+)* \))', line)
        if len(parts) > 1:
            for part in parts:
                if re.fullmatch(r'\( \w+: \S+(?:, \w+: \S+)* \)', part):
                    emit_para()
                    flush_pending(next_is_reaction=True)
                    reaction_str = part.strip()
                    if speaker_confirmed:
                        print(reaction_str)
                        sys.stdout.flush()
                    else:
                        pre_message_lines.append(reaction_str)
                    prev_was_quote = False
                elif part:
                    process_line(part)
            return

        if ": " in line:
            prefix, rest = line.split(": ", 1)
            if prefix.isalpha():
                emit_para()
                if speaker_confirmed:
                    flush_pending()
                    print()
                    sys.stdout.flush()
                else:
                    pending_line = None
                    para_buffer.clear()
                    pre_message_lines.clear()
                current_name = prefix
                pending_header = f"# {prefix}"
                speaker_confirmed = False
                if rest:
                    para_buffer.append(rest)
                prev_was_quote = rest.startswith(">") if rest else False
                return

        if current_name is None:
            return

        stripped = line.strip()
        if stripped == "" or stripped == ">":
            emit_para()
        else:
            new_is_quote = stripped.startswith(">")
            if para_buffer and para_buffer[0].startswith(">") != new_is_quote:
                emit_para()
            para_buffer.append(stripped)

    for line in prompt.split("\n"):
        process_line(line)

    for token in stream_ollama(prompt):
        if raw_file is not None:
            raw_file.write(token)
            raw_file.flush()
        text_buffer += token
        text_buffer = re.sub(r'<\|[^|]*\|>', '', text_buffer)
        while "\n" in text_buffer:
            line, text_buffer = text_buffer.split("\n", 1)
            process_line(line)

    if text_buffer:
        process_line(text_buffer)

    emit_para()
    flush_pending()
    if speaker_confirmed:
        print()
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--width", type=int, default=None)
    parser.add_argument("-r", "--raw", metavar="FILE", default=None)
    parser.add_argument("query", nargs="+")
    args = parser.parse_args()

    prompt = " ".join(args.query)
    if args.raw:
        with open(args.raw, "w") as raw_file:
            stream_formatted(prompt, width=args.width, raw_file=raw_file)
    else:
        stream_formatted(prompt, width=args.width)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        sys.exit(0)
