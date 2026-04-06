from dataclasses import dataclass


@dataclass
class Issue:
    detector: str   # e.g. "quote_only_message"
    speaker: str    # which speaker produced the problematic turn
    context: str    # relevant snippet for the report


class Detector:
    name: str

    def analyze(self, prompt: str, messages: list) -> list:
        """Analyze a parsed message list and return a list of Issue objects.

        Args:
            prompt: The original prompt string sent to the model.
            messages: List of dicts with keys "speaker" (str) and "lines" (list[str]).
                      Lines are the raw text lines produced by that speaker turn,
                      in order. Quote lines start with "> ".

        Returns:
            List of Issue objects describing detected problems.
        """
        raise NotImplementedError
