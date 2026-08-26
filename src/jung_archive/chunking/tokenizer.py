"""
Deterministic, local token counting.

Primary strategy: tiktoken's cl100k_base BPE (deterministic, fully local,
no network calls). Fallback when tiktoken is unavailable: a documented
regex word counter. Whichever is active is used consistently for chunk
generation AND validation; `active_counter_name()` reports which one.
"""
import re
from typing import List

_FALLBACK_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

_encoder = None
_counter_name = "unknown"


def _get_encoder():
    global _encoder, _counter_name
    if _encoder is None:
        try:
            import tiktoken

            _encoder = tiktoken.get_encoding("cl100k_base")
            _counter_name = "tiktoken_cl100k_base"
        except Exception:
            _encoder = False  # sentinel: fallback mode
            _counter_name = "fallback_regex_words"
    return _encoder


def active_counter_name() -> str:
    _get_encoder()
    return _counter_name


def count_tokens(text: str) -> int:
    enc = _get_encoder()
    if enc:  # tiktoken encoder
        return len(enc.encode(text))
    return len(_FALLBACK_RE.findall(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Prefix of text containing at most max_tokens tokens."""
    if max_tokens <= 0:
        return ""
    enc = _get_encoder()
    if enc:
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return enc.decode(tokens[:max_tokens])
    words = _FALLBACK_RE.findall(text)
    return "".join(_rejoin(words)[:max_tokens])


def _rejoin(word_tokens: List[str]) -> List[str]:
    # Regex fallback splits words; approximate original spacing by
    # appending a space after alphanumeric/word tokens only.
    out = []
    for w in word_tokens:
        out.append(w if re.match(r"^[^\w\s]+$", w) else w + " ")
    return out


def split_text_into_token_windows(
    text: str, window_tokens: int, overlap_tokens: int
) -> List[str]:
    """Split text into token windows with optional overlap (deterministic)."""
    enc = _get_encoder()
    if enc:
        tokens = enc.encode(text)
        step = max(1, window_tokens - overlap_tokens)
        windows = []
        for start in range(0, len(tokens), step):
            chunk_tokens = tokens[start:start + window_tokens]
            windows.append(enc.decode(chunk_tokens))
            if start + window_tokens >= len(tokens):
                break
        return windows
    words = _FALLBACK_RE.findall(text)
    rejoined = _rejoin(words)
    step = max(1, window_tokens - overlap_tokens)
    windows = []
    for start in range(0, len(rejoined), step):
        windows.append("".join(rejoined[start:start + window_tokens]))
        if start + window_tokens >= len(rejoined):
            break
    return windows