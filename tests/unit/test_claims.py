"""Claim overlap check used for IGNORE / L1 dedupe (pan.memory.claims)."""

from __future__ import annotations

from pan.memory.claims import claim_present, sentences, shorten, stem, tokens

PAGE = """# vLLM serves without tool calling

vLLM on the DGX Spark serves `RedHatAI/Qwen3.6-35B-A3B-NVFP4` as `primary` on port 8000, but was
started without `--enable-auto-tool-choice --tool-call-parser`, so tool calls fail. Hermes needs
tool calling. (observed)

## Services

| Service | Port |
|---|---|
| Langfuse | 3000 |

- A bullet that wraps
  onto a second line.
"""


def test_sentences_join_wrapped_lines_and_break_at_markdown():
    s = sentences(PAGE)
    assert s[0] == "vLLM serves without tool calling"
    assert s[1].startswith("vLLM on the DGX Spark serves") and s[1].endswith("so tool calls fail.")
    assert "Hermes needs tool calling. (observed)" in s
    assert "A bullet that wraps onto a second line." in s
    assert sentences("yes\nalways use uv", join_lines=False) == ["yes", "always use uv"]


def test_stem_and_tokens():
    assert stem("enabled") == stem("enable") == "enabl"
    assert stem("serves") == stem("served") == "serv"
    assert stem("8000") == "8000"
    assert tokens("The server is running") == ["server", "runn"]


def test_rephrased_claim_is_present():
    assert claim_present("vLLM serves RedHatAI/Qwen3.6-35B-A3B-NVFP4 as 'primary' on port 8000 without "
                         "tool calling enabled.", PAGE)
    assert claim_present("The vLLM server was started without --enable-auto-tool-choice and "
                         "--tool-call-parser, so tool calling is disabled.", PAGE)


def test_changed_number_is_new_information():
    assert not claim_present("vLLM serves primary on port 8001 without tool calling.", PAGE)
    assert claim_present("vLLM serves primary on port 8000 without tool calling.", PAGE)


def test_unrelated_or_extended_claims_are_not_present():
    assert not claim_present("vLLM serve --max-model-len 32768 --reasoning-parser qwen3", PAGE)
    assert not claim_present("Langfuse stores traces in ClickHouse.", PAGE)


def test_bag_of_words_across_the_page_does_not_count():
    # all words occur somewhere on the page, but not within one window of sentences
    assert not claim_present("Langfuse bullet wraps tool calling parser.", PAGE + "\n" + "x. " * 10)


def test_exact_substring_and_empty_claim():
    assert claim_present("so tool calls fail", PAGE)
    assert claim_present("the and of", PAGE)  # no content tokens → nothing to add


def test_shorten():
    assert shorten("Run curation in a separate service; the plugin writes events.", 36) == \
        "Run curation in a separate service"
    assert shorten("Where should curation run?", 100, title=False) == "Where should curation run?"
