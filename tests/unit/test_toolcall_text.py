"""Tool calls printed as text (M5): detection and stripping (pan.memory.toolcall_text)."""

from __future__ import annotations

import pytest

from pan.memory.toolcall_text import looks_like_tool_call_text, strip_tool_call_text

# Final answers of the shape a local model printed (Qwen3.8-27B-NVFP4, vLLM qwen3_coder parser, thinking off).
PRINTED_REPLACE_CALL = 'memory replace: content="vLLM server runs on port 8010.", old_text="port 8000", target=memory'
PRINTED_LABEL = "Memory (user):"
# The qwen3_coder wire format when the parser does not pick the call up (e.g. inside content).
QWEN_XML = ("<tool_call>\n<function=memory>\n<parameter=action>\nadd\n</parameter>\n<parameter=target>\nmemory\n"
            "</parameter>\n<parameter=content>\nvLLM server runs on port 8010.\n</parameter>\n</function>\n</tool_call>")
HERMES_JSON = '<tool_call>\n{"name": "memory_search", "arguments": {"query": "vllm port"}}\n</tool_call>'
BARE_JSON = '{"name": "terminal", "arguments": {"command": "nvidia-smi --query-gpu=name --format=csv,noheader"}}'
OPENAI_JSON = '{"tool_calls": [{"id": "c1", "type": "function", "function": {"name": "memory", "arguments": "{}"}}]}'
FENCED = 'Let me check.\n```json\n{"tool": "memory_search", "query": "langfuse port"}\n```'
CALL_SYNTAX = 'memory_search(query="vllm port", limit=3)'
FUNCTIONS_PREFIX = 'functions.memory_read(page="operations.vllm-server")'
MEMORY_SHAPE = '{"action": "add", "target": "memory", "content": "Grafana listens on 3100."}'
UNCLOSED = "Saving it now.\n<tool_call>\n<function=memory>\n<parameter=action>\nadd"
SPECIAL_TOKENS = "<|tool_call_begin|>functions.terminal:0<|tool_call_argument_begin|>{\"command\": \"ls\"}<|tool_call_end|>"


@pytest.mark.parametrize("text", [PRINTED_REPLACE_CALL, PRINTED_LABEL, QWEN_XML, HERMES_JSON, BARE_JSON, OPENAI_JSON,
                                  CALL_SYNTAX, FUNCTIONS_PREFIX, MEMORY_SHAPE])
def test_whole_answer_is_a_printed_call(text):
    cleaned, found = strip_tool_call_text(text)
    assert found and cleaned == ""


@pytest.mark.parametrize("text,keep", [
    (FENCED, "Let me check."),
    (UNCLOSED, "Saving it now."),
    ("Sure — saving that.\n" + HERMES_JSON + "\nThe vLLM server now listens on port 8010.",
     "Sure — saving that.\n\nThe vLLM server now listens on port 8010."),
])
def test_prose_around_a_printed_call_is_kept(text, keep):
    cleaned, found = strip_tool_call_text(text)
    assert found and cleaned == keep


def test_special_tokens_are_dropped():
    cleaned, found = strip_tool_call_text(SPECIAL_TOKENS)
    assert found and cleaned == ""


@pytest.mark.parametrize("text", [
    "The vLLM server runs on port 8000.",
    "Memory: 128 GB unified, shared by CPU and GPU.",       # "memory" as a noun, no call arguments
    "Use the `terminal` tool to run `nvidia-smi`.",
    "Noted: I'll answer in English and keep every reply under 60 words.",
    'The config sets `max_model_len: 32768` and `served_model_name: "primary"`.',
    '{"output": "NVIDIA GB10", "exit_code": 0}',            # a tool *result* quoted by the model
    "I'll open the vLLM page first.",                        # an announced call (no call syntax)
])
def test_ordinary_answers_are_untouched(text):
    assert strip_tool_call_text(text) == (text, False)
    assert not looks_like_tool_call_text(text)
