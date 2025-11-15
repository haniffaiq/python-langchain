"""
Token-Oriented Object Notation (TOON) - Python Implementation

TOON is a compact, human-readable, schema-aware JSON format designed
for LLM prompts. It reduces token usage by 30-60% compared to JSON.
"""

from .encoder import encode
from .decoder import decode
from .io import load, loads, dump, dumps
from .utils import count_tokens, compare_sizes, validate, format_toon
from .llm import prepare_for_llm, extract_from_llm_response, create_llm_prompt

__version__ = "0.1.0"
__all__ = [
    "encode",
    "decode",
    "load",
    "loads",
    "dump",
    "dumps",
    "count_tokens",
    "compare_sizes",
    "validate",
    "format_toon",
    "prepare_for_llm",
    "extract_from_llm_response",
    "create_llm_prompt",
]

