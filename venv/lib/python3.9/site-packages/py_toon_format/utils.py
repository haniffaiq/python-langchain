"""
Utility functions for TOON
"""

from typing import Any, Dict, Optional
import json
import re

from .encoder import encode
from .decoder import decode


def count_tokens(text: str, tokenizer: Optional[Any] = None) -> int:
    """
    Count tokens in a text string.
    
    Args:
        text: Text to count tokens for
        tokenizer: Optional tokenizer (tiktoken, transformers, etc.)
                  If None, uses a simple word-based approximation
    
    Returns:
        Approximate token count
    
    Example:
        >>> json_str = json.dumps({"key": "value"})
        >>> toon_str = encode({"key": "value"})
        >>> json_tokens = count_tokens(json_str)
        >>> toon_tokens = count_tokens(toon_str)
        >>> savings = (1 - toon_tokens / json_tokens) * 100
    """
    if tokenizer is not None:
        # Use provided tokenizer
        if hasattr(tokenizer, "encode"):
            return len(tokenizer.encode(text))
        elif callable(tokenizer):
            return len(tokenizer(text))
    
    # Simple approximation: ~4 characters per token
    # This is a rough estimate; actual tokenizers vary
    return len(text) // 4 + 1


def compare_sizes(data: Any, json_indent: int = 2) -> Dict[str, Any]:
    """
    Compare JSON and TOON representations of the same data.
    
    Args:
        data: Python object to compare
        json_indent: JSON indentation level for comparison
    
    Returns:
        Dictionary with comparison metrics:
        - json_size: JSON string length
        - toon_size: TOON string length
        - json_tokens: Approximate JSON token count
        - toon_tokens: Approximate TOON token count
        - size_reduction: Percentage reduction in size
        - token_reduction: Percentage reduction in tokens
    
    Example:
        >>> data = {"items": [{"id": 1}, {"id": 2}]}
        >>> metrics = compare_sizes(data)
        >>> print(f"Token reduction: {metrics['token_reduction']:.1f}%")
    """
    json_str = json.dumps(data, indent=json_indent, ensure_ascii=False)
    toon_str = encode(data)
    
    json_size = len(json_str)
    toon_size = len(toon_str)
    
    json_tokens = count_tokens(json_str)
    toon_tokens = count_tokens(toon_str)
    
    size_reduction = ((json_size - toon_size) / json_size * 100) if json_size > 0 else 0
    token_reduction = ((json_tokens - toon_tokens) / json_tokens * 100) if json_tokens > 0 else 0
    
    return {
        "json_size": json_size,
        "toon_size": toon_size,
        "json_tokens": json_tokens,
        "toon_tokens": toon_tokens,
        "size_reduction": round(size_reduction, 2),
        "token_reduction": round(token_reduction, 2),
    }


def validate(toon_str: str, *, indent: int = 2, strict: bool = True) -> tuple:
    """
    Validate TOON format string.
    
    Args:
        toon_str: TOON-formatted string to validate
        indent: Expected indentation level (default: 2)
        strict: Enable strict validation (default: True)
    
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if valid, False otherwise
        - error_message: Error message if invalid, None if valid
    
    Example:
        >>> is_valid, error = validate(toon_string)
        >>> if not is_valid:
        ...     print(f"Invalid TOON: {error}")
    """
    try:
        decode(toon_str, indent=indent, strict=strict)
        return True, None
    except Exception as e:
        return False, str(e)


def format_toon(toon_str: str, *, indent: int = 2) -> str:
    """
    Format/reformat TOON string with consistent indentation.
    
    Args:
        toon_str: TOON-formatted string
        indent: Desired indentation level (default: 2)
    
    Returns:
        Reformatted TOON string
    
    Example:
        >>> messy_toon = "id:1\\nname:Alice"
        >>> formatted = format_toon(messy_toon)
    """
    try:
        data = decode(toon_str)
        return encode(data, indent=indent)
    except Exception:
        # If decoding fails, return original
        return toon_str

