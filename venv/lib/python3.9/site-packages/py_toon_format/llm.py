"""
LLM integration helpers for TOON format
"""

from typing import Any, Dict, List, Optional
import json

from .encoder import encode
from .decoder import decode


def prepare_for_llm(
    data: Any,
    *,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    model: str = "gpt-4",
    delimiter: str = ",",
) -> Dict[str, Any]:
    """
    Prepare TOON data for LLM API calls (OpenAI, Anthropic, etc.).
    
    Args:
        data: Python object to encode
        system_prompt: Optional system prompt
        user_prompt: Optional user prompt
        model: Model name (for token counting)
        delimiter: Field delimiter (default: ",")
    
    Returns:
        Dictionary with 'messages' or 'prompt' ready for LLM API
    
    Example:
        >>> data = {"products": [{"id": 1, "name": "Widget"}]}
        >>> payload = prepare_for_llm(data, user_prompt="Analyze this data")
        >>> # Use with OpenAI API
        >>> # response = openai.ChatCompletion.create(**payload)
    """
    toon_str = encode(data, delimiter=delimiter)
    
    # Format as code block for better LLM understanding
    formatted_toon = f"```toon\n{toon_str}\n```"
    
    if user_prompt:
        full_prompt = f"{user_prompt}\n\n{formatted_toon}"
    else:
        full_prompt = formatted_toon
    
    # Return format compatible with common LLM APIs
    if system_prompt:
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt},
            ]
        }
    else:
        return {"prompt": full_prompt}


def extract_from_llm_response(response: Any, *, model: str = "gpt-4") -> Any:
    """
    Extract TOON data from LLM response.
    
    Args:
        response: LLM API response object
        model: Model name (for response parsing)
    
    Returns:
        Decoded Python object from TOON in response
    
    Example:
        >>> # OpenAI response
        >>> content = response.choices[0].message.content
        >>> data = extract_from_llm_response(content)
    """
    # Handle different response formats
    if hasattr(response, "choices"):
        # OpenAI format
        text = response.choices[0].message.content
    elif hasattr(response, "content"):
        # Anthropic format
        text = response.content[0].text if isinstance(response.content, list) else response.content
    elif isinstance(response, str):
        text = response
    else:
        raise ValueError("Unsupported response format")
    
    # Extract TOON code block if present
    toon_match = None
    if "```toon" in text:
        import re
        match = re.search(r"```toon\n(.*?)\n```", text, re.DOTALL)
        if match:
            toon_match = match.group(1)
    elif "```" in text:
        # Try generic code block
        import re
        match = re.search(r"```\n(.*?)\n```", text, re.DOTALL)
        if match:
            toon_match = match.group(1)
    else:
        # Assume entire text is TOON
        toon_match = text.strip()
    
    if toon_match:
        return decode(toon_match)
    else:
        raise ValueError("No TOON format found in response")


def create_llm_prompt(
    data: Any,
    task: str,
    *,
    format_instruction: bool = True,
    delimiter: str = ",",
) -> str:
    """
    Create a complete LLM prompt with TOON data and instructions.
    
    Args:
        data: Python object to encode
        task: Task description for the LLM
        format_instruction: Include format instructions (default: True)
        delimiter: Field delimiter (default: ",")
    
    Returns:
        Complete prompt string ready for LLM
    
    Example:
        >>> data = {"users": [{"id": 1, "name": "Alice"}]}
        >>> prompt = create_llm_prompt(
        ...     data,
        ...     "Return only users with id > 0 as TOON format"
        ... )
        >>> print(prompt)
    """
    toon_str = encode(data, delimiter=delimiter)
    
    instruction = ""
    if format_instruction:
        instruction = """
Data is in TOON format (2-space indent, arrays show length and fields).

Rules:
- Use 2-space indentation
- Array headers show count: key[N]{field1,field2}:
- Set [N] to match actual row count
- Output only the code block
"""
    
    prompt = f"""{task}

{instruction}
```toon
{toon_str}
```

Task: {task}
"""
    
    return prompt.strip()

