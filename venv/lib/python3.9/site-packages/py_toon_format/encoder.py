"""
TOON Encoder - Converts Python objects to TOON format
"""

from typing import Any, Dict, List, Optional, Union
import json


def encode(
    data: Any,
    *,
    indent: int = 2,
    delimiter: str = ",",
    key_folding: str = "safe",
) -> str:
    """
    Encode Python objects to TOON format.
    
    Args:
        data: Python object (dict, list, or primitive)
        indent: Number of spaces per indentation level (default: 2)
        delimiter: Field delimiter for tabular arrays (default: ",")
        key_folding: Key folding strategy - "safe" or "off" (default: "safe")
    
    Returns:
        TOON-formatted string
    """
    return _encode_value(data, indent=indent, delimiter=delimiter, level=0)


def _encode_value(
    value: Any,
    *,
    indent: int,
    delimiter: str,
    level: int = 0,
) -> str:
    """Internal encoder for values"""
    if value is None:
        return "null"
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, str):
        return _encode_string(value)
    elif isinstance(value, dict):
        return _encode_object(value, indent=indent, delimiter=delimiter, level=level)
    elif isinstance(value, list):
        return _encode_array(value, indent=indent, delimiter=delimiter, level=level)
    else:
        # Fallback to JSON string representation
        return json.dumps(value)


def _encode_string(s: str) -> str:
    """Encode string with proper quoting"""
    # Check if quoting is needed
    needs_quotes = (
        s == "" or
        s[0].isspace() or
        s[-1].isspace() or
        "," in s or
        ":" in s or
        "[" in s or
        "]" in s or
        "{" in s or
        "}" in s or
        s.lower() in ("true", "false", "null") or
        _looks_like_number(s)
    )
    
    if not needs_quotes:
        return s
    
    # Escape quotes and backslashes
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _looks_like_number(s: str) -> bool:
    """Check if string looks like a number"""
    try:
        float(s)
        return True
    except ValueError:
        return False


def _encode_object(
    obj: Dict[str, Any],
    *,
    indent: int,
    delimiter: str,
    level: int,
) -> str:
    """Encode dictionary object"""
    if not obj:
        return ""
    
    lines = []
    prefix = " " * (indent * level)
    
    for key, value in obj.items():
        if isinstance(value, dict) and value:
            # Nested object
            lines.append(f"{prefix}{key}:")
            nested = _encode_object(value, indent=indent, delimiter=delimiter, level=level + 1)
            if nested:
                lines.append(nested)
        elif isinstance(value, list) and value:
            # Array
            array_str = _encode_array(value, indent=indent, delimiter=delimiter, level=level)
            if array_str.startswith("["):
                # Primitive or list format
                lines.append(f"{prefix}{key}{array_str}")
            else:
                # Tabular format (already has proper indentation)
                lines.append(f"{prefix}{key}{array_str}")
        else:
            # Primitive value
            value_str = _encode_value(value, indent=indent, delimiter=delimiter, level=level)
            lines.append(f"{prefix}{key}: {value_str}")
    
    return "\n".join(lines)


def _encode_array(
    arr: List[Any],
    *,
    indent: int,
    delimiter: str,
    level: int,
) -> str:
    """Encode array - chooses between tabular and list format"""
    if not arr:
        return "[0]:"
    
    # Check if all elements are uniform objects (same keys)
    if _is_uniform_objects(arr):
        return _encode_tabular_array(arr, indent=indent, delimiter=delimiter, level=level)
    elif _is_primitive_array(arr):
        return _encode_primitive_array(arr)
    else:
        return _encode_list_array(arr, indent=indent, delimiter=delimiter, level=level)


def _is_uniform_objects(arr: List[Any]) -> bool:
    """Check if array contains uniform objects with same keys"""
    if not arr or not isinstance(arr[0], dict):
        return False
    
    first_keys = set(arr[0].keys())
    if not first_keys:
        return False
    
    # Check if all objects have same keys and all values are primitives
    for item in arr[1:]:
        if not isinstance(item, dict):
            return False
        if set(item.keys()) != first_keys:
            return False
        # Check if all values are primitives
        for v in item.values():
            if not isinstance(v, (str, int, float, bool, type(None))):
                return False
    
    return True


def _is_primitive_array(arr: List[Any]) -> bool:
    """Check if array contains only primitives"""
    return all(isinstance(x, (str, int, float, bool, type(None))) for x in arr)


def _encode_tabular_array(
    arr: List[Dict[str, Any]],
    *,
    indent: int,
    delimiter: str,
    level: int,
) -> str:
    """Encode uniform array of objects in tabular format"""
    if not arr:
        return "[0]:"
    
    # Get keys from first object
    keys = list(arr[0].keys())
    n = len(arr)
    
    prefix = " " * (indent * level)
    next_prefix = " " * (indent * (level + 1))
    
    # Header: key[N]{field1,field2,...}:
    keys_str = delimiter.join(keys)
    header = f"[{n}]{{{keys_str}}}:"
    
    lines = [f"{prefix}{header}"]
    
    # Rows
    for item in arr:
        values = []
        for key in keys:
            value = item.get(key)
            value_str = _encode_value(value, indent=indent, delimiter=delimiter, level=level)
            # Remove quotes if not needed for tabular format (except for strings with special chars)
            if isinstance(value, str) and not _needs_quotes_for_tabular(value, delimiter):
                values.append(value_str.strip('"'))
            else:
                values.append(value_str)
        
        row = delimiter.join(values)
        lines.append(f"{next_prefix}{row}")
    
    return "\n".join(lines)


def _needs_quotes_for_tabular(s: str, delimiter: str) -> bool:
    """Check if string needs quotes in tabular format"""
    return (
        delimiter in s or
        ":" in s or
        "\n" in s or
        s.strip() != s or
        s == "" or
        s.lower() in ("true", "false", "null")
    )


def _encode_primitive_array(arr: List[Any]) -> str:
    """Encode array of primitives"""
    n = len(arr)
    values = []
    for item in arr:
        value_str = _encode_value(item, indent=2, delimiter=",", level=0)
        values.append(value_str)
    
    values_str = ",".join(values)
    return f"[{n}]: {values_str}"


def _encode_list_array(
    arr: List[Any],
    *,
    indent: int,
    delimiter: str,
    level: int,
) -> str:
    """Encode mixed/non-uniform array in list format"""
    n = len(arr)
    prefix = " " * (indent * level)
    next_prefix = " " * (indent * (level + 1))
    
    lines = [f"{prefix}[{n}]:"]
    
    for item in arr:
        if isinstance(item, dict):
            # Object in list
            obj_str = _encode_object(item, indent=indent, delimiter=delimiter, level=level + 1)
            if obj_str:
                # Add dash prefix to first line
                first_line, rest = obj_str.split("\n", 1) if "\n" in obj_str else (obj_str, "")
                lines.append(f"{next_prefix}- {first_line}")
                if rest:
                    lines.append(rest)
        elif isinstance(item, list):
            # Nested array
            arr_str = _encode_array(item, indent=indent, delimiter=delimiter, level=level + 1)
            lines.append(f"{next_prefix}- {arr_str}")
        else:
            # Primitive
            value_str = _encode_value(item, indent=indent, delimiter=delimiter, level=level)
            lines.append(f"{next_prefix}- {value_str}")
    
    return "\n".join(lines)

