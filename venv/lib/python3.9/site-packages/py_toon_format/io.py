"""
File I/O utilities for TOON format
Similar to json.load/dump API
"""

from pathlib import Path
from typing import Any, Union, TextIO
import json

from .encoder import encode
from .decoder import decode


def load(fp: Union[str, Path, TextIO], *, indent: int = 2, strict: bool = True) -> Any:
    """
    Load TOON data from a file-like object or file path.
    
    Args:
        fp: File path (str/Path) or file-like object
        indent: Expected indentation level (default: 2)
        strict: Enable strict validation (default: True)
    
    Returns:
        Python object decoded from TOON
    
    Example:
        >>> data = py_toon_format.load("data.toon")
        >>> data = py_toon_format.load(open("data.toon"))
    """
    if isinstance(fp, (str, Path)):
        with open(fp, "r", encoding="utf-8") as f:
            return decode(f.read(), indent=indent, strict=strict)
    else:
        return decode(fp.read(), indent=indent, strict=strict)


def loads(s: str, *, indent: int = 2, strict: bool = True) -> Any:
    """
    Load TOON data from a string.
    
    Args:
        s: TOON-formatted string
        indent: Expected indentation level (default: 2)
        strict: Enable strict validation (default: True)
    
    Returns:
        Python object decoded from TOON
    
    Example:
        >>> toon_str = "id: 1\\nname: Alice"
        >>> data = py_toon_format.loads(toon_str)
    """
    return decode(s, indent=indent, strict=strict)


def dump(obj: Any, fp: Union[str, Path, TextIO], *, indent: int = 2, delimiter: str = ",") -> None:
    """
    Dump Python object to TOON format in a file.
    
    Args:
        obj: Python object to encode
        fp: File path (str/Path) or file-like object
        indent: Indentation level (default: 2)
        delimiter: Field delimiter for tabular arrays (default: ",")
    
    Example:
        >>> data = {"key": "value"}
        >>> py_toon_format.dump(data, "output.toon")
        >>> py_toon_format.dump(data, open("output.toon", "w"))
    """
    toon_str = encode(obj, indent=indent, delimiter=delimiter)
    
    if isinstance(fp, (str, Path)):
        path = Path(fp)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(toon_str)
    else:
        fp.write(toon_str)


def dumps(obj: Any, *, indent: int = 2, delimiter: str = ",") -> str:
    """
    Dump Python object to TOON format string.
    
    Args:
        obj: Python object to encode
        indent: Indentation level (default: 2)
        delimiter: Field delimiter for tabular arrays (default: ",")
    
    Returns:
        TOON-formatted string
    
    Example:
        >>> data = {"key": "value"}
        >>> toon_str = py_toon_format.dumps(data)
    """
    return encode(obj, indent=indent, delimiter=delimiter)

