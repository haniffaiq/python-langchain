"""
TOON Decoder - Converts TOON format to Python objects
"""

from typing import Any, Dict, List, Optional, Tuple
import re


def decode(
    input_str: str,
    *,
    indent: int = 2,
    strict: bool = True,
    expand_paths: str = "off",
) -> Any:
    """
    Decode TOON-formatted string to Python objects.
    
    Args:
        input_str: TOON-formatted string
        indent: Expected number of spaces per indentation level (default: 2)
        strict: Enable strict validation (default: True)
        expand_paths: Enable path expansion - "off" or "safe" (default: "off")
    
    Returns:
        Python object (dict, list, or primitive)
    """
    lines = [line.rstrip() for line in input_str.strip().split("\n") if line.strip()]
    if not lines:
        return {}
    
    parser = _TOONParser(lines, indent=indent, strict=strict)
    return parser.parse()


class _TOONParser:
    """Internal TOON parser"""
    
    def __init__(self, lines: List[str], *, indent: int, strict: bool):
        self.lines = lines
        self.indent = indent
        self.strict = strict
        self.pos = 0
    
    def parse(self) -> Any:
        """Parse TOON lines into Python object"""
        if not self.lines:
            return {}
        
        # Check if root is an array
        first_line = self.lines[0].lstrip()
        if first_line.startswith("["):
            return self._parse_array(first_line, level=0)
        
        # Parse as object
        return self._parse_object(level=0)
    
    def _parse_object(self, level: int) -> Dict[str, Any]:
        """Parse object starting at current position"""
        obj = {}
        prefix = " " * (self.indent * level)
        
        while self.pos < len(self.lines):
            line = self.lines[self.pos]
            
            # Check indentation level
            if not line.startswith(prefix) and line != line.lstrip():
                # Wrong indentation, go back
                break
            
            stripped = line.lstrip()
            if not stripped:
                self.pos += 1
                continue
            
            # Check if we've gone too deep
            if line.startswith(prefix) and len(line) - len(line.lstrip()) < len(prefix):
                break
            
            # Parse key-value pair
            if ":" in stripped:
                # Check if this is an array header (key[N]{fields}: or key[N]: value)
                # Match: key[N]{fields}: or key[N]: value
                array_match = re.match(r'^([^:\[\]]+)(\[(\d+)\](?:\{([^}]+)\})?):\s*(.*)$', stripped)
                if array_match:
                    # Array header
                    key = array_match.group(1).strip()
                    array_part = array_match.group(2)  # [N] or [N]{fields}
                    n = int(array_match.group(3))
                    fields_str = array_match.group(4)  # fields part if exists
                    value_part = array_match.group(5) if len(array_match.groups()) > 4 else ""
                    
                    self.pos += 1
                    
                    # Parse array
                    if fields_str:
                        # Tabular format: key[N]{fields}:
                        delimiter = self._detect_delimiter(fields_str)
                        fields = [f.strip() for f in fields_str.split(delimiter)]
                        arr = self._parse_tabular_array(n, fields, delimiter, level)
                    elif value_part.strip():
                        # Inline primitive array: key[N]: value1,value2,value3
                        arr = self._parse_primitive_array(value_part)
                    else:
                        # Multi-line list format: key[N]:
                        arr = self._parse_list_array(n, level)
                    
                    obj[key] = arr
                    continue
                
                # Regular key-value pair
                key, value_part = stripped.split(":", 1)
                key = key.strip()
                value_part = value_part.strip()
                
                # Check for inline array
                if value_part.startswith("["):
                    # Inline array
                    arr = self._parse_array(value_part, level=level)
                    obj[key] = arr
                    self.pos += 1
                elif not value_part:
                    # Nested object or empty value
                    next_line_idx = self.pos + 1
                    if next_line_idx < len(self.lines):
                        next_line = self.lines[next_line_idx]
                        next_prefix = " " * (self.indent * (level + 1))
                        if next_line.startswith(next_prefix):
                            # Check if it's an array or nested object
                            next_stripped = next_line.lstrip()
                            if next_stripped.startswith("["):
                                # Array on next line
                                self.pos += 1
                                arr = self._parse_array(next_stripped, level=level)
                                obj[key] = arr
                                continue
                            else:
                                # Nested object
                                self.pos += 1
                                nested = self._parse_object(level=level + 1)
                                obj[key] = nested
                                continue
                    
                    obj[key] = None
                    self.pos += 1
                else:
                    # Primitive value
                    obj[key] = self._parse_primitive(value_part)
                    self.pos += 1
            else:
                # No colon, might be continuation or error
                self.pos += 1
        
        return obj
    
    def _parse_array(self, header: str, level: int) -> List[Any]:
        """Parse array from header line"""
        # Parse header: [N]{fields}: or [N]: value
        # Extract value part if present
        colon_idx = header.find(":")
        value_part = header[colon_idx + 1:].strip() if colon_idx != -1 else ""
        header_part = header[:colon_idx + 1] if colon_idx != -1 else header
        
        # Parse header: [N]{fields}: or [N]:
        match = re.match(r'\[(\d+)\](?:\{([^}]+)\})?:', header_part)
        if not match:
            if self.strict:
                raise ValueError(f"Invalid array header: {header_part}")
            return []
        
        n = int(match.group(1))
        fields_str = match.group(2) if match.group(2) else None
        
        if fields_str:
            # Tabular format
            delimiter = self._detect_delimiter(fields_str)
            fields = [f.strip() for f in fields_str.split(delimiter)]
            return self._parse_tabular_array(n, fields, delimiter, level)
        else:
            # Check if primitive array or list format
            if value_part:
                # Primitive array inline: [N]: value1,value2,value3
                return self._parse_primitive_array(value_part)
            
            # List format (multi-line)
            return self._parse_list_array(n, level)
    
    def _detect_delimiter(self, fields_str: str) -> str:
        """Detect delimiter from fields string"""
        # Check for tabs, pipes, or commas
        if "\t" in fields_str:
            return "\t"
        elif "|" in fields_str:
            return "|"
        else:
            return ","
    
    def _parse_tabular_array(
        self,
        n: int,
        fields: List[str],
        delimiter: str,
        level: int,
    ) -> List[Dict[str, Any]]:
        """Parse tabular array format"""
        arr = []
        prefix = " " * (self.indent * (level + 1))
        
        for i in range(n):
            if self.pos >= len(self.lines):
                if self.strict:
                    raise ValueError(f"Expected {n} rows, got {i}")
                break
            
            line = self.lines[self.pos]
            if not line.startswith(prefix):
                if self.strict:
                    raise ValueError(f"Unexpected indentation at row {i + 1}")
                break
            
            stripped = line.lstrip()
            values = self._split_row(stripped, delimiter)
            
            if len(values) != len(fields):
                if self.strict:
                    raise ValueError(
                        f"Row {i + 1}: expected {len(fields)} values, got {len(values)}"
                    )
            
            # Create object
            item = {}
            for j, field in enumerate(fields):
                if j < len(values):
                    item[field] = self._parse_primitive(values[j].strip())
                else:
                    item[field] = None
            
            arr.append(item)
            self.pos += 1
        
        return arr
    
    def _split_row(self, row: str, delimiter: str) -> List[str]:
        """Split row respecting quoted values"""
        parts = []
        current = ""
        in_quotes = False
        escape_next = False
        
        for char in row:
            if escape_next:
                current += char
                escape_next = False
            elif char == "\\":
                escape_next = True
                current += char
            elif char == '"':
                in_quotes = not in_quotes
                current += char
            elif char == delimiter and not in_quotes:
                parts.append(current)
                current = ""
            else:
                current += char
        
        if current:
            parts.append(current)
        
        return parts
    
    def _parse_primitive_array(self, value_str: str) -> List[Any]:
        """Parse inline primitive array"""
        values = [v.strip() for v in value_str.split(",")]
        return [self._parse_primitive(v) for v in values]
    
    def _parse_list_array(self, n: int, level: int) -> List[Any]:
        """Parse list format array"""
        arr = []
        prefix = " " * (self.indent * (level + 1))
        
        for i in range(n):
            if self.pos >= len(self.lines):
                if self.strict:
                    raise ValueError(f"Expected {n} items, got {i}")
                break
            
            line = self.lines[self.pos]
            if not line.startswith(prefix):
                if self.strict:
                    raise ValueError(f"Unexpected indentation at item {i + 1}")
                break
            
            stripped = line.lstrip()
            if not stripped.startswith("- "):
                if self.strict:
                    raise ValueError(f"Expected list item marker '- ' at item {i + 1}")
                break
            
            item_str = stripped[2:].strip()
            
            # Parse item
            if item_str.startswith("["):
                # Nested array
                item = self._parse_array(item_str, level=level + 1)
            elif ":" in item_str and not item_str.startswith('"'):
                # Object
                # Create a temporary line and parse as object
                temp_lines = [item_str]
                old_pos = self.pos
                self.pos = 0
                temp_parser = _TOONParser(temp_lines, indent=self.indent, strict=self.strict)
                item = temp_parser._parse_object(level=0)
                self.pos = old_pos
            else:
                # Primitive
                item = self._parse_primitive(item_str)
            
            arr.append(item)
            self.pos += 1
        
        return arr
    
    def _parse_primitive(self, value_str: str) -> Any:
        """Parse primitive value"""
        value_str = value_str.strip()
        
        if not value_str:
            return None
        
        # String with quotes
        if value_str.startswith('"') and value_str.endswith('"'):
            # Unescape
            inner = value_str[1:-1]
            return inner.replace('\\"', '"').replace('\\\\', '\\')
        
        # Boolean
        if value_str.lower() == "true":
            return True
        if value_str.lower() == "false":
            return False
        
        # Null
        if value_str.lower() == "null":
            return None
        
        # Number
        try:
            if "." in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            pass
        
        # String without quotes
        return value_str
    
    def _peek_next_is_array(self, level: int) -> bool:
        """Check if next line is an array"""
        if self.pos + 1 >= len(self.lines):
            return False
        
        next_line = self.lines[self.pos + 1].lstrip()
        return next_line.startswith("[")
    
    def _peek_next(self) -> Optional[str]:
        """Peek at next line"""
        if self.pos + 1 < len(self.lines):
            return self.lines[self.pos + 1]
        return None

