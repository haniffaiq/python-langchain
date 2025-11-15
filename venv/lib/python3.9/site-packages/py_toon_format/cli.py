"""
Command-line interface for py-toon-format
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

from . import encode, decode


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Token-Oriented Object Notation (TOON) converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert JSON to TOON
  py-toon encode input.json
  py-toon encode input.json -o output.toon
  
  # Convert TOON to JSON
  py-toon decode input.toon
  py-toon decode input.toon -o output.json
  
  # Read from stdin
  echo '{"key": "value"}' | py-toon encode
  cat data.toon | py-toon decode
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Encode command
    encode_parser = subparsers.add_parser("encode", help="Convert JSON to TOON")
    encode_parser.add_argument("input", nargs="?", help="Input JSON file (or stdin if not provided)")
    encode_parser.add_argument("-o", "--output", help="Output TOON file (or stdout if not provided)")
    encode_parser.add_argument("-i", "--indent", type=int, default=2, help="Indentation level (default: 2)")
    encode_parser.add_argument("-d", "--delimiter", default=",", help="Field delimiter (default: ',')")
    
    # Decode command
    decode_parser = subparsers.add_parser("decode", help="Convert TOON to JSON")
    decode_parser.add_argument("input", nargs="?", help="Input TOON file (or stdin if not provided)")
    decode_parser.add_argument("-o", "--output", help="Output JSON file (or stdout if not provided)")
    decode_parser.add_argument("-i", "--indent", type=int, default=2, help="Indentation level (default: 2)")
    decode_parser.add_argument("--no-strict", action="store_true", help="Disable strict validation")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == "encode":
            _encode_command(args)
        elif args.command == "decode":
            _decode_command(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _encode_command(args):
    """Handle encode command"""
    # Read input
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: File not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        # Read from stdin
        data = json.load(sys.stdin)
    
    # Convert to TOON
    toon_output = encode(data, indent=args.indent, delimiter=args.delimiter)
    
    # Write output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(toon_output)
    else:
        print(toon_output)


def _decode_command(args):
    """Handle decode command"""
    # Read input
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: File not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        with open(input_path, "r", encoding="utf-8") as f:
            toon_input = f.read()
    else:
        # Read from stdin
        toon_input = sys.stdin.read()
    
    # Convert to JSON
    data = decode(toon_input, indent=args.indent, strict=not args.no_strict)
    
    # Write output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        json.dump(data, sys.stdout, indent=2, ensure_ascii=False)
        print()  # Newline after JSON


if __name__ == "__main__":
    main()

