"""Command-line entry point for schema_review."""
from __future__ import annotations

import argparse
import sys

from .validate import load_json, load_schema, validate_review


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="schema_review",
        description="Validate a review JSON file.",
    )
    parser.add_argument("review", help="path to a review JSON file")
    parser.add_argument("--schema", help="path to a schema file (default: bundled)")
    parser.add_argument(
        "--structural-only",
        action="store_true",
        help="skip semantic rules; use with any schema other than the bundled one",
    )
    args = parser.parse_args(argv)

    try:
        document = load_json(args.review)
        schema = load_schema(args.schema)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"error: cannot read or parse input/schema: {exc}", file=sys.stderr)
        return 2

    errors, warnings = validate_review(
        document,
        schema,
        structural_only=args.structural_only,
    )
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    print("OK" if not errors else f"FAILED ({len(errors)} errors)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
