import argparse
import json
import sys

from .validate import load_schema, validate_review


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="schema_review", description="Validate a review JSON file.")
    p.add_argument("review", help="path to a review JSON file")
    p.add_argument("--schema", help="path to a schema file (default: bundled)")
    p.add_argument(
        "--structural-only",
        action="store_true",
        help="skip semantic rules; use with any schema other than the bundled one",
    )
    args = p.parse_args(argv)
    try:
        with open(args.review, encoding="utf-8") as fh:
            doc = json.load(fh)
        schema = load_schema(args.schema)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read input: {exc}", file=sys.stderr)
        return 2
    errs, warns = validate_review(doc, schema, structural_only=args.structural_only)
    for w in warns:
        print(f"warning: {w}", file=sys.stderr)
    for e in errs:
        print(f"error: {e}", file=sys.stderr)
    print("OK" if not errs else f"FAILED ({len(errs)} errors)")
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
