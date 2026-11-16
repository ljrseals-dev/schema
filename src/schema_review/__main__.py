import argparse
import json
import sys

from .validate import load_schema, validate_review


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="schema_review", description="Validate a review JSON file.")
    p.add_argument("review", help="path to a review JSON file")
    p.add_argument("--schema", help="path to a schema file (default: bundled)")
    args = p.parse_args(argv)
    with open(args.review, encoding="utf-8") as fh:
        doc = json.load(fh)
    errs, warns = validate_review(doc, load_schema(args.schema))
    for w in warns:
        print(f"warning: {w}", file=sys.stderr)
    for e in errs:
        print(f"error: {e}", file=sys.stderr)
    print("OK" if not errs else f"FAILED ({len(errs)} errors)")
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
