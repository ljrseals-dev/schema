#!/usr/bin/env python3
"""Differential, clean-room validation of the bundled review schema.

The reference oracle is jsonschema's independent Draft 7 implementation. The
project validator is exercised only through its public CLI in a copied package.
This is scoped to the bundled schema and fixture; semantic rules still need
their own independent oracle and review claims still require source verification.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

from jsonschema import Draft7Validator

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "schema_review"
SCHEMA_PATH = PACKAGE / "data" / "quantized_emitter_review.schema.json"
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "review_example.json"

WRONG_TYPE: dict[type, Any] = {
    dict: [],
    list: {},
    str: 123,
    bool: "not-a-boolean",
    int: "not-an-integer",
    float: "not-a-number",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_schema(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    """Resolve local JSON Pointer references for schema-guided mutation."""
    while "$ref" in schema:
        reference = schema["$ref"]
        if not reference.startswith("#/"):
            raise ValueError(f"unsupported schema reference: {reference}")
        target: Any = root
        for raw in reference[2:].split("/"):
            part = raw.replace("~1", "/").replace("~0", "~")
            target = target[part]
        schema = target
    return schema


def walk_instances(
    instance: Any,
    schema: dict[str, Any],
    root: dict[str, Any],
    path: tuple[str | int, ...] = (),
) -> Iterator[tuple[tuple[str | int, ...], Any, dict[str, Any]]]:
    schema = resolve_schema(schema, root)
    yield path, instance, schema
    if isinstance(instance, dict):
        for key, child_schema in schema.get("properties", {}).items():
            if key in instance:
                yield from walk_instances(instance[key], child_schema, root, path + (key,))
    elif isinstance(instance, list) and "items" in schema:
        for index, item in enumerate(instance):
            yield from walk_instances(item, schema["items"], root, path + (index,))


def get_at(document: Any, path: tuple[str | int, ...]) -> Any:
    value = document
    for component in path:
        value = value[component]
    return value


def set_at(document: Any, path: tuple[str | int, ...], value: Any) -> Any:
    if not path:
        return value
    get_at(document, path[:-1])[path[-1]] = value
    return document


def delete_at(document: Any, path: tuple[str | int, ...]) -> Any:
    del get_at(document, path[:-1])[path[-1]]
    return document


def describe(path: tuple[str | int, ...]) -> str:
    return "$" + "".join(f"[{part!r}]" for part in path)


def generate_mutants(
    fixture: dict[str, Any], schema: dict[str, Any]
) -> Iterator[tuple[str, bool, str, Any]]:
    """Yield category, expected-invalid, description, and mutated instance."""
    for path, value, node_schema in walk_instances(fixture, schema, schema):
        location = describe(path)

        if isinstance(value, dict):
            for key in node_schema.get("required", []):
                if key in value:
                    mutant = copy.deepcopy(fixture)
                    delete_at(mutant, path + (key,))
                    yield "delete-required", True, f"{location}: delete {key!r}", mutant

            if node_schema.get("additionalProperties") is False:
                mutant = copy.deepcopy(fixture)
                get_at(mutant, path)["__independent_check_extra__"] = True
                yield "extra-property", True, f"{location}: add unexpected property", mutant

            properties = node_schema.get("properties", {})
            for key in value:
                if key in properties and key not in node_schema.get("required", []):
                    mutant = copy.deepcopy(fixture)
                    delete_at(mutant, path + (key,))
                    yield "delete-optional", False, f"{location}: delete {key!r}", mutant

        if "type" in node_schema:
            replacement = WRONG_TYPE[type(value)]
            mutant = copy.deepcopy(fixture)
            yield (
                "wrong-type",
                True,
                f"{location}: replace {type(value).__name__} with wrong type",
                set_at(mutant, path, replacement),
            )

        if path and "enum" in node_schema:
            invalid = "__INVALID_ENUM_VALUE__"
            while invalid in node_schema["enum"]:
                invalid += "_"
            mutant = copy.deepcopy(fixture)
            yield "bad-enum", True, f"{location}: invalid enum", set_at(mutant, path, invalid)

        if path and "const" in node_schema:
            current = node_schema["const"]
            replacement = not current if isinstance(current, bool) else "__INVALID_CONST__"
            mutant = copy.deepcopy(fixture)
            yield "bad-const", True, f"{location}: violate const", set_at(mutant, path, replacement)

        if path and isinstance(value, str):
            minimum = node_schema.get("minLength", 0)
            if isinstance(minimum, int) and minimum > 0:
                mutant = copy.deepcopy(fixture)
                yield "min-length", True, f"{location}: violate minLength", set_at(mutant, path, "")
            if node_schema.get("pattern") == r"\S":
                for blank in ("", "   "):
                    mutant = copy.deepcopy(fixture)
                    yield "blank-string", True, f"{location}: blank {blank!r}", set_at(mutant, path, blank)

        if path and isinstance(value, list) and "minItems" in node_schema:
            minimum = node_schema["minItems"]
            if isinstance(minimum, int) and minimum > 0:
                mutant = copy.deepcopy(fixture)
                shorter = value[: max(0, minimum - 1)]
                yield (
                    "min-items",
                    True,
                    f"{location}: reduce array below minItems {minimum}",
                    set_at(mutant, path, shorter),
                )
        elif path and isinstance(value, list) and node_schema.get("minItems", 0) == 0:
            mutant = copy.deepcopy(fixture)
            yield (
                "empty-array-unconstrained",
                False,
                f"{location}: empty unconstrained array",
                set_at(mutant, path, []),
            )


def invoke_cli(
    package_parent: Path,
    review_path: Path,
    cwd: Path,
    *,
    structural_only: bool = False,
) -> subprocess.CompletedProcess[str]:
    environment = {**os.environ, "PYTHONPATH": str(package_parent)}
    command = [sys.executable, "-m", "schema_review", str(review_path)]
    if structural_only:
        command.append("--structural-only")
    return subprocess.run(
        command,
        env=environment,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def main(report_path: str | None = None) -> int:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    Draft7Validator.check_schema(schema)
    oracle = Draft7Validator(schema)

    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "oracle": "jsonschema Draft7Validator",
        "sha256": {
            "schema": sha256(SCHEMA_PATH),
            "fixture": sha256(FIXTURE_PATH),
            "validator": sha256(PACKAGE / "validate.py"),
        },
    }
    passed = oracle.is_valid(fixture)

    with tempfile.TemporaryDirectory(prefix="schema-review-independent-") as temporary:
        temp = Path(temporary)
        isolated_site = temp / "site"
        isolated_site.mkdir()
        shutil.copytree(
            PACKAGE,
            isolated_site / "schema_review",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        unrelated_cwd = temp / "unrelated-working-directory"
        unrelated_cwd.mkdir()

        first = invoke_cli(isolated_site, FIXTURE_PATH, unrelated_cwd)
        second = invoke_cli(isolated_site, FIXTURE_PATH, unrelated_cwd)
        report["clean_room"] = {
            "exit_code": first.returncode,
            "stdout": first.stdout.strip(),
            "stderr": first.stderr.strip(),
        }
        report["deterministic"] = (
            first.returncode, first.stdout, first.stderr
        ) == (
            second.returncode, second.stdout, second.stderr
        )
        passed = passed and first.returncode == 0 and report["deterministic"]

        counts: dict[str, dict[str, int | bool]] = {}
        survivors: list[dict[str, Any]] = []
        disagreements: list[dict[str, Any]] = []
        mutant_path = temp / "mutant.json"

        for category, expected_invalid, description, mutant in generate_mutants(fixture, schema):
            oracle_valid = oracle.is_valid(mutant)
            mutant_path.write_text(json.dumps(mutant, ensure_ascii=False), encoding="utf-8")
            result = invoke_cli(
                isolated_site,
                mutant_path,
                unrelated_cwd,
                structural_only=True,
            )
            cli_valid = result.returncode == 0
            agreement = oracle_valid == cli_valid

            if category not in counts:
                counts[category] = {
                    "expected_invalid": expected_invalid,
                    "total": 0,
                    "oracle_invalid": 0,
                    "cli_rejected": 0,
                    "agreements": 0,
                }
            counts[category]["total"] += 1
            counts[category]["oracle_invalid"] += int(not oracle_valid)
            counts[category]["cli_rejected"] += int(not cli_valid)
            counts[category]["agreements"] += int(agreement)

            if not agreement:
                disagreements.append({
                    "category": category,
                    "description": description,
                    "oracle_valid": oracle_valid,
                    "cli_exit_code": result.returncode,
                    "stdout": result.stdout.strip(),
                    "stderr": result.stderr.strip(),
                })
            if expected_invalid and (oracle_valid or cli_valid):
                survivors.append({
                    "category": category,
                    "description": description,
                    "oracle_valid": oracle_valid,
                    "cli_exit_code": result.returncode,
                })

        report["mutation"] = {
            "by_category": counts,
            "total": sum(int(item["total"]) for item in counts.values()),
            "oracle_invalid": sum(int(item["oracle_invalid"]) for item in counts.values()),
            "cli_rejected": sum(int(item["cli_rejected"]) for item in counts.values()),
            "agreements": sum(int(item["agreements"]) for item in counts.values()),
            "disagreements": disagreements,
            "mandatory_survivors": survivors,
        }
        passed = (
            passed
            and not disagreements
            and not survivors
            and report["mutation"]["total"] > 0
        )

    report["passed"] = bool(passed)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if report_path:
        Path(report_path).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
