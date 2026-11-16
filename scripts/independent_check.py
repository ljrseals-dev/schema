#!/usr/bin/env python3
"""Independent execution and schema-derived mutation validation.

This harness deliberately does not import schema_review. It reads the bundled
JSON Schema as data, independently walks the subset of keywords used by that
schema, mutates the reference review, and invokes the validator only through
its public CLI in a subprocess.

The harness is intentionally scoped to the bundled schema and fixture. It is
not a general JSON Schema conformance suite and does not independently oracle
the validator's semantic rules.
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

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "schema_review"
SCHEMA_PATH = PACKAGE / "data" / "quantized_emitter_review.schema.json"
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "review_example.json"

# Replacement values are deliberately of a different JSON/Python type.
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
    """Resolve the local #/definitions references used by the bundled schema."""
    while "$ref" in schema:
        reference = schema["$ref"]
        if not reference.startswith("#/"):
            raise ValueError(f"unsupported schema reference: {reference}")
        target: Any = root
        for component in reference[2:].split("/"):
            target = target[component]
        siblings = {key: value for key, value in schema.items() if key != "$ref"}
        schema = {**target, **siblings}
    return schema


def walk_instances(
    instance: Any,
    schema: dict[str, Any],
    root: dict[str, Any],
    path: tuple[str | int, ...] = (),
) -> Iterator[tuple[tuple[str | int, ...], Any, dict[str, Any]]]:
    """Yield each fixture value with its resolved schema node."""
    schema = resolve_schema(schema, root)
    yield path, instance, schema

    if isinstance(instance, dict):
        for key, child_schema in schema.get("properties", {}).items():
            if key in instance:
                yield from walk_instances(
                    instance[key], child_schema, root, path + (key,)
                )
    elif isinstance(instance, list) and "items" in schema:
        for index, item in enumerate(instance):
            yield from walk_instances(
                item, schema["items"], root, path + (index,)
            )


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
) -> Iterator[tuple[str, bool, str, dict[str, Any]]]:
    """Yield category, mandatory, description, and one mutated review."""
    for path, value, node_schema in walk_instances(fixture, schema, schema):
        location = describe(path)

        if isinstance(value, dict):
            for key in node_schema.get("required", []):
                mutant = copy.deepcopy(fixture)
                delete_at(mutant, path + (key,))
                yield (
                    "delete-required",
                    True,
                    f"{location}: delete required key {key!r}",
                    mutant,
                )

            if node_schema.get("additionalProperties") is False:
                mutant = copy.deepcopy(fixture)
                get_at(mutant, path)["__independent_check_extra__"] = True
                yield (
                    "extra-property",
                    True,
                    f"{location}: add forbidden property",
                    mutant,
                )

            properties = node_schema.get("properties", {})
            for key in value:
                if key in properties and key not in node_schema.get("required", []):
                    mutant = copy.deepcopy(fixture)
                    delete_at(mutant, path + (key,))
                    yield (
                        "delete-optional",
                        False,
                        f"{location}: delete optional key {key!r}",
                        mutant,
                    )

        if path and "type" in node_schema:
            replacement = WRONG_TYPE[type(value)]
            mutant = copy.deepcopy(fixture)
            yield (
                "wrong-type",
                True,
                f"{location}: replace {type(value).__name__} with wrong type",
                set_at(mutant, path, replacement),
            )

        if path and "enum" in node_schema:
            mutant = copy.deepcopy(fixture)
            yield (
                "bad-enum",
                True,
                f"{location}: replace enum value",
                set_at(mutant, path, "__INVALID_ENUM_VALUE__"),
            )

        if path and "const" in node_schema:
            mutant = copy.deepcopy(fixture)
            yield (
                "bad-const",
                True,
                f"{location}: violate const",
                set_at(mutant, path, not node_schema["const"]),
            )

        if (
            path
            and isinstance(value, str)
            and node_schema.get("pattern") == r"\S"
        ):
            for blank in ("", "   "):
                mutant = copy.deepcopy(fixture)
                yield (
                    "blank-string",
                    True,
                    f"{location}: replace string with {blank!r}",
                    set_at(mutant, path, blank),
                )

        if path and isinstance(value, list):
            mandatory = node_schema.get("minItems", 0) >= 1
            mutant = copy.deepcopy(fixture)
            yield (
                "empty-array" if mandatory else "empty-array-unconstrained",
                mandatory,
                f"{location}: empty array",
                set_at(mutant, path, []),
            )


def invoke_cli(package_parent: Path, review_path: Path, cwd: Path) -> subprocess.CompletedProcess[str]:
    environment = {**os.environ, "PYTHONPATH": str(package_parent)}
    return subprocess.run(
        [sys.executable, "-m", "schema_review", str(review_path)],
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

    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "sha256": {
            "schema": sha256(SCHEMA_PATH),
            "fixture": sha256(FIXTURE_PATH),
            "validator": sha256(PACKAGE / "validate.py"),
        },
    }
    passed = True

    with tempfile.TemporaryDirectory(prefix="schema-review-independent-") as temporary:
        temporary_path = Path(temporary)
        isolated_site = temporary_path / "site"
        isolated_site.mkdir()
        shutil.copytree(
            PACKAGE,
            isolated_site / "schema_review",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        unrelated_cwd = temporary_path / "unrelated-working-directory"
        unrelated_cwd.mkdir()

        first = invoke_cli(isolated_site, FIXTURE_PATH, unrelated_cwd)
        second = invoke_cli(isolated_site, FIXTURE_PATH, unrelated_cwd)
        report["clean_room"] = {
            "exit_code": first.returncode,
            "stdout": first.stdout.strip(),
            "stderr": first.stderr.strip(),
        }
        report["deterministic"] = (
            first.returncode,
            first.stdout,
            first.stderr,
        ) == (
            second.returncode,
            second.stdout,
            second.stderr,
        )
        passed = passed and first.returncode == 0 and report["deterministic"]

        counts: dict[str, dict[str, int | bool]] = {}
        survivors: list[dict[str, Any]] = []
        mutant_path = temporary_path / "mutant.json"

        for category, mandatory, description, mutant in generate_mutants(fixture, schema):
            mutant_path.write_text(
                json.dumps(mutant, ensure_ascii=False), encoding="utf-8"
            )
            result = invoke_cli(isolated_site, mutant_path, unrelated_cwd)
            rejected = result.returncode != 0

            if category not in counts:
                counts[category] = {
                    "mandatory": mandatory,
                    "total": 0,
                    "rejected": 0,
                }
            counts[category]["total"] += 1
            counts[category]["rejected"] += int(rejected)

            if not rejected:
                survivors.append(
                    {
                        "category": category,
                        "mandatory": mandatory,
                        "description": description,
                        "exit_code": result.returncode,
                        "stdout": result.stdout.strip(),
                        "stderr": result.stderr.strip(),
                    }
                )

        mandatory_total = sum(
            int(item["total"]) for item in counts.values() if item["mandatory"]
        )
        mandatory_rejected = sum(
            int(item["rejected"]) for item in counts.values() if item["mandatory"]
        )
        report["mutation"] = {
            "by_category": counts,
            "total": sum(int(item["total"]) for item in counts.values()),
            "rejected": sum(int(item["rejected"]) for item in counts.values()),
            "mandatory_total": mandatory_total,
            "mandatory_rejected": mandatory_rejected,
            "survivors": survivors,
        }
        passed = passed and mandatory_total == mandatory_rejected

    report["passed"] = bool(passed)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if report_path:
        Path(report_path).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
