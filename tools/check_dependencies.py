#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Enforce the direction of dependence: nothing shippable may reach outside.

lgpsf is a general-purpose library; the glaciology (PIG) work is a downstream
consumer of it. So PIG may depend on lgpsf and lgpsf may not depend on PIG --
and the same for any other private repository or absolute path on one machine.

This is the mechanical form of that rule, because a policy nobody can run is a
policy that erodes. It checks everything a reader can reach -- what ships in a
wheel or tarball, plus the docs, developer notes and experiments that are
public on the repo even when excluded from the sdist. The one exception is
`archive/`, which is frozen history: editing it to satisfy a lint would falsify
the record.

Naming a private problem in prose is FINE and deliberate -- `docs/validation.md`
explains what PIG is so those citations resolve. What is banned is a path: an
import, an include, or a filename that only exists on one machine. And even in
prose, keep the reference GENERAL: cite the result ("validated on the Pine
Island Glacier Hessian"), never a specific private artifact -- a note, a
table, a driver script -- because a reference the reader cannot open is a dead
end however it is phrased, and this check cannot catch the described kind.

Run:  python tools/check_dependencies.py
"""
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

# Every tracked directory a reader can reach: what ships in a wheel or
# tarball, plus the docs, developer notes and experiments that are public on
# the repo even when excluded from the sdist. `archive/` is the one exception
# -- it is frozen third-party-to-us history, excluded from the sdist, and
# editing it to satisfy a lint would falsify the record.
SHIPPED = ["include", "bindings", "tests", "examples", "docs", "dev",
           "experiments", "cmake", ".github", "CMakeLists.txt",
           "pyproject.toml", "README.md", "CONTRIBUTING.md", "CHANGELOG.md",
           "CITATION.cff"]

# `ellipsoid_tree` is a PUBLIC library we explicitly depend on, so referencing
# it is correct rather than a leak. Everything else outside the repo is not.
ALLOWED_EXTERNAL = {"ellipsoid_tree"}

# POINTERS, not names. "the localpsf paper" is a citation and must pass;
# anywhere a reader is invited to go and cannot must not. For a directory the
# distinction is a trailing slash, which is what turns a name into somewhere a
# build could reach. For a file it is the name itself: a public reader cannot
# open `slice38_lgpsf_operator.py`, so citing it by filename -- with or without
# a line number -- is a dead reference dressed up as a source. Cite the result
# instead ("measured on the PIG Hessian"), not the artifact behind it.
PATTERNS = [
    (re.compile(r"(?:~|/home/[^/\s]+|/Users/[^/\s]+)/"),
     "an absolute path that exists on one machine"),
    (re.compile(r"\b(?:nicks_research_experiments|ellipsoid_psf\w*"
                r"|localpsf\w*|ymir[\w-]*)/"),
     "a path into a private repo"),
    (re.compile(r"\bslice\d+[a-z]?_[a-z0-9_]+\.(?:py|npz|cpp)\b"),
     "a file that only exists in the private research repo"),
]

SKIP_SUFFIXES = {".png", ".pdf", ".pyc", ".so"}


def offending_lines(path):
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return
    for number, line in enumerate(text.splitlines(), start=1):
        for pattern, why in PATTERNS:
            match = pattern.search(line)
            if match and match.group(0).strip("/ ") not in ALLOWED_EXTERNAL:
                yield number, line.strip(), why


def main():
    problems = []
    for entry in SHIPPED:
        root = REPO / entry
        if not root.exists():
            continue
        paths = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in paths:
            if not path.is_file() or path.suffix in SKIP_SUFFIXES:
                continue
            if "__pycache__" in path.parts:
                continue
            for number, line, why in offending_lines(path):
                problems.append(
                    f"{path.relative_to(REPO)}:{number}: {why}\n    {line}")

    if problems:
        print("lgpsf must not depend on anything private. Found "
              f"{len(problems)} reference(s):\n", file=sys.stderr)
        for problem in problems:
            print(problem, file=sys.stderr)
        print("\nNaming a private problem in PROSE is fine -- see "
              "docs/validation.md. A PATH is not.", file=sys.stderr)
        return 1

    print(f"ok: no private dependencies in {', '.join(SHIPPED)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
