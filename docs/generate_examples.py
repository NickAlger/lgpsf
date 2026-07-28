# SPDX-License-Identifier: MIT
"""Generate example-based documentation for lgpsf.

For every program in `examples/`, this runs it in a scratch directory, captures
its stdout and any figures it writes, copies the figures into `docs/img/`, and
emits `docs/examples/<name>.md` showing the program, its real output, and its
figures. Show, don't tell.

The generated pages are committed. `--check` re-extracts each page's embedded
PROGRAM SOURCE and fails if it differs from the file on disk, which is what CI
runs: it catches the common drift of editing an example and forgetting to
regenerate, and it takes no time at all.

It deliberately does NOT re-run the examples in CI. A full pass is tens of
minutes here -- the operator examples fit thousands of rows -- and results
legitimately differ across compiler flags (see docs/reproducibility.md), so a
byte-comparison of captured output would be flaky by construction. Regenerate
locally when behavior changes:

    python3 docs/generate_examples.py                       # everything
    python3 docs/generate_examples.py --only lg_modes       # one page
    python3 docs/generate_examples.py --check               # sources current?

C++ pages need the examples built; point `--build-dir` at a Release build.
"""
import argparse
import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples"
PAGES = REPO / "docs" / "examples"
IMG = REPO / "docs" / "img"

# Shared infrastructure, not a lesson: it defines the test problem the other
# examples fit, and has no output of its own.
SKIP = {"frog_kernel"}

# Every example is documented at its own defaults. If one ever gets slow
# enough to need abbreviating here, that is a signal its default is wrong.
ABBREVIATED = {}


def page_name(path):
    """Page stem for an example source.

    Three examples exist in BOTH languages -- fit_one_psf, operator_fit_frog,
    preconditioner -- so the stem alone would collide and one page would
    silently overwrite the other.
    """
    return path.stem if path.suffix == ".py" else f"{path.stem}-cpp"


def python_intro(source):
    """Module docstring -> (title, intro markdown, source without the docstring).

    The docstring's first line is the title. The trailing "Run:" block is
    dropped: on a generated page the invocation is already implicit.
    """
    tree = ast.parse(source)
    doc = ast.get_docstring(tree) or ""
    lines = doc.splitlines()
    title = lines[0].rstrip(".") if lines else "Example"

    body = []
    for line in lines[1:]:
        if re.match(r"^\s*(Run:|python examples/)", line):
            break
        body.append(line)
    intro = "\n".join(body).strip()

    # Strip the docstring itself from the displayed program: the page shows it
    # as prose above, and repeating it doubles the length of every listing.
    stripped = source.split('"""', 2)
    program = stripped[2].lstrip("\n") if len(stripped) == 3 else source
    return title, intro, program


def cpp_intro(source):
    """Leading '//' block -> (title, intro markdown, source without it)."""
    lines = source.splitlines()
    intro, start = [], 0
    for i, line in enumerate(lines):
        if line.startswith("//"):
            # Strip the marker and at most ONE space: display equations in
            # these banners are indented, and lstrip() would flatten them into
            # prose.
            text = line[2:]
            intro.append(text[1:] if text.startswith(" ") else text)
        elif line.strip() == "" and intro:
            continue
        else:
            start = i
            break
    while start < len(lines) and lines[start].strip() == "":
        start += 1
    # Drop the SPDX line and any blank separators before the real title.
    intro = [line for line in intro if not line.startswith("SPDX")]
    while intro and not intro[0]:
        intro.pop(0)
    title = intro[0] if intro else "Example"
    return title, "\n".join(intro[1:]).strip(), "\n".join(lines[start:])


def run(command, scratch, env=None):
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True,
                            timeout=3600, env=env)
    if result.returncode != 0:
        sys.exit(f"example failed: {' '.join(map(str, command))}\n"
                 f"{result.stdout}\n{result.stderr}")
    figures = sorted(p for p in Path(scratch).iterdir() if p.suffix == ".png")
    # The scratch directory is a fresh temporary path on every run, so leaving
    # it in the captured output would rewrite these pages each time they are
    # regenerated, for no reason. Show where the example writes by default.
    return result.stdout.replace(str(scratch), "examples"), figures


def build_page(name, title, intro, program, language, stdout, figures, note):
    parts = [f"# {title}", ""]
    if intro:
        parts += [intro, ""]
    if figures:
        parts += ["## Figures", ""]
        for figure in figures:
            parts += [f"![{figure}](../img/{name}__{figure})", ""]
    parts += ["## Output", "", "```text", stdout.rstrip() or "(no output)",
              "```", ""]
    if note:
        parts += [note, ""]
    parts += [f"## Program", "", f"```{language}", program.rstrip(), "```", ""]
    suffix = "cpp" if language == "cpp" else "py"
    stem = name[:-4] if name.endswith("-cpp") else name
    parts += ["---", "",
              "*Generated by `docs/generate_examples.py` from "
              f"[`examples/{stem}.{suffix}`](../../examples/{stem}.{suffix}); "
              "the output and figures above come from actually running it.*", ""]
    return "\n".join(parts)


def embedded_program(page_text, language):
    """The program listing a committed page carries, or None."""
    match = re.search(rf"## Program\n\n```{language}\n(.*?)\n```", page_text,
                      re.DOTALL)
    return match.group(1) if match else None


def check():
    """Does every page's embedded source still match the file on disk?"""
    stale, checked = [], 0
    for page in sorted(PAGES.glob("*.md")):
        if page.name == "README.md":
            continue
        stem = page.stem
        if stem.endswith("-cpp"):
            source_path, language = EXAMPLES / f"{stem[:-4]}.cpp", "cpp"
        else:
            source_path, language = EXAMPLES / f"{stem}.py", "python"
        if not source_path.exists():
            stale.append(f"docs/examples/{page.name} (no such example)")
            continue
        checked += 1
        source = source_path.read_text()
        program = (python_intro(source) if language == "python"
                   else cpp_intro(source))[2]
        if embedded_program(page.read_text(), language) != program.rstrip():
            stale.append(f"docs/examples/{page.name}")
    if stale:
        print("These pages no longer match their example source:",
              file=sys.stderr)
        for path in stale:
            print(f"  {path}", file=sys.stderr)
        print("\nRegenerate with: python3 docs/generate_examples.py",
              file=sys.stderr)
        return 1
    print(f"ok: all {checked} example pages carry current source")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default="build-release")
    parser.add_argument("--only", help="regenerate a single page")
    parser.add_argument("--check", action="store_true",
                        help="verify embedded sources are current, run nothing")
    args = parser.parse_args()

    if args.check:
        return check()

    PAGES.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    build_dir = (REPO / args.build_dir).resolve()

    entries = []
    for path in sorted(EXAMPLES.glob("*.py")) + sorted(EXAMPLES.glob("*.cpp")):
        if path.stem in SKIP or (args.only and page_name(path) != args.only):
            continue
        entries.append(path)
    if not entries:
        sys.exit("nothing to generate")

    index = []
    for path in entries:
        name = page_name(path)
        source = path.read_text()
        is_python = path.suffix == ".py"
        title, intro, program = (python_intro(source) if is_python
                                 else cpp_intro(source))

        with tempfile.TemporaryDirectory() as scratch:
            extra = ABBREVIATED.get(name, [])
            if is_python:
                env = dict(os.environ)
                env["MPLCONFIGDIR"] = scratch
                command = [sys.executable, str(path), "--outdir", scratch]
                # Only the figure-writing examples accept --outdir.
                if "--outdir" not in source:
                    command = [sys.executable, str(path)]
                command += extra
            else:
                executable = build_dir / "examples" / path.stem
                if not executable.exists():
                    sys.exit(f"missing {executable}; build with "
                             "-DCMAKE_BUILD_TYPE=Release first")
                env = None
                command = [str(executable), scratch]
            print(f"running {name} ...", flush=True)
            stdout, figures = run(command, scratch, env)

            names = []
            for figure in figures:
                shutil.copyfile(figure, IMG / f"{name}__{figure.name}")
                names.append(figure.name)

        note = None
        if name in ABBREVIATED:
            note = (f"> Run here with `{' '.join(ABBREVIATED[name])}` to keep "
                    "documentation generation quick; the defaults in the "
                    "program are larger.")
        page = build_page(name, title, intro, program,
                          "python" if is_python else "cpp", stdout, names, note)
        (PAGES / f"{name}.md").write_text(page)
        index.append((name, title, is_python))
        print(f"  wrote docs/examples/{name}.md ({len(names)} figures)")

    if not args.only:
        lines = ["# Examples", "",
                 "Each page is a complete program, its real output, and the "
                 "figures it draws — regenerated from the code by "
                 "`docs/generate_examples.py`.", "",
                 "See [`../../examples/README.md`](../../examples/README.md) "
                 "for what each one teaches and the order to read them in.", ""]
        for language, heading in ((True, "Python"), (False, "C++")):
            selected = [e for e in index if e[2] is language]
            if not selected:
                continue
            lines += [f"## {heading}", ""]
            lines += [f"- [{title}]({name}.md)" for name, title, _ in selected]
            lines += [""]
        (PAGES / "README.md").write_text("\n".join(lines))
        print("  wrote docs/examples/README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
