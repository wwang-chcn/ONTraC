#!/usr/bin/env python3
"""Generate Markdown API reference files from Python docstrings.

This script parses source files via ``ast`` and emits one Markdown page per
module. It is designed to produce portable API docs that can be copied into
another documentation repository (MkDocs, Docusaurus, Hugo, etc.).
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Iterable, Sequence


@dataclass
class FunctionDoc:
    """Structured function or method documentation item."""

    name: str
    signature: str
    docstring: str
    is_property: bool = False


@dataclass
class ClassDoc:
    """Structured class documentation item."""

    name: str
    docstring: str
    methods: list[FunctionDoc] = field(default_factory=list)


@dataclass
class ModuleDoc:
    """Structured module documentation item."""

    name: str
    docstring: str
    is_package: bool = False
    functions: list[FunctionDoc] = field(default_factory=list)
    classes: list[ClassDoc] = field(default_factory=list)
    submodules: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for API generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=Path("src/ONTraC"),
        help="Path to the package source directory (default: src/ONTraC).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/api_reference"),
        help="Directory where markdown files are written.",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=["__pycache__"],
        help="Directory names to exclude. Can be passed multiple times.",
    )
    parser.add_argument(
        "--hide-private",
        action="store_true",
        help="Hide members whose names start with a single underscore.",
    )
    return parser.parse_args()


def clean_docstring(docstring: str | None, fallback: str) -> str:
    """Normalize docstring text for Markdown output."""
    if not docstring:
        return fallback
    text = dedent(docstring).strip()
    return text if text else fallback


def should_skip_name(name: str, hide_private: bool) -> bool:
    """Return ``True`` if the symbol should be hidden from output."""
    if name.startswith("__") and name.endswith("__"):
        return False
    return hide_private and name.startswith("_")


def format_arg(arg: ast.arg, default: ast.expr | None = None) -> str:
    """Format one argument including annotation/default."""
    text = arg.arg
    if arg.annotation is not None:
        text += f": {ast.unparse(arg.annotation)}"
    if default is not None:
        text += f" = {ast.unparse(default)}"
    return text


def format_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Convert AST function node into a Python-style signature string."""
    parts: list[str] = []

    positional = list(node.args.posonlyargs) + list(node.args.args)
    defaults = list(node.args.defaults)
    n_defaults = len(defaults)
    default_start = len(positional) - n_defaults

    for i, arg in enumerate(positional):
        default = defaults[i - default_start] if i >= default_start else None
        parts.append(format_arg(arg, default))

    if node.args.posonlyargs:
        parts.insert(len(node.args.posonlyargs), "/")

    if node.args.vararg is not None:
        vararg = f"*{node.args.vararg.arg}"
        if node.args.vararg.annotation is not None:
            vararg += f": {ast.unparse(node.args.vararg.annotation)}"
        parts.append(vararg)
    elif node.args.kwonlyargs:
        parts.append("*")

    for kwarg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        parts.append(format_arg(kwarg, default))

    if node.args.kwarg is not None:
        kwarg = f"**{node.args.kwarg.arg}"
        if node.args.kwarg.annotation is not None:
            kwarg += f": {ast.unparse(node.args.kwarg.annotation)}"
        parts.append(kwarg)

    signature = f"{node.name}({', '.join(parts)})"
    if node.returns is not None:
        signature += f" -> {ast.unparse(node.returns)}"
    return signature


def is_property_method(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check whether a method is decorated with ``@property``."""
    for deco in node.decorator_list:
        if isinstance(deco, ast.Name) and deco.id == "property":
            return True
    return False


def parse_class(node: ast.ClassDef, hide_private: bool) -> ClassDoc:
    """Parse class docs and contained methods from AST."""
    class_doc = ClassDoc(
        name=node.name,
        docstring=clean_docstring(
            ast.get_docstring(node),
            fallback=f"API reference entry for class `{node.name}`.",
        ),
    )
    for child in node.body:
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if should_skip_name(child.name, hide_private):
            continue
        method_doc = FunctionDoc(
            name=child.name,
            signature=format_signature(child),
            docstring=clean_docstring(
                ast.get_docstring(child),
                fallback=f"API reference entry for `{node.name}.{child.name}`.",
            ),
            is_property=is_property_method(child),
        )
        class_doc.methods.append(method_doc)
    return class_doc


def module_name_from_path(path: Path, src_root: Path) -> str:
    """Resolve importable module name from source file path."""
    relative = path.relative_to(src_root.parent).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def iter_python_files(src_dir: Path, excluded_dir_names: Sequence[str]) -> Iterable[Path]:
    """Yield Python files under ``src_dir`` in deterministic order."""
    excluded = set(excluded_dir_names)
    files = []
    for path in src_dir.rglob("*.py"):
        if path == src_dir / "__init__.py":
            continue
        if any(part in excluded for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def parse_module(path: Path, src_root: Path, hide_private: bool) -> ModuleDoc:
    """Parse one Python module into a structured documentation object."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    module_name = module_name_from_path(path, src_root)
    top_functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    top_classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    is_package = path.name == "__init__.py"
    module_fallback = (
        f"Auto-generated API reference for module `{module_name}`.\n\n"
        f"This module defines {len(top_functions)} top-level function(s) and "
        f"{len(top_classes)} class(es)."
    )
    module_doc = ModuleDoc(
        name=module_name,
        docstring=clean_docstring(ast.get_docstring(tree), fallback=module_fallback),
        is_package=is_package,
    )

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if should_skip_name(node.name, hide_private):
                continue
            module_doc.functions.append(
                FunctionDoc(
                    name=node.name,
                    signature=format_signature(node),
                    docstring=clean_docstring(
                        ast.get_docstring(node),
                        fallback=f"API reference entry for function `{node.name}`.",
                    ),
                )
            )
        elif isinstance(node, ast.ClassDef):
            if should_skip_name(node.name, hide_private):
                continue
            module_doc.classes.append(parse_class(node=node, hide_private=hide_private))

    return module_doc


def render_function(item: FunctionDoc, class_name: str | None = None) -> list[str]:
    """Render one function/method block as Markdown lines."""
    if class_name is None:
        title = f"### `{item.signature}`"
    elif item.is_property:
        title = f"#### `{class_name}.{item.name}` *(property)*"
    else:
        title = f"#### `{class_name}.{item.signature}`"
    return [title, "", item.docstring, ""]


def render_module(module_doc: ModuleDoc) -> str:
    """Render one module API page to Markdown text."""
    lines: list[str] = [f"# `{module_doc.name}`", "", module_doc.docstring, ""]

    if module_doc.is_package and module_doc.submodules:
        lines.extend(["## Submodules", ""])
        for child in module_doc.submodules:
            lines.append(f"- [{child}]({child}.md)")
        lines.append("")

    if module_doc.functions:
        lines.extend(["## Functions", ""])
        for func in module_doc.functions:
            lines.extend(render_function(func))

    if module_doc.classes:
        lines.extend(["## Classes", ""])
        for cls in module_doc.classes:
            lines.extend([f"### `{cls.name}`", "", cls.docstring, ""])
            if cls.methods:
                lines.extend(["#### Methods", ""])
                for method in cls.methods:
                    lines.extend(render_function(method, class_name=cls.name))

    return "\n".join(lines).strip() + "\n"


def render_index(modules: Sequence[ModuleDoc]) -> str:
    """Render website-friendly API landing page."""
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    top_level_modules: list[ModuleDoc] = []
    grouped_submodules: dict[str, list[ModuleDoc]] = {}
    for module in modules:
        parts = module.name.split(".")
        if len(parts) <= 2:
            top_level_modules.append(module)
        else:
            group = parts[1]
            grouped_submodules.setdefault(group, []).append(module)

    lines = [
        "# ONTraC API Reference",
        "",
        "This section documents ONTraC public modules, classes, and functions.",
        "All pages are generated directly from in-code docstrings.",
        "",
        "## How To Use This Section",
        "",
        "- Start from the package area that matches your task (for example `analysis`, `model`, `train`).",
        "- Open module pages to view functions, classes, and method/property docs.",
        "- Use signatures on each page as the source of truth for call patterns.",
        "",
        "## Generation Metadata",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Source: `{Path('src/ONTraC')}`",
        "",
        "## Top-Level Modules",
        "",
    ]

    for module in sorted(top_level_modules, key=lambda item: item.name):
        filename = f"{module.name}.md"
        lines.append(f"- [{module.name}]({filename})")

    lines.extend(["", "## Submodules By Area", ""])

    for group_name in sorted(grouped_submodules.keys(), key=str.lower):
        lines.extend([f"### {group_name}", ""])
        for module in sorted(grouped_submodules[group_name], key=lambda item: item.name):
            filename = f"{module.name}.md"
            lines.append(f"- [{module.name}]({filename})")
        lines.append("")
    return "\n".join(lines)


def render_generation_doc() -> str:
    """Render maintainer notes for regenerating API markdown."""
    return "\n".join(
        [
            "# API Reference Generation",
            "",
            "This note is for maintainers who regenerate API docs from ONTraC source code.",
            "",
            "## Output",
            "",
            "- Website-ready API pages are generated into `docs/api_reference/`.",
            "- The landing page is `docs/api_reference/index.md`.",
            "",
            "## Regenerate",
            "",
            "```bash",
            "python scripts/generate_api_reference.py",
            "```",
            "",
            "Optional flags:",
            "",
            "```bash",
            "python scripts/generate_api_reference.py --hide-private",
            "python scripts/generate_api_reference.py --out-dir docs/api_reference",
            "```",
            "",
        ]
    )


def render_transfer_doc() -> str:
    """Render maintainer notes for copying docs to a website repository."""
    return "\n".join(
        [
            "# API Reference Transfer Guide",
            "",
            "This note is for maintainers who publish API pages in another website repository.",
            "",
            "## What To Copy",
            "",
            "- Copy the full `docs/api_reference/` directory.",
            "- Do not copy this transfer note unless you want maintainer instructions on the site.",
            "",
            "## Typical Publish Flow",
            "",
            "1. Regenerate docs in ONTraC repo.",
            "2. Copy `docs/api_reference/` into your website docs tree.",
            "3. Add `api_reference/index.md` to website navigation.",
            "4. Commit and deploy website docs.",
            "",
        ]
    )


def main() -> None:
    """Generate module markdown pages and an index."""
    args = parse_args()
    src_dir: Path = args.src_dir.resolve()
    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    modules: list[ModuleDoc] = []
    for py_file in iter_python_files(src_dir=src_dir, excluded_dir_names=args.exclude_dir):
        module_doc = parse_module(path=py_file, src_root=src_dir, hide_private=args.hide_private)
        modules.append(module_doc)

    by_name = {module.name: module for module in modules}
    all_names = set(by_name.keys())
    for module in modules:
        if not module.is_package:
            continue
        prefix = f"{module.name}."
        direct_children = []
        for candidate in all_names:
            if not candidate.startswith(prefix):
                continue
            suffix = candidate[len(prefix):]
            if suffix and "." not in suffix:
                direct_children.append(candidate)
        module.submodules = sorted(direct_children)

    for module_doc in modules:
        (out_dir / f"{module_doc.name}.md").write_text(render_module(module_doc), encoding="utf-8")

    modules.sort(key=lambda x: x.name)

    expected_files = {f"{module.name}.md" for module in modules}
    expected_files.add("index.md")
    for existing in out_dir.glob("*.md"):
        if existing.name not in expected_files:
            existing.unlink()

    (out_dir / "index.md").write_text(render_index(modules=modules), encoding="utf-8")
    (out_dir.parent / "api_reference_generation.md").write_text(render_generation_doc(), encoding="utf-8")
    (out_dir.parent / "api_reference_transfer.md").write_text(render_transfer_doc(), encoding="utf-8")
    print(f"Generated {len(modules)} module pages in {out_dir}")


if __name__ == "__main__":
    main()
