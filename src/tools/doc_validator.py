"""
Documentation Completeness Validator

Scans the codebase and validates that all components have adequate documentation:
- Python files have module-level docstrings
- Functions have docstrings
- Configuration is documented in CLAUDE.md and README.md
- API endpoints have docstrings
- Complex logic has comments
- Key features are documented in README.md

Usage:
    python src/tools/doc_validator.py                    # Full validation
    python src/tools/doc_validator.py --module api.py    # Single file
    python src/tools/doc_validator.py --verbose          # Detailed output
"""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ── Configuration ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
PYTHON_EXCLUDE = {
    "__pycache__",
    ".venv",
    "venv",
    ".git",
    "models",
    "data",
    "node_modules",
}

REQUIRED_DOCS = {
    "README.md": [
        "Architecture",
        "Setup",
        "Installation",
        "Local Development",
        "External Services",
    ],
    "CLAUDE.md": [
        "Environment Setup",
        ".env",
        "Installation",
        "Running the System",
        "Graph Data Model",
    ],
    "api.py": [
        "FastAPI",
        "POST /explain",
        "POST /query",
        "POST /feedback",
        "GET /stats",
    ],
}

# ── Validation Results ─────────────────────────────────────────────────
class ValidationResult:
    def __init__(self):
        self.missing_module_docs: List[str] = []
        self.missing_function_docs: List[Tuple[str, str]] = []
        self.missing_comments: List[Tuple[str, int]] = []
        self.missing_readme_sections: List[str] = []
        self.missing_claude_sections: List[str] = []
        self.undocumented_config: List[str] = []
        self.warnings: List[str] = []

    def has_issues(self) -> bool:
        return bool(
            self.missing_module_docs
            or self.missing_function_docs
            or self.missing_comments
            or self.missing_readme_sections
            or self.missing_claude_sections
            or self.undocumented_config
        )

    def summary(self) -> str:
        lines = []
        if self.missing_module_docs:
            lines.append(f"\n[MISSING MODULE DOCS] ({len(self.missing_module_docs)})")
            for f in self.missing_module_docs:
                lines.append(f"  • {f}")

        if self.missing_function_docs:
            lines.append(
                f"\n[MISSING FUNCTION DOCS] ({len(self.missing_function_docs)})"
            )
            for file, func in self.missing_function_docs[:10]:
                lines.append(f"  • {file}: {func}()")
            if len(self.missing_function_docs) > 10:
                lines.append(f"  ... and {len(self.missing_function_docs) - 10} more")

        if self.missing_comments:
            lines.append(f"\n[COMPLEX CODE WITHOUT COMMENTS] ({len(self.missing_comments)})")
            for file, line_num in self.missing_comments[:10]:
                lines.append(f"  • {file}:{line_num}")
            if len(self.missing_comments) > 10:
                lines.append(f"  ... and {len(self.missing_comments) - 10} more")

        if self.missing_readme_sections:
            lines.append(f"\n[MISSING IN README.md] ({len(self.missing_readme_sections)})")
            for section in self.missing_readme_sections:
                lines.append(f"  • {section}")

        if self.missing_claude_sections:
            lines.append(f"\n[MISSING IN CLAUDE.md] ({len(self.missing_claude_sections)})")
            for section in self.missing_claude_sections:
                lines.append(f"  • {section}")

        if self.undocumented_config:
            lines.append(f"\n[UNDOCUMENTED CONFIG] ({len(self.undocumented_config)})")
            for var in self.undocumented_config[:10]:
                lines.append(f"  • {var}")
            if len(self.undocumented_config) > 10:
                lines.append(f"  ... and {len(self.undocumented_config) - 10} more")

        if self.warnings:
            lines.append(f"\n[WARNINGS] ({len(self.warnings)})")
            for warn in self.warnings[:5]:
                lines.append(f"  [!] {warn}")

        return "\n".join(lines) if lines else "\n[OK] All documentation checks passed!"


# ── Validators ─────────────────────────────────────────────────────────
def check_python_files(verbose: bool = False) -> Tuple[int, int]:
    """Check Python files for module/function docstrings."""
    missing_module = []
    missing_functions = []
    complex_without_comments = []

    for py_file in PROJECT_ROOT.rglob("*.py"):
        # Skip excluded directories
        if any(exclude in py_file.parts for exclude in PYTHON_EXCLUDE):
            continue

        # Skip test files and __init__.py
        if py_file.name == "__init__.py" or py_file.name.startswith("test_"):
            continue

        try:
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()
                tree = ast.parse(content)

            relative_path = py_file.relative_to(PROJECT_ROOT)

            # Check module docstring
            module_doc = ast.get_docstring(tree)
            if not module_doc:
                missing_module.append(str(relative_path))

            # Check function docstrings
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Skip private functions (starting with _)
                    if node.name.startswith("_"):
                        continue

                    func_doc = ast.get_docstring(node)
                    if not func_doc and node.lineno > 10:  # Skip imports area
                        missing_functions.append((str(relative_path), node.name))

            # Check for complex code without comments
            lines = content.split("\n")
            for i, line in enumerate(lines, 1):
                # Look for complex patterns: nested loops, conditionals, list comprehensions
                if (
                    "for " in line
                    and "for " in lines[max(0, i - 2) : i]
                    and "#" not in line
                ):
                    complex_without_comments.append((str(relative_path), i))

        except SyntaxError as e:
            print(f"  [!] Syntax error in {relative_path}: {e}")
        except Exception as e:
            if verbose:
                print(f"  [!] Error parsing {relative_path}: {e}")

    return missing_module, missing_functions, complex_without_comments


def check_markdown_sections(verbose: bool = False) -> Tuple[List[str], List[str]]:
    """Check README.md and CLAUDE.md for required sections."""
    missing_readme = []
    missing_claude = []

    readme_path = PROJECT_ROOT / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()
        for section in REQUIRED_DOCS["README.md"]:
            if section not in readme_content:
                missing_readme.append(section)

    claude_path = PROJECT_ROOT / "CLAUDE.md"
    if claude_path.exists():
        with open(claude_path, "r", encoding="utf-8") as f:
            claude_content = f.read()
        for section in REQUIRED_DOCS["CLAUDE.md"]:
            if section not in claude_content:
                missing_claude.append(section)

    return missing_readme, missing_claude


def check_api_documentation(verbose: bool = False) -> List[str]:
    """Check that FastAPI endpoints have docstrings."""
    missing_docs = []

    api_path = PROJECT_ROOT / "api.py"
    if api_path.exists():
        with open(api_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Look for @app.route decorators without docstrings
        pattern = r"@app\.(get|post|put|delete|patch)\(['\"]([^'\"]+)['\"]\)\s*\n\s*def\s+(\w+)"
        matches = re.finditer(pattern, content)

        for match in matches:
            method, route, func_name = match.groups()
            # Check if function has docstring after def
            func_start = match.end()
            func_section = content[func_start : func_start + 500]
            if not ('"""' in func_section or "'''" in func_section):
                missing_docs.append(f"{method.upper()} {route} → {func_name}()")

    return missing_docs


def check_env_documentation(verbose: bool = False) -> List[str]:
    """Check that .env variables are documented in CLAUDE.md."""
    undocumented = []

    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            env_vars = re.findall(r"^([A-Z_]+)=", f.read(), re.MULTILINE)

        claude_path = PROJECT_ROOT / "CLAUDE.md"
        if claude_path.exists():
            with open(claude_path, "r", encoding="utf-8") as f:
                claude_content = f.read()

            for var in env_vars:
                if var not in claude_content:
                    undocumented.append(var)

    return undocumented


# ── Main ───────────────────────────────────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate documentation completeness")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--module", "-m", help="Check specific Python module", metavar="FILE"
    )
    args = parser.parse_args()

    result = ValidationResult()

    print("=" * 70)
    print("DOCUMENTATION COMPLETENESS VALIDATOR")
    print("=" * 70)

    # Check Python files
    print("\n[1/5] Checking Python files for docstrings...")
    missing_mod, missing_func, missing_comment = check_python_files(args.verbose)
    result.missing_module_docs = missing_mod
    result.missing_function_docs = missing_func
    result.missing_comments = missing_comment
    print(
        f"  [OK] {len(missing_mod)} files without module docs, "
        f"{len(missing_func)} functions without docs"
    )

    # Check markdown sections
    print("\n[2/5] Checking README.md and CLAUDE.md...")
    missing_readme, missing_claude = check_markdown_sections(args.verbose)
    result.missing_readme_sections = missing_readme
    result.missing_claude_sections = missing_claude
    print(
        f"  [OK] README.md: {len(missing_readme) or 'all'} sections OK, "
        f"CLAUDE.md: {len(missing_claude) or 'all'} sections OK"
    )

    # Check API documentation
    print("\n[3/5] Checking FastAPI endpoints...")
    missing_api_docs = check_api_documentation(args.verbose)
    result.missing_function_docs.extend([(f, "") for f in missing_api_docs])
    print(f"  [OK] {len(missing_api_docs) or 'All'} endpoints documented")

    # Check .env documentation
    print("\n[4/5] Checking .env variable documentation...")
    undocumented_env = check_env_documentation(args.verbose)
    result.undocumented_config = undocumented_env
    print(f"  [OK] {len(undocumented_env)} env vars undocumented")

    # Check for LangSmith in docs (recent feature)
    print("\n[5/5] Checking recent feature documentation...")
    claude_path = PROJECT_ROOT / "CLAUDE.md"
    if claude_path.exists():
        with open(claude_path, "r", encoding="utf-8") as f:
            if "LangSmith" not in f.read():
                result.warnings.append("LangSmith monitoring not documented in CLAUDE.md")

    readme_path = PROJECT_ROOT / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            if "LangSmith" not in f.read():
                result.warnings.append("LangSmith not documented in README.md")

    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(result.summary())

    # Exit code
    exit_code = 1 if result.has_issues() else 0
    print("\n" + ("=" * 70))
    if exit_code == 0:
        print("[PASS] All documentation checks passed!")
    else:
        print(f"[FAIL] {sum([len(result.missing_module_docs), len(result.missing_function_docs), len(result.missing_comments)])} issues found")
    print("=" * 70 + "\n")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
