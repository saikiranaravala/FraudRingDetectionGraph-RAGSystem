# Documentation Completeness Validator

A comprehensive script that scans your codebase and validates documentation completeness across Python files, API endpoints, configuration, and markdown files.

## Overview

The validator checks for:

1. **Python Files**
   - Module-level docstrings
   - Function docstrings (public functions only)
   - Comments explaining complex code

2. **API Endpoints** (api.py)
   - FastAPI endpoint docstrings
   - Required endpoints documented

3. **Configuration**
   - .env variables documented in CLAUDE.md
   - Configuration options explained

4. **Markdown Documentation**
   - Required sections in README.md
   - Required sections in CLAUDE.md

5. **Recent Features**
   - LangSmith monitoring documented
   - Recent implementations mentioned in docs

## Installation

No additional dependencies needed. Uses Python standard library only.

## Usage

### Full Validation

```bash
python src/tools/doc_validator.py
```

Runs all checks and reports issues.

### Verbose Mode

```bash
python src/tools/doc_validator.py --verbose
```

Shows detailed output and additional warnings.

### Check Specific Module

```bash
python src/tools/doc_validator.py --module api.py
```

Validates a specific Python file.

## Output

The validator produces a structured report with sections:

```
[MISSING MODULE DOCS] — Files without module-level docstrings
[MISSING FUNCTION DOCS] — Functions without docstrings
[COMPLEX CODE WITHOUT COMMENTS] — Code sections needing clarification
[MISSING IN README.md] — Required README sections
[MISSING IN CLAUDE.md] — Required CLAUDE.md sections
[UNDOCUMENTED CONFIG] — .env variables not documented
[WARNINGS] — General warnings about documentation
```

### Exit Codes

- **0** — All documentation checks passed
- **1** — Issues found (see VALIDATION SUMMARY)

## Current Status (2026-04-25)

Run the validator to see current documentation status:

```bash
$ python src/tools/doc_validator.py
[MISSING MODULE DOCS] (1)
  • testOpenRouterKey.py

[MISSING FUNCTION DOCS] (51)
  • api.py: health()
  • src/main.py: cmd_load_graph()
  • ... and 41 more

[MISSING IN README.md] (2)
  • Setup
  • Installation
```

## How to Fix Issues

### Missing Function Docstrings

Add docstrings to public functions:

```python
def your_function(param1: str, param2: int) -> dict:
    """Brief description of what the function does.
    
    Args:
        param1: What this parameter is for
        param2: What this parameter is for
    
    Returns:
        Dictionary with keys: 'result', 'status'
    """
    # Implementation
```

### Missing Module Docstring

Add at the top of Python files:

```python
"""
Module name and brief description.

This module handles [specific responsibility].
Key classes/functions: [ClassName, function_name]
"""

from __future__ import annotations
# Rest of imports and code
```

### Missing README.md Sections

Ensure README.md includes:
- Architecture overview
- Setup instructions
- Installation steps
- Local Development guide
- External Services list

### Missing CLAUDE.md Sections

Ensure CLAUDE.md includes:
- Environment Setup
- .env template with all variables
- Installation instructions
- Running the System
- Graph Data Model documentation

## Integration with CI/CD

Add to your CI/CD pipeline to validate documentation before merging:

```bash
# In your build script
python src/tools/doc_validator.py
if [ $? -ne 0 ]; then
    echo "Documentation validation failed"
    exit 1
fi
```

## Configuration

To modify what the validator checks, edit the `REQUIRED_DOCS` dictionary in `src/tools/doc_validator.py`:

```python
REQUIRED_DOCS = {
    "README.md": [
        "Your Required Sections",
        "Here",
    ],
    "CLAUDE.md": [
        "Your Required Sections",
        "Here",
    ],
    "api.py": [
        "Your Required Endpoints",
        "Here",
    ],
}
```

## Excluded Directories

By default, the validator skips:
- `__pycache__`
- `.venv` and `venv`
- `.git`
- `models` and `data`
- `node_modules`

Add more to `PYTHON_EXCLUDE` if needed.

## Tips for Better Documentation

1. **Write docs as you code** — Don't leave it for later
2. **Make docstrings specific** — "Process claim data" is better than "Processes data"
3. **Document complex logic** — If it took you 5 minutes to understand, add a comment
4. **Keep docs in sync** — When you change code, update the docs
5. **Link to related docs** — Reference sections in CLAUDE.md from code comments

## Example: Running on Your Codebase

```bash
$ cd FraudRingDetectionGraph-RAGSystem
$ python src/tools/doc_validator.py

======================================================================
DOCUMENTATION COMPLETENESS VALIDATOR
======================================================================

[1/5] Checking Python files for docstrings...
  [OK] 1 files without module docs, 51 functions without docs

[2/5] Checking README.md and CLAUDE.md...
  [OK] README.md: 2 sections OK, CLAUDE.md: all sections OK

[3/5] Checking FastAPI endpoints...
  [OK] All endpoints documented

[4/5] Checking .env variable documentation...
  [OK] 0 env vars undocumented

[5/5] Checking recent feature documentation...

======================================================================
VALIDATION SUMMARY
======================================================================
[FAIL] 52 issues found
```

## Next Steps

1. Run the validator: `python src/tools/doc_validator.py`
2. Review the issues reported
3. Add missing docstrings/comments to highest-impact files first
4. Re-run validator to verify improvements
5. Add to CI/CD to catch new undocumented code
