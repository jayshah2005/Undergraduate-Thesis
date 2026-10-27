---
name: commenting-python
description: "Follow Python commenting and documentation conventions. Use when: adding comments to Python files, writing docstrings, documenting classes, using Sphinx param style, suppressing linter warnings with noqa, or following flake8 style rules (Q000, D205, D209, N802)."
---

# Commenting Python files

This skill describes the commenting conventions used in this project.
Follow these patterns when adding or editing comments so the codebase stays
consistent.

## Module-level docstring

Every `.py` file starts with a **single-line** triple-quoted docstring.
Sentence case, ends with a period.

```python
"""Load one mean spectrum per vine from vine_hsi.h5."""
```

```python
"""Ask whether SNV leaf-mean spectra differ by grape variety."""
```

## Class docstrings

Use a triple-quoted docstring immediately after the `class` statement. For
simple classes, a single-line docstring is preferred. For complex classes, use a
summary line, a blank line, then a longer description. Sentence case, ends with
a period.

```python
class LeafTable:
    """One row of labels per vine."""
```

```python
class MainModuleBinder:
    """Prepares a main module for transpiling.

    This transformer collects the names of classes, functions, imports,
    and filters out everything else at the root level.
    """
```

## Function / method docstrings

Start with a **verb in imperative mood**. Single-line for simple functions,
multi-line for complex ones. Ends with a period.

**Single-line:**

```python
def snv(X):
    """Centre and scale each spectrum over its own bands."""
```

**Multi-line (summary + body):**

```python
def pca_fit(X, n_pcs=10):
    """Fit covariance PCA of X with a full SVD.

    998 x 204 is tiny, so this does not use a randomised approximation.
    """
```

## Parameter documentation — Sphinx style

Use Sphinx `:param:` / `:type:` / `:return:` / `:rtype:` fields for parameter
documentation. This is the single accepted style across the project.

```python
def bind_module(tree: ast.Module, path: str = None) -> MainBindings:
    """Reduce a module to the bindings a worker needs.

    :param tree: The source tree.
    :type tree: ast.Module
    :return: A bindings result with code and metadata.
    :rtype: MainBindings
    """
```

```python
def __init__(self, num_workers: Optional[int], export_dir: Optional[str]):
    """Create a new Behaviors scheduler.

    :param num_workers: The number of worker interpreters to start. If
        None, defaults to the number of available cores minus one.
    :type num_workers: Optional[int]
    :param export_dir: The directory to which the target module will be
        exported for worker import. If None, a temporary directory will
        be created and removed on shutdown.
    :type export_dir: Optional[str]
    """
```

## `.pyi` stub file docstrings

Stub files use **Sphinx-style** docstrings on every public function, class,
and constant. Constants use an attribute docstring on the line after the
declaration:

```python
TIMEOUT: str
"""Sentinel value returned by :func:`receive` when a timeout occurs."""
```

```python
def send(tag: str, contents: Any):
    """Send a message.

    :param tag: The tag is an arbitrary label that can be used to receive this message.
    :type tag: str
    :param contents: The contents of the message.
    :type contents: Any
    """
```

## Inline `#` comments

**An inline comment defaults to a single line of at most 120
characters, or it is deleted.** Verbose multi-line inline comments rot
as the surrounding code changes and reduce readability. Place the
comment on the line above the code, at the current indentation:

```python
        orphan_cowns = _core.cowns()
        if len(orphan_cowns) != 0:
            logger.debug("acquiring orphan cowns")
            # acquire orphaned cowns so their XIData is freed before teardown
```

A multi-line `#` inline comment is permitted **only** when it records
something the code cannot express and that does not fit on one line:
a non-obvious concurrency invariant, a behavior-changing transpiler
rule, the rationale above a version gate, or a reference anchor that
needs context. When in doubt, collapse to one line. Docstrings are
*not* inline comments and are exempt — they should carry in-depth,
useful documentation across as many lines as the reader needs.

Same-line `#` comments are acceptable for very short annotations:

```python
        except RuntimeError:
            pass  # already destroyed
```

Comment **why**, not what the line does. Skip comments that restate the code.

## `# noqa:` suppression comments

Suppress linter warnings with `# noqa: CODE` at the end of the line.
Place the suppression on the **line that contains the violation**, not a
surrounding line:

```python
    def visit_FunctionDef(self, node: ast.FunctionDef):  # noqa: N802
```

```python
from pca_common import snv  # noqa: E402
```

```python
# CORRECT — noqa on the line that references the loop variable
@when(acc)
def _(a):
    a.value.add(val_to_add)  # noqa: B023

# WRONG — noqa on the def line does not suppress the reference on the next line
@when(acc)
def _(a):  # noqa: B023
    a.value.add(val_to_add)   # still triggers B023
```

## `# BEGIN` / `# END` markers

Code-generation insertion points:

```python
# BEGIN boc_export
# END boc_export
```

## TODO comments

Use `# TODO` on its own line, followed by the item(s):

```python
# TODO
# invert 2x2, 3x3, general algorithm for NxN
# det
# trace
```

---

## Linting rules (flake8)

The project enforces style with `flake8`:

| Rule | Setting |
|------|---------|
| `inline-quotes` | `double` — use `"` not `'` (Q000) |
| `max-line-length` | 120 |
| `docstring-convention` | `google` (with Sphinx `:param:` fields — `napoleon` handles both) |
| `extend-ignore` | E203, N812, N817 |
| `per-file-ignores` | `test/*`: D103 (missing public-function docstring), D403 |

### Multi-line class docstrings (D205 / D209)

A multi-line docstring must have a **summary line**, a **blank line**, then the
body. The closing `"""` must be on its own line:

```python
# CORRECT
class Foo:
    """Short summary line.

    Longer description goes here after the blank line.
    """

# WRONG — triggers D205 and D209
class Foo:
    """This wraps onto a second line
    without a blank separator."""
```

### Naming in test files (N802)

Test function names must be **lowercase**. Even when testing a property like
`.T`, spell the test name with a lowercase letter:

```python
# CORRECT
def test_t_equals_transpose(self, mat): ...

# WRONG — triggers N802
def test_T_equals_transpose(self, mat): ...
```

---

## Quick reference

| Element | Python convention |
|---------|-------------------|
| Function docs | `"""Imperative summary."""` |
| Class docs | `"""Single-line or multi-line."""` after `class` |
| Parameter docs | `:param name:` / `:type name:` (Sphinx) |
| Inline comments | `#` on line above code |
| End-of-line | `#` for very short annotations or `# noqa:` |
| Sentence case | Docstrings start with uppercase |
| Trailing period | Docstrings end with a period |
| TODOs | `# TODO` on its own line |
| Docstring style | Sphinx `:param:` / `:type:` / `:return:` / `:rtype:` |

## Common pitfalls

| Pitfall | Fix |
|---------|-----|
| Using Google-style `Args:` blocks | Use Sphinx `:param:` / `:type:` style instead |
| Forgetting the trailing period in Python docstrings | All Python docstrings end with a period |
| Duplicating comments | Do not place a `#` line that repeats the docstring below it |
| Placing `# noqa:` on the wrong line | Put it on the line with the actual violation, not a surrounding `def` or decorator line |
| Multi-line docstring without a blank line after the summary | Add a blank line between the summary sentence and the body (D205). Close `"""` on its own line (D209) |
| Using single quotes in Python | The project enforces double quotes (`inline-quotes = double`). Use `"nan"` not `'nan'` |
| Assigning a value to an unused variable | If the return value is not needed, call the function without assignment to avoid F841 |
