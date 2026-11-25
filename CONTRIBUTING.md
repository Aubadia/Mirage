# Contributing to Mirage

## Sync the development environment

Install uv, then run these commands from the repository root. The Python version comes from [`.python-version`](.python-version). Development tools are declared in the `dev` group in [`pyproject.toml`](pyproject.toml).

```sh
uv python install 3.14
uv sync --locked --dev
```

This installs Mirage in editable mode for local development. Re-run the sync after changing branches or pulling dependency updates. `--locked` checks that [`uv.lock`](uv.lock) agrees with the manifest and uses the locked versions. For an approved dependency change, update the manifest and lockfile together. Use `uv lock --check` to check their consistency.

CI installs a non-editable copy of Mirage. To reproduce that environment locally, use `uv sync --locked --dev --no-editable` and re-sync after source changes. Run `uv sync --locked --dev` to return to editable development. [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)

## Continuous integration

The [CI workflow](.github/workflows/ci.yml) runs on pull requests, pushes to `main`, and manual dispatch. GitHub runs the three jobs in parallel on Linux with Python 3.14 and the locked development dependencies. After syncing, run the same checks locally.

| Job | Commands |
| --- | --- |
| Lint | `uv run --no-sync ruff check .` and `uv run --no-sync ruff format --check .` |
| Static Analysis | `uv run --no-sync pyrefly check` |
| Tests | `uv run --no-sync pytest tests --hypothesis-profile=ci --cov=mirage --cov-report=term-missing` |

Each job reports its own result. The Tests job prints coverage for `mirage` and lists uncovered lines. CI currently reports coverage without a minimum percentage.

## Fix lint and formatting failures

Use the file path, line, and rule code in Ruff's output to locate each finding. Run the auto-fixer and formatter on the files you changed. For example, to fix an edit in the CLI module, run the following.

```sh
uv run --no-sync ruff check --fix src/mirage/__init__.py
uv run --no-sync ruff format src/mirage/__init__.py
git diff -- src/mirage/__init__.py
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
```

Resolve remaining lint findings manually. Use `uv run --no-sync ruff rule F401` to look up a rule, substituting the code from the diagnostic. Review the diff before committing. [Ruff linter guide](https://docs.astral.sh/ruff/linter/)

## Pass Pyrefly type checking

Run `uv run --no-sync pyrefly check` from the repository root. Pyrefly loads the `[tool.pyrefly]` section in `pyproject.toml`. Use the diagnostic's location and expected and inferred types to trace the mismatch.

- For unresolved imports, sync the development environment and verify the import path and dependency declaration.
- For incompatible arguments or return values, align the implementation and annotations with the approved interface. Narrow optional values with a check such as `if value is None` before using them where a concrete value is required.
- Use a targeted suppression only when the reason can be documented. Rerun the full type check after the fix.

[Pyrefly configuration guide](https://pyrefly.org/en/docs/configuration/)

## Write and run tests

Follow the spec-first [testing workflow](AGENTS.md#testing). Write unit and end-to-end tests from the approved `requirement.md` before implementation. Partition inputs into relevant equivalence classes, select representative values, and cover edge cases. Have a human review the baseline before implementing.

| Tool | Use |
| --- | --- |
| `pytest` | Write assertions and reusable fixtures. Use `@pytest.mark.parametrize` for representative inputs and edge cases, `pytest.raises` for expected exceptions, and fixtures such as `tmp_path` or `capsys` for files and CLI output. |
| `pytest-mock` | Use the `mocker` fixture to patch external calls or spy on interactions. For example, stub an HTTP request in a unit test or check that the installed CLI entry point calls `main`. |
| `hypothesis` | Use `@given` and `hypothesis.strategies` to test invariants across generated inputs. Keep explicit tests for acceptance cases with specific inputs and expected outcomes. CI selects the `ci` profile. |
| `pytest-cov` | Measure which lines in `mirage` the tests exercise. Use the missing-line report to identify behavior that needs a test. |

Run the full suite or select a test by file or name.

```sh
uv run --no-sync pytest tests
uv run --no-sync pytest tests/test_ci_smoke.py -k console_entry_point
uv run --no-sync pytest tests --hypothesis-profile=ci --cov=mirage --cov-report=term-missing
```

The current smoke tests cover the installed CLI entry point and the property-based test tooling. The CI command discovers additional tests placed under `tests/`.
