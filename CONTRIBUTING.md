# Contributing to llm-response-cache-disk

Thanks for your interest in contributing! This project is a small, zero-dependency,
SQLite-backed disk cache for LLM responses, targeting Python 3.10+. Contributions of
all kinds are welcome: bug reports, documentation, tests, and code.

## Getting started

1. Fork the repository and clone your fork.
2. Create a virtual environment and install the project in editable mode with dev
   dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   pip install --upgrade pip
   pip install -e ".[dev]"
   ```

## Development workflow

1. Create a feature branch off `main`:

   ```bash
   git checkout -b my-change
   ```

2. Make your change. Keep the project dependency-free at runtime — runtime code should
   rely only on the Python standard library.
3. Run the lint and test suite locally before pushing (these mirror CI):

   ```bash
   ruff check src/ tests/
   pytest -v --tb=short
   ```

4. Commit your work with a clear, descriptive message and open a pull request against
   the `main` branch.

## Pull request guidelines

- Keep pull requests focused on a single concern.
- Add or update tests for any behavior change.
- Make sure CI is green: the GitHub Actions workflow runs `ruff` and `pytest` across
  Python 3.10–3.13.
- Update documentation (including the README) when you change public behavior.

## Reporting issues

When filing an issue, please include:

- The Python version and operating system you are using.
- A minimal reproduction of the problem.
- The expected behavior versus what actually happened.

## License

By contributing, you agree that your contributions will be licensed under the
project's [MIT License](LICENSE).
