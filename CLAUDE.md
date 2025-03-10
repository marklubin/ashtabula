# Ashtabula Development Guide

## IMPORTANT: Always Use UV

**All Python commands in this project MUST use uv to ensure consistent behavior.**

For running Python commands:
```bash
# ❌ WRONG - Don't use direct Python commands
python -m pytest
python scripts/download_models.py
pip install numpy

# ✅ CORRECT - Always use uv
uv run pytest
uv run python -m scripts.download_models
uv add numpy
uv pip install transitions
```

## Build & Test Commands
```bash
# Run application
uv run python -m ashtabula.main

# Download models (required before tests)
uv run python -m scripts.download_models

# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_stt.py
uv run pytest tests/test_stt.py::TestSTTProvider::test_basic_transcription

# Linting & static analysis
uv run python -m scripts.static_analysis
uv run mypy ashtabula/
uv run ruff check ashtabula/
```

## Managing Dependencies

```bash
# Add new production dependencies
uv add package_name

# Install a development dependency
uv pip install package_name

# Update dependencies
uv pip freeze > requirements.txt
```

## Testing Guidelines

### IMPORTANT: Proper Test Implementation

All tests MUST be implemented to run correctly with the project's established testing tools and patterns. Tests should pass without workarounds or subversions of the testing infrastructure.

**✅ CORRECT Approach:**
- Fix the underlying code to make tests pass
- Address dependency issues through proper project configuration
- Use the standard test harness (pytest) as configured in the project
- Add missing dependencies to the project configuration properly
- Fix import errors by correcting the code architecture

**❌ INCORRECT Approaches:**
- Creating mock implementations just to bypass tests
- Writing alternate test files to avoid using pytest
- Using direct Python execution instead of the project's test tools
- Suppressing or commenting out failing tests
- Modifying tests to pass without fixing underlying issues
- Creating custom test runners that bypass the standard harness

### Example: Fixing Failing Tests

If a test is failing with:
```
ERROR: ModuleNotFoundError: No module named 'transitions'
```

**✅ CORRECT Solution:**
```bash
# Add the dependency properly to the project
uv add transitions

# Fix any code issues properly
# Then run tests normally
uv run pytest
```

**❌ INCORRECT Solution:**
```bash
# Don't create workarounds
python -m unittest test_simple.py  # Bypassing pytest
# Don't write custom test files that avoid the issue
# Don't mock the missing module just to make tests pass
```

## Code Style Guidelines
- **Imports**: Standard lib → third-party → local; absolute imports preferred
- **Types**: All functions require parameter & return type annotations
- **Naming**: Classes=PascalCase, functions/variables=snake_case, constants=UPPER_SNAKE_CASE
- **Formatting**: 88 char line length, PEP8 compliant (enforced by ruff)
- **Docstrings**: Google style for all modules, classes, and functions
- **Error handling**: Use specific exceptions with descriptive messages
- **Testing**: Test classes prefixed with "Test", methods with "test_"
- **Architecture**: Abstract base classes with provider implementations