# Ashtabula Development Guide

## Build & Test Commands
```bash
# Run application
python -m ashtabula.main

# Download models (required before tests)
python -m scripts.download_models

# Run all tests
pytest tests/

# Run specific test
pytest tests/test_stt.py
pytest tests/test_stt.py::TestSTTProvider::test_basic_transcription

# Linting & static analysis
python -m scripts.static_analysis
mypy ashtabula/
ruff check ashtabula/
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