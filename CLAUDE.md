# MCP Manager Development Guide

## IMPORTANT: Always Use UV

**All Python commands in this project MUST use uv to ensure consistent behavior.**

For running Python commands:
```bash
# ❌ WRONG - Don't use direct Python commands
python -m pytest
python mcp_manager/cli.py
pip install textual

# ✅ CORRECT - Always use uv
uv run pytest
uv run python -m mcp_manager
uv add textual
uv pip install textual
```

## Build & Test Commands
```bash
# Run application
uv run python -m mcp_manager

# Run all tests
uv run pytest

# Run specific test
uv run pytest tests/unit/test_models.py
uv run pytest tests/integration/test_server_lifecycle.py

# Run with coverage
uv run pytest --cov=mcp_manager --cov-report=html

# Linting & static analysis
uv run mypy mcp_manager/
uv run ruff check mcp_manager/
uv run ruff format mcp_manager/
```

## Managing Dependencies

```bash
# Add new production dependencies
uv add package_name

# Install development dependencies
uv pip install -e ".[dev]"

# Sync dependencies
uv pip sync
```

## Testing Guidelines

### Test Structure

- `tests/unit/`: Unit tests for individual components
- `tests/integration/`: Integration tests for end-to-end flows
- All test files must be prefixed with `test_`
- All test classes must be prefixed with `Test`
- All test methods must be prefixed with `test_`

### Writing Tests

```python
import pytest
from mcp_manager.domain.models import MCPServer

class TestMCPServer:
    def test_server_initialization(self) -> None:
        server = MCPServer(name="test", repository="https://example.com")
        assert server.name == "test"
```

## Code Style Guidelines

- **Imports**: Standard lib → third-party → local; absolute imports preferred
- **Types**: All functions require parameter & return type annotations
- **Naming**:
  - Classes: PascalCase
  - Functions/variables: snake_case
  - Constants: UPPER_SNAKE_CASE
- **Formatting**: 88 char line length, enforced by ruff
- **Docstrings**: Google style for all modules, classes, and public functions
- **Error handling**: Use specific exceptions with descriptive messages
- **Architecture**: Clean architecture with clear separation of concerns

## Project Structure

```
mcp_manager/
├── domain/          # Core business models and interfaces
├── services/        # Business logic and orchestration
├── adapters/        # External integrations
└── ui/              # Textual UI components

tests/
├── unit/            # Unit tests
└── integration/     # Integration tests
```

## Architecture Principles

1. **Domain-Driven Design**: Core domain models in `domain/`
2. **Dependency Inversion**: Depend on abstractions, not concretions
3. **Separation of Concerns**: UI, business logic, and data access are separate
4. **Pluggable Architecture**: Easy to extend with new adapters
5. **Type Safety**: Comprehensive type hints throughout

## Pre-commit Checklist

- [ ] All tests pass: `uv run pytest`
- [ ] Type checking passes: `uv run mypy mcp_manager/`
- [ ] Linting passes: `uv run ruff check mcp_manager/`
- [ ] Code is formatted: `uv run ruff format mcp_manager/`
- [ ] Documentation is updated
