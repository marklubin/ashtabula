# MCP Manager

A powerful Text User Interface (TUI) for managing Model Context Protocol (MCP) servers across multiple clients.

## Features

- 🔍 **Browse & Discover**: Browse MCP servers from trusted repositories
- ⚙️ **Guided Installation**: Interactive setup with environment variable configuration
- 🧪 **Interactive Testing**: Test MCP server tools locally with an API browser interface
- 🔌 **Multi-Client Support**: Publish configurations to Claude Code CLI, Claude Desktop, and more
- 📊 **Configuration Tracking**: Visual tracking of configurations across clients with rollback support
- 🔧 **Pluggable Architecture**: Easy to extend with new client integrations

## Installation

```bash
# Install with uv
uv pip install -e .

# Or for development
uv pip install -e ".[dev]"
```

## Usage

```bash
# Launch the TUI
mcp-manager

# Or run directly with uv
uv run python -m mcp_manager
```

## Development

```bash
# Run tests
uv run pytest

# Run type checking
uv run mypy mcp_manager/

# Run linting
uv run ruff check mcp_manager/

# Format code
uv run ruff format mcp_manager/
```

## Architecture

The project follows a clean architecture approach:

- `domain/`: Core business models and interfaces
- `services/`: Business logic and orchestration
- `adapters/`: External integrations (file system, HTTP, MCP clients)
- `ui/`: Textual-based user interface components

## Client Support

Out of the box support for:
- Claude Code CLI
- Claude Desktop

Easy to extend with custom client integrations through the pluggable adapter system.

## License

MIT
