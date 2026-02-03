"""CLI entry point for MCP Manager."""

from pathlib import Path

import typer

from mcp_manager.ui import MCPManagerApp

app = typer.Typer(
    name="mcp-manager",
    help="MCP Manager - A TUI for managing MCP servers across multiple clients",
)


@app.command()
def main(
    config_dir: Path = typer.Option(
        None,
        "--config-dir",
        "-c",
        help="Configuration directory (default: ~/.mcp-manager)",
    ),
) -> None:
    """Launch the MCP Manager TUI."""
    mcp_app = MCPManagerApp(config_dir=config_dir)
    mcp_app.run()


if __name__ == "__main__":
    app()
