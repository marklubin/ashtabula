"""Publisher for Claude Code CLI configuration."""

import json
from pathlib import Path
from typing import Any

import aiofiles

from mcp_manager.domain.interfaces import ClientPublisher
from mcp_manager.domain.models import Client, ClientConfig


class ClaudeCodePublisher(ClientPublisher):
    """Publish MCP server configurations to Claude Code CLI."""

    async def publish_config(self, config: ClientConfig) -> None:
        """Publish configuration to Claude Code CLI."""
        try:
            config_path = Path(config.client.config_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)

            # Build Claude Code configuration format
            mcp_config: dict[str, Any] = {"mcpServers": {}}

            for server_config in config.server_configs:
                # Parse run command into command and args
                cmd_parts = server_config.server.run_command.split()
                command = cmd_parts[0] if cmd_parts else "npx"
                args = cmd_parts[1:] if len(cmd_parts) > 1 else []

                server_entry = {
                    "command": command,
                    "args": args,
                }

                # Add environment variables if any
                if server_config.environment:
                    server_entry["env"] = server_config.environment

                mcp_config["mcpServers"][server_config.name] = server_entry

            # Write configuration file
            async with aiofiles.open(config_path, "w") as f:
                await f.write(json.dumps(mcp_config, indent=2))

        except Exception as e:
            raise RuntimeError(
                f"Failed to publish to Claude Code CLI: {e}"
            ) from e

    async def get_current_config(self, client: Client) -> ClientConfig | None:
        """Get current published configuration for Claude Code CLI."""
        try:
            config_path = Path(client.config_path)
            if not config_path.exists():
                return None

            async with aiofiles.open(config_path, "r") as f:
                content = await f.read()
                data = json.loads(content)

            # Parse the configuration
            # Note: We can't fully reconstruct ServerConfig objects from the file
            # This is a simplified implementation
            return None  # Would need additional data to reconstruct

        except Exception:
            return None

    def supports_client_type(self, client_type: str) -> bool:
        """Check if this publisher supports Claude Code CLI."""
        return client_type == "claude_code"
