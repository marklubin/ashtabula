"""GitHub-based MCP server registry reader."""

from typing import Any
from uuid import UUID, uuid5, NAMESPACE_URL

import httpx
from pydantic import HttpUrl

from mcp_manager.domain.interfaces import ServerRegistryReader
from mcp_manager.domain.models import (
    EnvironmentVariable,
    MCPServer,
    ServerRegistry,
    Tool,
    ToolParameter,
)


class GitHubRegistryReader(ServerRegistryReader):
    """Read MCP servers from GitHub-based registries."""

    def __init__(self, registry_urls: list[str] | None = None) -> None:
        """Initialize the GitHub registry reader.

        Args:
            registry_urls: List of registry URLs. If None, uses default registries.
        """
        self._registries: list[ServerRegistry] = []

        # Default registries (example - would need real ones)
        default_urls = registry_urls or [
            "https://raw.githubusercontent.com/modelcontextprotocol/servers/main/registry.json"
        ]

        for url in default_urls:
            self._registries.append(
                ServerRegistry(
                    name="MCP Official Registry", url=HttpUrl(url), trusted=True
                )
            )

        self._cache: dict[UUID, MCPServer] = {}

    async def list_registries(self) -> list[ServerRegistry]:
        """List all available registries."""
        return self._registries.copy()

    async def list_servers(self, registry: ServerRegistry) -> list[MCPServer]:
        """List all servers in a registry."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(str(registry.url))
                response.raise_for_status()
                data = response.json()

                servers = []
                for server_data in data.get("servers", []):
                    server = self._parse_server_data(server_data, registry)
                    servers.append(server)
                    self._cache[server.id] = server

                return servers

            except httpx.HTTPError as e:
                raise RuntimeError(f"Failed to fetch registry: {e}") from e
            except Exception as e:
                raise RuntimeError(f"Failed to parse registry data: {e}") from e

    async def get_server(self, server_id: UUID) -> MCPServer | None:
        """Get a specific server by ID."""
        # Check cache first
        if server_id in self._cache:
            return self._cache[server_id]

        # Refresh cache from all registries
        for registry in self._registries:
            await self.list_servers(registry)

        return self._cache.get(server_id)

    def _parse_server_data(
        self, data: dict[str, Any], registry: ServerRegistry
    ) -> MCPServer:
        """Parse server data from registry JSON.

        Args:
            data: Server data from registry.
            registry: Source registry.

        Returns:
            Parsed MCP server.
        """
        # Generate deterministic UUID from repository URL
        repo_url = data["repository"]
        server_id = uuid5(NAMESPACE_URL, repo_url)

        # Parse environment variables
        env_vars = []
        for env_data in data.get("environment_variables", []):
            env_vars.append(
                EnvironmentVariable(
                    name=env_data["name"],
                    description=env_data.get("description", ""),
                    required=env_data.get("required", False),
                    default=env_data.get("default"),
                )
            )

        # Parse tools
        tools = []
        for tool_data in data.get("tools", []):
            params = []
            for param_data in tool_data.get("parameters", []):
                params.append(
                    ToolParameter(
                        name=param_data["name"],
                        type=param_data["type"],
                        description=param_data.get("description", ""),
                        required=param_data.get("required", False),
                        default=param_data.get("default"),
                    )
                )

            tools.append(
                Tool(
                    name=tool_data["name"],
                    description=tool_data.get("description", ""),
                    parameters=params,
                )
            )

        return MCPServer(
            id=server_id,
            name=data["name"],
            description=data.get("description", ""),
            repository=HttpUrl(repo_url),
            registry=registry,
            install_command=data.get("install_command", "npm install"),
            run_command=data.get("run_command", "npx mcp-server"),
            environment_variables=env_vars,
            tools=tools,
            version=data.get("version", "latest"),
            tags=data.get("tags", []),
        )
