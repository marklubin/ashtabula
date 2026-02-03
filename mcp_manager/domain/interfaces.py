"""Interfaces and protocols for MCP Manager components."""

from abc import ABC, abstractmethod
from typing import Any, Protocol
from uuid import UUID

from mcp_manager.domain.models import (
    Client,
    ClientConfig,
    ConfigVersion,
    MCPServer,
    ServerConfig,
    ServerRegistry,
    Tool,
)


class ServerRegistryReader(ABC):
    """Interface for reading MCP servers from registries."""

    @abstractmethod
    async def list_registries(self) -> list[ServerRegistry]:
        """List all available registries.

        Returns:
            List of server registries.
        """
        ...

    @abstractmethod
    async def list_servers(self, registry: ServerRegistry) -> list[MCPServer]:
        """List all servers in a registry.

        Args:
            registry: The registry to query.

        Returns:
            List of MCP servers.
        """
        ...

    @abstractmethod
    async def get_server(self, server_id: UUID) -> MCPServer | None:
        """Get a specific server by ID.

        Args:
            server_id: The server ID.

        Returns:
            MCP server or None if not found.
        """
        ...


class ServerExecutor(ABC):
    """Interface for executing MCP servers locally."""

    @abstractmethod
    async def start_server(self, config: ServerConfig) -> None:
        """Start an MCP server.

        Args:
            config: Server configuration.

        Raises:
            RuntimeError: If server fails to start.
        """
        ...

    @abstractmethod
    async def stop_server(self, config_id: UUID) -> None:
        """Stop a running MCP server.

        Args:
            config_id: Configuration ID.
        """
        ...

    @abstractmethod
    async def get_server_status(self, config_id: UUID) -> str:
        """Get the status of a server.

        Args:
            config_id: Configuration ID.

        Returns:
            Server status string.
        """
        ...

    @abstractmethod
    async def test_tool(
        self, config_id: UUID, tool_name: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Test a tool on a running server.

        Args:
            config_id: Configuration ID.
            tool_name: Name of the tool to test.
            parameters: Tool parameters.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not running or tool execution fails.
        """
        ...

    @abstractmethod
    async def list_tools(self, config_id: UUID) -> list[Tool]:
        """List all available tools for a running server.

        Args:
            config_id: Configuration ID.

        Returns:
            List of available tools.
        """
        ...


class ClientPublisher(ABC):
    """Interface for publishing configurations to clients."""

    @abstractmethod
    async def publish_config(self, config: ClientConfig) -> None:
        """Publish configuration to a client.

        Args:
            config: Client configuration to publish.

        Raises:
            RuntimeError: If publishing fails.
        """
        ...

    @abstractmethod
    async def get_current_config(self, client: Client) -> ClientConfig | None:
        """Get current published configuration for a client.

        Args:
            client: The client.

        Returns:
            Current configuration or None.
        """
        ...

    @abstractmethod
    def supports_client_type(self, client_type: str) -> bool:
        """Check if this publisher supports a client type.

        Args:
            client_type: The client type to check.

        Returns:
            True if supported, False otherwise.
        """
        ...


class ConfigStore(ABC):
    """Interface for storing and retrieving configurations."""

    @abstractmethod
    async def save_server_config(self, config: ServerConfig) -> None:
        """Save a server configuration.

        Args:
            config: Server configuration to save.
        """
        ...

    @abstractmethod
    async def get_server_config(self, config_id: UUID) -> ServerConfig | None:
        """Get a server configuration by ID.

        Args:
            config_id: Configuration ID.

        Returns:
            Server configuration or None.
        """
        ...

    @abstractmethod
    async def list_server_configs(self) -> list[ServerConfig]:
        """List all server configurations.

        Returns:
            List of server configurations.
        """
        ...

    @abstractmethod
    async def delete_server_config(self, config_id: UUID) -> None:
        """Delete a server configuration.

        Args:
            config_id: Configuration ID.
        """
        ...

    @abstractmethod
    async def save_client(self, client: Client) -> None:
        """Save a client.

        Args:
            client: Client to save.
        """
        ...

    @abstractmethod
    async def get_client(self, client_id: UUID) -> Client | None:
        """Get a client by ID.

        Args:
            client_id: Client ID.

        Returns:
            Client or None.
        """
        ...

    @abstractmethod
    async def list_clients(self) -> list[Client]:
        """List all clients.

        Returns:
            List of clients.
        """
        ...

    @abstractmethod
    async def save_config_version(self, version: ConfigVersion) -> None:
        """Save a configuration version.

        Args:
            version: Configuration version to save.
        """
        ...

    @abstractmethod
    async def get_config_history(self, client_id: UUID) -> list[ConfigVersion]:
        """Get configuration history for a client.

        Args:
            client_id: Client ID.

        Returns:
            List of configuration versions, newest first.
        """
        ...

    @abstractmethod
    async def get_config_version(self, version_id: UUID) -> ConfigVersion | None:
        """Get a specific configuration version.

        Args:
            version_id: Version ID.

        Returns:
            Configuration version or None.
        """
        ...
