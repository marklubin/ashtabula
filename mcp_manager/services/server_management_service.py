"""Service for managing MCP server configurations."""

from uuid import UUID

from mcp_manager.domain.interfaces import ConfigStore
from mcp_manager.domain.models import (
    EnvironmentVariable,
    MCPServer,
    ServerConfig,
    ServerStatus,
)


class ServerManagementService:
    """Service for managing MCP server installations and configurations."""

    def __init__(self, config_store: ConfigStore) -> None:
        """Initialize the server management service.

        Args:
            config_store: Configuration store implementation.
        """
        self._config_store = config_store

    async def create_server_config(
        self, server: MCPServer, name: str, environment: dict[str, str] | None = None
    ) -> ServerConfig:
        """Create a new server configuration.

        Args:
            server: The MCP server to configure.
            name: User-defined name for this instance.
            environment: Environment variables (optional).

        Returns:
            Created server configuration.

        Raises:
            ValueError: If required environment variables are missing.
        """
        # Validate required environment variables
        env = environment or {}
        missing_vars = []

        for env_var in server.environment_variables:
            if env_var.required and env_var.name not in env:
                if env_var.default is None:
                    missing_vars.append(env_var.name)

        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

        # Create configuration
        config = ServerConfig(
            server=server,
            name=name,
            environment=env,
            status=ServerStatus.STOPPED,
        )

        await self._config_store.save_server_config(config)
        return config

    async def update_server_config(
        self, config_id: UUID, name: str | None = None, environment: dict[str, str] | None = None
    ) -> ServerConfig:
        """Update an existing server configuration.

        Args:
            config_id: Configuration ID to update.
            name: New name (optional).
            environment: New environment variables (optional).

        Returns:
            Updated configuration.

        Raises:
            ValueError: If configuration not found or validation fails.
        """
        config = await self._config_store.get_server_config(config_id)
        if not config:
            raise ValueError(f"Configuration {config_id} not found")

        if name is not None:
            config.name = name

        if environment is not None:
            # Validate required vars
            missing_vars = []
            for env_var in config.server.environment_variables:
                if env_var.required and env_var.name not in environment:
                    if env_var.default is None:
                        missing_vars.append(env_var.name)

            if missing_vars:
                raise ValueError(
                    f"Missing required environment variables: {', '.join(missing_vars)}"
                )

            config.environment = environment

        await self._config_store.save_server_config(config)
        return config

    async def get_server_config(self, config_id: UUID) -> ServerConfig | None:
        """Get a server configuration by ID.

        Args:
            config_id: Configuration ID.

        Returns:
            Server configuration or None if not found.
        """
        return await self._config_store.get_server_config(config_id)

    async def list_server_configs(self) -> list[ServerConfig]:
        """List all server configurations.

        Returns:
            List of server configurations.
        """
        return await self._config_store.list_server_configs()

    async def delete_server_config(self, config_id: UUID) -> None:
        """Delete a server configuration.

        Args:
            config_id: Configuration ID to delete.

        Raises:
            ValueError: If configuration not found.
        """
        config = await self._config_store.get_server_config(config_id)
        if not config:
            raise ValueError(f"Configuration {config_id} not found")

        if config.status == ServerStatus.RUNNING:
            raise ValueError("Cannot delete a running server configuration")

        await self._config_store.delete_server_config(config_id)

    def get_configuration_guide(self, server: MCPServer) -> list[EnvironmentVariable]:
        """Get the configuration guide for a server.

        Args:
            server: The MCP server.

        Returns:
            List of environment variables to configure.
        """
        return server.environment_variables
