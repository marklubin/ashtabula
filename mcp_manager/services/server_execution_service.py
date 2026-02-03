"""Service for executing and testing MCP servers."""

from datetime import datetime
from uuid import UUID

from mcp_manager.domain.interfaces import ConfigStore, ServerExecutor
from mcp_manager.domain.models import ServerConfig, ServerStatus, Tool


class ServerExecutionService:
    """Service for running and testing MCP servers locally."""

    def __init__(
        self, server_executor: ServerExecutor, config_store: ConfigStore
    ) -> None:
        """Initialize the server execution service.

        Args:
            server_executor: Server executor implementation.
            config_store: Configuration store implementation.
        """
        self._executor = server_executor
        self._config_store = config_store

    async def start_server(self, config_id: UUID) -> None:
        """Start an MCP server.

        Args:
            config_id: Configuration ID.

        Raises:
            ValueError: If configuration not found.
            RuntimeError: If server fails to start.
        """
        config = await self._config_store.get_server_config(config_id)
        if not config:
            raise ValueError(f"Configuration {config_id} not found")

        if config.status == ServerStatus.RUNNING:
            return  # Already running

        try:
            config.status = ServerStatus.STARTING
            config.error_message = None
            await self._config_store.save_server_config(config)

            await self._executor.start_server(config)

            config.status = ServerStatus.RUNNING
            config.last_started_at = datetime.now()
            await self._config_store.save_server_config(config)

        except Exception as e:
            config.status = ServerStatus.ERROR
            config.error_message = str(e)
            await self._config_store.save_server_config(config)
            raise RuntimeError(f"Failed to start server: {e}") from e

    async def stop_server(self, config_id: UUID) -> None:
        """Stop a running MCP server.

        Args:
            config_id: Configuration ID.

        Raises:
            ValueError: If configuration not found.
        """
        config = await self._config_store.get_server_config(config_id)
        if not config:
            raise ValueError(f"Configuration {config_id} not found")

        if config.status == ServerStatus.STOPPED:
            return  # Already stopped

        try:
            await self._executor.stop_server(config_id)
            config.status = ServerStatus.STOPPED
            config.error_message = None
            await self._config_store.save_server_config(config)

        except Exception as e:
            config.status = ServerStatus.ERROR
            config.error_message = str(e)
            await self._config_store.save_server_config(config)
            raise

    async def restart_server(self, config_id: UUID) -> None:
        """Restart an MCP server.

        Args:
            config_id: Configuration ID.
        """
        await self.stop_server(config_id)
        await self.start_server(config_id)

    async def get_server_status(self, config_id: UUID) -> ServerStatus:
        """Get the status of a server.

        Args:
            config_id: Configuration ID.

        Returns:
            Server status.

        Raises:
            ValueError: If configuration not found.
        """
        config = await self._config_store.get_server_config(config_id)
        if not config:
            raise ValueError(f"Configuration {config_id} not found")

        return config.status

    async def list_tools(self, config_id: UUID) -> list[Tool]:
        """List available tools for a running server.

        Args:
            config_id: Configuration ID.

        Returns:
            List of available tools.

        Raises:
            ValueError: If configuration not found.
            RuntimeError: If server is not running.
        """
        config = await self._config_store.get_server_config(config_id)
        if not config:
            raise ValueError(f"Configuration {config_id} not found")

        if config.status != ServerStatus.RUNNING:
            raise RuntimeError("Server is not running")

        return await self._executor.list_tools(config_id)

    async def test_tool(
        self, config_id: UUID, tool_name: str, parameters: dict[str, any]
    ) -> dict[str, any]:
        """Test a tool on a running server.

        Args:
            config_id: Configuration ID.
            tool_name: Name of the tool to test.
            parameters: Tool parameters.

        Returns:
            Tool execution result.

        Raises:
            ValueError: If configuration not found.
            RuntimeError: If server is not running or tool execution fails.
        """
        config = await self._config_store.get_server_config(config_id)
        if not config:
            raise ValueError(f"Configuration {config_id} not found")

        if config.status != ServerStatus.RUNNING:
            raise RuntimeError("Server is not running")

        return await self._executor.test_tool(config_id, tool_name, parameters)
