"""Unit tests for services."""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
from pydantic import HttpUrl

from mcp_manager.domain.interfaces import (
    ClientPublisher,
    ConfigStore,
    ServerExecutor,
    ServerRegistryReader,
)
from mcp_manager.domain.models import (
    Client,
    ClientConfig,
    ClientType,
    ConfigVersion,
    MCPServer,
    ServerConfig,
    ServerRegistry,
    ServerStatus,
    Tool,
)
from mcp_manager.services import (
    ClientConfigService,
    ConfigVersionService,
    RegistryService,
    ServerExecutionService,
    ServerManagementService,
)


# Mock implementations for testing


class MockRegistryReader(ServerRegistryReader):
    """Mock registry reader for testing."""

    def __init__(self) -> None:
        """Initialize mock registry reader."""
        self.registries = [
            ServerRegistry(
                name="Test Registry",
                url=HttpUrl("https://example.com/registry.json"),
                trusted=True,
            )
        ]
        self.servers = [
            MCPServer(
                name="test-server-1",
                description="Test server 1",
                repository=HttpUrl("https://github.com/test/server1"),
                install_command="npm install server1",
                run_command="npx server1",
            ),
            MCPServer(
                name="test-server-2",
                description="Test server 2",
                repository=HttpUrl("https://github.com/test/server2"),
                install_command="npm install server2",
                run_command="npx server2",
            ),
        ]

    async def list_registries(self) -> list[ServerRegistry]:
        """List registries."""
        return self.registries

    async def list_servers(self, registry: ServerRegistry) -> list[MCPServer]:
        """List servers."""
        return self.servers

    async def get_server(self, server_id: UUID) -> MCPServer | None:
        """Get server by ID."""
        for server in self.servers:
            if server.id == server_id:
                return server
        return None


class MockConfigStore(ConfigStore):
    """Mock config store for testing."""

    def __init__(self) -> None:
        """Initialize mock config store."""
        self.server_configs: dict[UUID, ServerConfig] = {}
        self.clients: dict[UUID, Client] = {}
        self.versions: dict[UUID, ConfigVersion] = {}

    async def save_server_config(self, config: ServerConfig) -> None:
        """Save server config."""
        self.server_configs[config.id] = config

    async def get_server_config(self, config_id: UUID) -> ServerConfig | None:
        """Get server config."""
        return self.server_configs.get(config_id)

    async def list_server_configs(self) -> list[ServerConfig]:
        """List server configs."""
        return list(self.server_configs.values())

    async def delete_server_config(self, config_id: UUID) -> None:
        """Delete server config."""
        if config_id in self.server_configs:
            del self.server_configs[config_id]

    async def save_client(self, client: Client) -> None:
        """Save client."""
        self.clients[client.id] = client

    async def get_client(self, client_id: UUID) -> Client | None:
        """Get client."""
        return self.clients.get(client_id)

    async def list_clients(self) -> list[Client]:
        """List clients."""
        return list(self.clients.values())

    async def save_config_version(self, version: ConfigVersion) -> None:
        """Save config version."""
        self.versions[version.id] = version

    async def get_config_history(self, client_id: UUID) -> list[ConfigVersion]:
        """Get config history."""
        versions = [v for v in self.versions.values() if v.client.id == client_id]
        return sorted(versions, key=lambda v: v.version, reverse=True)

    async def get_config_version(self, version_id: UUID) -> ConfigVersion | None:
        """Get config version."""
        return self.versions.get(version_id)


class MockServerExecutor(ServerExecutor):
    """Mock server executor for testing."""

    def __init__(self) -> None:
        """Initialize mock executor."""
        self.running_servers: set[UUID] = set()

    async def start_server(self, config: ServerConfig) -> None:
        """Start server."""
        self.running_servers.add(config.id)

    async def stop_server(self, config_id: UUID) -> None:
        """Stop server."""
        self.running_servers.discard(config_id)

    async def get_server_status(self, config_id: UUID) -> str:
        """Get server status."""
        return "running" if config_id in self.running_servers else "stopped"

    async def test_tool(
        self, config_id: UUID, tool_name: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Test tool."""
        return {"result": "success", "tool": tool_name, "params": parameters}

    async def list_tools(self, config_id: UUID) -> list[Tool]:
        """List tools."""
        return [
            Tool(name="test-tool", description="Test tool", parameters=[]),
        ]


class MockClientPublisher(ClientPublisher):
    """Mock client publisher for testing."""

    def __init__(self, client_type: str) -> None:
        """Initialize mock publisher."""
        self._client_type = client_type
        self.published_configs: dict[UUID, ClientConfig] = {}

    async def publish_config(self, config: ClientConfig) -> None:
        """Publish config."""
        self.published_configs[config.client.id] = config

    async def get_current_config(self, client: Client) -> ClientConfig | None:
        """Get current config."""
        return self.published_configs.get(client.id)

    def supports_client_type(self, client_type: str) -> bool:
        """Check if supports client type."""
        return client_type == self._client_type


# Tests for RegistryService


class TestRegistryService:
    """Tests for RegistryService."""

    @pytest.mark.asyncio
    async def test_list_available_registries(self) -> None:
        """Test listing available registries."""
        mock_reader = MockRegistryReader()
        service = RegistryService(mock_reader)

        registries = await service.list_available_registries()

        assert len(registries) == 1
        assert registries[0].name == "Test Registry"

    @pytest.mark.asyncio
    async def test_browse_servers(self) -> None:
        """Test browsing servers."""
        mock_reader = MockRegistryReader()
        service = RegistryService(mock_reader)

        servers = await service.browse_servers()

        assert len(servers) == 2
        assert servers[0].name == "test-server-1"
        assert servers[1].name == "test-server-2"

    @pytest.mark.asyncio
    async def test_search_servers(self) -> None:
        """Test searching servers."""
        mock_reader = MockRegistryReader()
        service = RegistryService(mock_reader)

        results = await service.search_servers("server-1")

        assert len(results) == 1
        assert results[0].name == "test-server-1"


# Tests for ServerManagementService


class TestServerManagementService:
    """Tests for ServerManagementService."""

    @pytest.mark.asyncio
    async def test_create_server_config(self) -> None:
        """Test creating a server config."""
        mock_store = MockConfigStore()
        service = ServerManagementService(mock_store)

        server = MCPServer(
            name="test-server",
            description="Test",
            repository=HttpUrl("https://github.com/test/server"),
            install_command="npm install",
            run_command="npx server",
        )

        config = await service.create_server_config(
            server, "my-server", {"API_KEY": "secret"}
        )

        assert config.name == "my-server"
        assert config.environment["API_KEY"] == "secret"
        assert config.status == ServerStatus.STOPPED

    @pytest.mark.asyncio
    async def test_list_server_configs(self) -> None:
        """Test listing server configs."""
        mock_store = MockConfigStore()
        service = ServerManagementService(mock_store)

        server = MCPServer(
            name="test-server",
            description="Test",
            repository=HttpUrl("https://github.com/test/server"),
            install_command="npm install",
            run_command="npx server",
        )

        await service.create_server_config(server, "server-1")
        await service.create_server_config(server, "server-2")

        configs = await service.list_server_configs()

        assert len(configs) == 2


# Tests for ServerExecutionService


class TestServerExecutionService:
    """Tests for ServerExecutionService."""

    @pytest.mark.asyncio
    async def test_start_server(self) -> None:
        """Test starting a server."""
        mock_executor = MockServerExecutor()
        mock_store = MockConfigStore()
        service = ServerExecutionService(mock_executor, mock_store)

        server = MCPServer(
            name="test-server",
            description="Test",
            repository=HttpUrl("https://github.com/test/server"),
            install_command="npm install",
            run_command="npx server",
        )

        config = ServerConfig(server=server, name="test")
        await mock_store.save_server_config(config)

        await service.start_server(config.id)

        updated_config = await mock_store.get_server_config(config.id)
        assert updated_config is not None
        assert updated_config.status == ServerStatus.RUNNING

    @pytest.mark.asyncio
    async def test_stop_server(self) -> None:
        """Test stopping a server."""
        mock_executor = MockServerExecutor()
        mock_store = MockConfigStore()
        service = ServerExecutionService(mock_executor, mock_store)

        server = MCPServer(
            name="test-server",
            description="Test",
            repository=HttpUrl("https://github.com/test/server"),
            install_command="npm install",
            run_command="npx server",
        )

        config = ServerConfig(server=server, name="test", status=ServerStatus.RUNNING)
        await mock_store.save_server_config(config)

        await service.stop_server(config.id)

        updated_config = await mock_store.get_server_config(config.id)
        assert updated_config is not None
        assert updated_config.status == ServerStatus.STOPPED


# Tests for ClientConfigService


class TestClientConfigService:
    """Tests for ClientConfigService."""

    @pytest.mark.asyncio
    async def test_create_client(self) -> None:
        """Test creating a client."""
        mock_store = MockConfigStore()
        mock_publisher = MockClientPublisher("claude_code")
        service = ClientConfigService(mock_store, [mock_publisher])

        client = await service.create_client(
            "My Client",
            "claude_code",
            "/path/to/config",
            "Test client",
        )

        assert client.name == "My Client"
        assert client.type == "claude_code"

    @pytest.mark.asyncio
    async def test_publish_configuration(self) -> None:
        """Test publishing a configuration."""
        mock_store = MockConfigStore()
        mock_publisher = MockClientPublisher("claude_code")
        service = ClientConfigService(mock_store, [mock_publisher])

        client = Client(
            name="Test",
            type=ClientType.CLAUDE_CODE,
            config_path="/path",
        )

        config = ClientConfig(client=client, server_configs=[])

        await service.publish_configuration(config)

        published = mock_publisher.published_configs.get(client.id)
        assert published is not None


# Tests for ConfigVersionService


class TestConfigVersionService:
    """Tests for ConfigVersionService."""

    @pytest.mark.asyncio
    async def test_save_version(self) -> None:
        """Test saving a config version."""
        mock_store = MockConfigStore()
        service = ConfigVersionService(mock_store)

        client = Client(
            name="Test",
            type=ClientType.CLAUDE_CODE,
            config_path="/path",
        )

        config = ClientConfig(client=client, server_configs=[])

        version = await service.save_version(client, config, "Initial version")

        assert version.version == 1
        assert version.description == "Initial version"

    @pytest.mark.asyncio
    async def test_rollback_to_version(self) -> None:
        """Test rolling back to a previous version."""
        mock_store = MockConfigStore()
        service = ConfigVersionService(mock_store)

        client = Client(
            name="Test",
            type=ClientType.CLAUDE_CODE,
            config_path="/path",
        )

        config = ClientConfig(client=client, server_configs=[])

        v1 = await service.save_version(client, config, "Version 1")
        v2 = await service.save_version(client, config, "Version 2")

        rollback = await service.rollback_to_version(v1.id, "Rollback to v1")

        assert rollback.is_rollback is True
        assert rollback.version == 3  # New version number
