"""Integration tests for end-to-end workflows."""

import tempfile
from pathlib import Path

import pytest
from pydantic import HttpUrl

from mcp_manager.adapters import (
    ClaudeCodePublisher,
    FileSystemConfigStore,
)
from mcp_manager.domain.models import (
    Client,
    ClientType,
    EnvironmentVariable,
    MCPServer,
    ServerConfig,
)
from mcp_manager.services import (
    ClientConfigService,
    ConfigVersionService,
    ServerManagementService,
)


class TestServerLifecycle:
    """Test complete server configuration lifecycle."""

    @pytest.mark.asyncio
    async def test_create_and_manage_server_config(self) -> None:
        """Test creating and managing a server configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_store = FileSystemConfigStore(Path(tmpdir))
            service = ServerManagementService(config_store)

            # Create a server with environment variables
            server = MCPServer(
                name="test-server",
                description="Test MCP Server",
                repository=HttpUrl("https://github.com/test/server"),
                install_command="npm install -g test-server",
                run_command="npx test-server",
                environment_variables=[
                    EnvironmentVariable(
                        name="API_KEY",
                        description="API Key",
                        required=True,
                    ),
                    EnvironmentVariable(
                        name="PORT",
                        description="Server port",
                        required=False,
                        default="3000",
                    ),
                ],
            )

            # Create configuration
            config = await service.create_server_config(
                server,
                "my-test-server",
                {"API_KEY": "test-key-123", "PORT": "8080"},
            )

            assert config.name == "my-test-server"
            assert config.environment["API_KEY"] == "test-key-123"

            # Retrieve configuration
            retrieved = await service.get_server_config(config.id)
            assert retrieved is not None
            assert retrieved.name == "my-test-server"

            # List configurations
            configs = await service.list_server_configs()
            assert len(configs) == 1

            # Update configuration
            updated = await service.update_server_config(
                config.id,
                name="updated-server",
                environment={"API_KEY": "new-key", "PORT": "9000"},
            )

            assert updated.name == "updated-server"
            assert updated.environment["API_KEY"] == "new-key"

            # Delete configuration
            await service.delete_server_config(config.id)
            configs = await service.list_server_configs()
            assert len(configs) == 0


class TestClientConfigurationPublishing:
    """Test client configuration publishing workflow."""

    @pytest.mark.asyncio
    async def test_publish_to_claude_code(self) -> None:
        """Test publishing configuration to Claude Code CLI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config_store = FileSystemConfigStore(config_dir / "data")

            # Create client config file path
            client_config_path = config_dir / "claude-code-config.json"

            # Create publisher
            publisher = ClaudeCodePublisher()

            # Create client
            client = Client(
                name="Claude Code Test",
                type=ClientType.CLAUDE_CODE,
                config_path=str(client_config_path),
            )

            # Create server config
            server = MCPServer(
                name="test-server",
                description="Test",
                repository=HttpUrl("https://github.com/test/server"),
                install_command="npm install",
                run_command="npx test-server",
            )

            server_config = ServerConfig(
                server=server,
                name="my-server",
                environment={"API_KEY": "secret"},
            )

            # Create client config
            from mcp_manager.domain.models import ClientConfig

            client_config = ClientConfig(
                client=client,
                server_configs=[server_config],
            )

            # Publish
            await publisher.publish_config(client_config)

            # Verify file was created
            assert client_config_path.exists()

            # Verify file contents
            import json

            with open(client_config_path) as f:
                data = json.load(f)

            assert "mcpServers" in data
            assert "my-server" in data["mcpServers"]
            assert data["mcpServers"]["my-server"]["env"]["API_KEY"] == "secret"


class TestConfigVersioning:
    """Test configuration versioning and rollback."""

    @pytest.mark.asyncio
    async def test_version_history_and_rollback(self) -> None:
        """Test creating version history and rolling back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_store = FileSystemConfigStore(Path(tmpdir))
            version_service = ConfigVersionService(config_store)
            server_management = ServerManagementService(config_store)

            # Create client
            client = Client(
                name="Test Client",
                type=ClientType.CLAUDE_CODE,
                config_path="/path/to/config",
            )
            await config_store.save_client(client)

            # Create first server config
            server1 = MCPServer(
                name="server-1",
                description="Server 1",
                repository=HttpUrl("https://github.com/test/server1"),
                install_command="npm install",
                run_command="npx server1",
            )
            config1 = await server_management.create_server_config(server1, "server-1")

            # Create first version
            from mcp_manager.domain.models import ClientConfig

            client_config_v1 = ClientConfig(
                client=client,
                server_configs=[config1],
            )
            v1 = await version_service.save_version(
                client, client_config_v1, "Initial configuration"
            )

            assert v1.version == 1

            # Create second server config
            server2 = MCPServer(
                name="server-2",
                description="Server 2",
                repository=HttpUrl("https://github.com/test/server2"),
                install_command="npm install",
                run_command="npx server2",
            )
            config2 = await server_management.create_server_config(server2, "server-2")

            # Create second version with both servers
            client_config_v2 = ClientConfig(
                client=client,
                server_configs=[config1, config2],
            )
            v2 = await version_service.save_version(
                client, client_config_v2, "Added server-2"
            )

            assert v2.version == 2

            # Get history
            history = await version_service.get_version_history(client.id)
            assert len(history) == 2
            assert history[0].version == 2  # Newest first
            assert history[1].version == 1

            # Rollback to v1
            rollback = await version_service.rollback_to_version(
                v1.id, "Rollback to initial config"
            )

            assert rollback.is_rollback is True
            assert rollback.version == 3
            assert len(rollback.config.server_configs) == 1

            # Verify history now has 3 versions
            history = await version_service.get_version_history(client.id)
            assert len(history) == 3

            # Compare versions
            comparison = await version_service.compare_versions(v1.id, v2.id)
            assert len(comparison["added_servers"]) == 1
            assert comparison["added_servers"][0].name == "server-2"


class TestFullWorkflow:
    """Test complete end-to-end workflow."""

    @pytest.mark.asyncio
    async def test_complete_workflow(self) -> None:
        """Test complete workflow from server creation to publishing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config_store = FileSystemConfigStore(config_dir / "data")

            # Setup services
            server_management = ServerManagementService(config_store)
            publishers = [ClaudeCodePublisher()]
            client_config_service = ClientConfigService(config_store, publishers)
            version_service = ConfigVersionService(config_store)

            # 1. Create MCP server
            server = MCPServer(
                name="weather-server",
                description="Weather information server",
                repository=HttpUrl("https://github.com/test/weather"),
                install_command="npm install -g weather-server",
                run_command="npx weather-server",
                environment_variables=[
                    EnvironmentVariable(
                        name="WEATHER_API_KEY",
                        description="Weather API key",
                        required=True,
                    )
                ],
            )

            # 2. Configure server instance
            server_config = await server_management.create_server_config(
                server,
                "my-weather-server",
                {"WEATHER_API_KEY": "test-api-key"},
            )

            # 3. Create client
            client_config_path = config_dir / "client-config.json"
            client = await client_config_service.create_client(
                "My Claude Code",
                "claude_code",
                str(client_config_path),
                "My local Claude Code instance",
            )

            # 4. Build client configuration
            from mcp_manager.domain.models import ClientConfig

            client_config = ClientConfig(
                client=client,
                server_configs=[server_config],
            )

            # 5. Save version
            version = await version_service.save_version(
                client,
                client_config,
                "Initial weather server configuration",
            )

            assert version.version == 1

            # 6. Publish configuration
            await client_config_service.publish_configuration(client_config)

            # 7. Verify published file
            assert client_config_path.exists()

            import json

            with open(client_config_path) as f:
                data = json.load(f)

            assert "mcpServers" in data
            assert "my-weather-server" in data["mcpServers"]
            assert (
                data["mcpServers"]["my-weather-server"]["env"]["WEATHER_API_KEY"]
                == "test-api-key"
            )

            # 8. Verify version history
            history = await version_service.get_version_history(client.id)
            assert len(history) == 1
            assert history[0].description == "Initial weather server configuration"
