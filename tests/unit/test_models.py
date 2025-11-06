"""Unit tests for domain models."""

from datetime import datetime
from uuid import UUID

import pytest
from pydantic import HttpUrl

from mcp_manager.domain.models import (
    Client,
    ClientConfig,
    ClientType,
    ConfigVersion,
    EnvironmentVariable,
    MCPServer,
    ServerConfig,
    ServerRegistry,
    ServerStatus,
    Tool,
    ToolParameter,
)


class TestEnvironmentVariable:
    """Tests for EnvironmentVariable model."""

    def test_create_required_env_var(self) -> None:
        """Test creating a required environment variable."""
        env_var = EnvironmentVariable(
            name="API_KEY",
            description="API key for authentication",
            required=True,
        )

        assert env_var.name == "API_KEY"
        assert env_var.required is True
        assert env_var.value is None

    def test_create_optional_env_var_with_default(self) -> None:
        """Test creating optional env var with default value."""
        env_var = EnvironmentVariable(
            name="PORT",
            description="Server port",
            required=False,
            default="3000",
        )

        assert env_var.default == "3000"
        assert env_var.required is False

    def test_set_env_var_value(self) -> None:
        """Test setting environment variable value."""
        env_var = EnvironmentVariable(
            name="API_KEY",
            description="API key",
            required=True,
        )

        env_var.value = "secret-key-123"
        assert env_var.value == "secret-key-123"


class TestToolParameter:
    """Tests for ToolParameter model."""

    def test_create_required_parameter(self) -> None:
        """Test creating a required parameter."""
        param = ToolParameter(
            name="query",
            type="string",
            description="Search query",
            required=True,
        )

        assert param.name == "query"
        assert param.required is True

    def test_create_optional_parameter_with_default(self) -> None:
        """Test creating optional parameter with default."""
        param = ToolParameter(
            name="limit",
            type="integer",
            description="Result limit",
            required=False,
            default=10,
        )

        assert param.default == 10


class TestTool:
    """Tests for Tool model."""

    def test_create_tool_with_parameters(self) -> None:
        """Test creating a tool with parameters."""
        params = [
            ToolParameter(
                name="query",
                type="string",
                description="Search query",
                required=True,
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="Result limit",
                required=False,
                default=10,
            ),
        ]

        tool = Tool(
            name="search",
            description="Search for items",
            parameters=params,
        )

        assert tool.name == "search"
        assert len(tool.parameters) == 2
        assert tool.parameters[0].name == "query"


class TestServerRegistry:
    """Tests for ServerRegistry model."""

    def test_create_trusted_registry(self) -> None:
        """Test creating a trusted registry."""
        registry = ServerRegistry(
            name="Official Registry",
            url=HttpUrl("https://example.com/registry.json"),
            trusted=True,
        )

        assert registry.trusted is True
        assert registry.name == "Official Registry"


class TestMCPServer:
    """Tests for MCPServer model."""

    def test_create_mcp_server(self) -> None:
        """Test creating an MCP server."""
        registry = ServerRegistry(
            name="Test Registry",
            url=HttpUrl("https://example.com/registry.json"),
            trusted=True,
        )

        server = MCPServer(
            name="test-server",
            description="A test MCP server",
            repository=HttpUrl("https://github.com/test/server"),
            registry=registry,
            install_command="npm install -g test-server",
            run_command="npx test-server",
            version="1.0.0",
            tags=["test", "example"],
        )

        assert server.name == "test-server"
        assert isinstance(server.id, UUID)
        assert server.version == "1.0.0"
        assert "test" in server.tags

    def test_server_with_environment_variables(self) -> None:
        """Test server with environment variables."""
        env_vars = [
            EnvironmentVariable(
                name="API_KEY",
                description="API key",
                required=True,
            ),
            EnvironmentVariable(
                name="PORT",
                description="Port",
                required=False,
                default="3000",
            ),
        ]

        server = MCPServer(
            name="test-server",
            description="Test server",
            repository=HttpUrl("https://github.com/test/server"),
            install_command="npm install",
            run_command="npx server",
            environment_variables=env_vars,
        )

        assert len(server.environment_variables) == 2
        assert server.environment_variables[0].required is True


class TestServerConfig:
    """Tests for ServerConfig model."""

    def test_create_server_config(self) -> None:
        """Test creating a server configuration."""
        server = MCPServer(
            name="test-server",
            description="Test",
            repository=HttpUrl("https://github.com/test/server"),
            install_command="npm install",
            run_command="npx server",
        )

        config = ServerConfig(
            server=server,
            name="my-test-server",
            environment={"API_KEY": "secret"},
        )

        assert config.name == "my-test-server"
        assert config.status == ServerStatus.STOPPED
        assert config.environment["API_KEY"] == "secret"
        assert isinstance(config.id, UUID)

    def test_server_config_status_transitions(self) -> None:
        """Test server config status transitions."""
        server = MCPServer(
            name="test-server",
            description="Test",
            repository=HttpUrl("https://github.com/test/server"),
            install_command="npm install",
            run_command="npx server",
        )

        config = ServerConfig(server=server, name="test")

        assert config.status == ServerStatus.STOPPED

        config.status = ServerStatus.STARTING
        assert config.status == ServerStatus.STARTING

        config.status = ServerStatus.RUNNING
        assert config.status == ServerStatus.RUNNING


class TestClient:
    """Tests for Client model."""

    def test_create_claude_code_client(self) -> None:
        """Test creating a Claude Code client."""
        client = Client(
            name="My Claude Code",
            type=ClientType.CLAUDE_CODE,
            config_path="~/.config/claude-code/mcp.json",
            description="My local Claude Code CLI",
        )

        assert client.type == ClientType.CLAUDE_CODE
        assert client.name == "My Claude Code"

    def test_create_claude_desktop_client(self) -> None:
        """Test creating a Claude Desktop client."""
        client = Client(
            name="My Claude Desktop",
            type=ClientType.CLAUDE_DESKTOP,
            config_path="~/Library/Application Support/Claude/config.json",
        )

        assert client.type == ClientType.CLAUDE_DESKTOP


class TestClientConfig:
    """Tests for ClientConfig model."""

    def test_create_client_config(self) -> None:
        """Test creating a client configuration."""
        client = Client(
            name="Test Client",
            type=ClientType.CLAUDE_CODE,
            config_path="/path/to/config",
        )

        server = MCPServer(
            name="server",
            description="Test",
            repository=HttpUrl("https://github.com/test/server"),
            install_command="npm install",
            run_command="npx server",
        )

        server_config = ServerConfig(server=server, name="test")

        client_config = ClientConfig(
            client=client,
            server_configs=[server_config],
        )

        assert len(client_config.server_configs) == 1
        assert client_config.active is True


class TestConfigVersion:
    """Tests for ConfigVersion model."""

    def test_create_config_version(self) -> None:
        """Test creating a configuration version."""
        client = Client(
            name="Test Client",
            type=ClientType.CLAUDE_CODE,
            config_path="/path/to/config",
        )

        client_config = ClientConfig(client=client, server_configs=[])

        version = ConfigVersion(
            client=client,
            config=client_config,
            version=1,
            description="Initial version",
        )

        assert version.version == 1
        assert version.is_rollback is False
        assert isinstance(version.created_at, datetime)

    def test_create_rollback_version(self) -> None:
        """Test creating a rollback version."""
        client = Client(
            name="Test Client",
            type=ClientType.CLAUDE_CODE,
            config_path="/path/to/config",
        )

        client_config = ClientConfig(client=client, server_configs=[])

        version = ConfigVersion(
            client=client,
            config=client_config,
            version=2,
            description="Rollback to v1",
            is_rollback=True,
        )

        assert version.is_rollback is True
