"""Core domain models for MCP Manager."""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class ServerStatus(str, Enum):
    """Status of an MCP server instance."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"


class ClientType(str, Enum):
    """Supported client types."""

    CLAUDE_CODE = "claude_code"
    CLAUDE_DESKTOP = "claude_desktop"
    CUSTOM = "custom"


class EnvironmentVariable(BaseModel):
    """Environment variable definition for MCP server configuration."""

    name: str = Field(..., description="Environment variable name")
    description: str = Field(..., description="Description of the variable")
    required: bool = Field(default=False, description="Whether the variable is required")
    default: str | None = Field(
        default=None, description="Default value if not provided"
    )
    value: str | None = Field(default=None, description="Configured value")

    model_config = ConfigDict(frozen=False)


class ToolParameter(BaseModel):
    """Parameter definition for an MCP tool."""

    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type")
    description: str = Field(..., description="Parameter description")
    required: bool = Field(default=False, description="Whether parameter is required")
    default: Any | None = Field(default=None, description="Default value")

    model_config = ConfigDict(frozen=True)


class Tool(BaseModel):
    """MCP tool definition."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    parameters: list[ToolParameter] = Field(
        default_factory=list, description="Tool parameters"
    )

    model_config = ConfigDict(frozen=True)


class ServerRegistry(BaseModel):
    """Registry information for MCP servers."""

    name: str = Field(..., description="Registry name")
    url: HttpUrl = Field(..., description="Registry URL")
    trusted: bool = Field(default=False, description="Whether registry is trusted")

    model_config = ConfigDict(frozen=True)


class MCPServer(BaseModel):
    """MCP Server definition from a registry."""

    id: UUID = Field(default_factory=uuid4, description="Unique server ID")
    name: str = Field(..., description="Server name")
    description: str = Field(..., description="Server description")
    repository: HttpUrl = Field(..., description="Server repository URL")
    registry: ServerRegistry | None = Field(
        default=None, description="Registry source"
    )
    install_command: str = Field(
        ..., description="Command to install the server (e.g., npm install -g @package)"
    )
    run_command: str = Field(
        ..., description="Command to run the server (e.g., npx @package)"
    )
    environment_variables: list[EnvironmentVariable] = Field(
        default_factory=list, description="Environment variables"
    )
    tools: list[Tool] = Field(default_factory=list, description="Available tools")
    version: str = Field(default="latest", description="Server version")
    tags: list[str] = Field(default_factory=list, description="Server tags/categories")

    model_config = ConfigDict(frozen=False)


class ServerConfig(BaseModel):
    """Configuration for a running MCP server instance."""

    id: UUID = Field(default_factory=uuid4, description="Unique config ID")
    server: MCPServer = Field(..., description="MCP Server definition")
    name: str = Field(..., description="Instance name (user-defined)")
    status: ServerStatus = Field(
        default=ServerStatus.STOPPED, description="Current status"
    )
    environment: dict[str, str] = Field(
        default_factory=dict, description="Configured environment variables"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    last_started_at: datetime | None = Field(
        default=None, description="Last start timestamp"
    )
    error_message: str | None = Field(default=None, description="Last error message")

    model_config = ConfigDict(frozen=False)


class Client(BaseModel):
    """Client application that can use MCP servers."""

    id: UUID = Field(default_factory=uuid4, description="Unique client ID")
    name: str = Field(..., description="Client name")
    type: ClientType = Field(..., description="Client type")
    config_path: str = Field(..., description="Path to client configuration file")
    description: str = Field(default="", description="Client description")

    model_config = ConfigDict(frozen=False)


class ClientConfig(BaseModel):
    """Configuration published to a client."""

    id: UUID = Field(default_factory=uuid4, description="Unique config ID")
    client: Client = Field(..., description="Target client")
    server_configs: list[ServerConfig] = Field(
        default_factory=list, description="Server configurations to publish"
    )
    published_at: datetime = Field(
        default_factory=datetime.now, description="Publication timestamp"
    )
    active: bool = Field(default=True, description="Whether config is active")

    model_config = ConfigDict(frozen=False)


class ConfigVersion(BaseModel):
    """Version history for client configurations."""

    id: UUID = Field(default_factory=uuid4, description="Unique version ID")
    client: Client = Field(..., description="Target client")
    config: ClientConfig = Field(..., description="Configuration snapshot")
    version: int = Field(..., description="Version number")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Version timestamp"
    )
    description: str = Field(default="", description="Version description")
    is_rollback: bool = Field(
        default=False, description="Whether this is a rollback version"
    )

    model_config = ConfigDict(frozen=True)
