"""File system based configuration store implementation."""

import json
from pathlib import Path
from typing import Any
from uuid import UUID

import aiofiles

from mcp_manager.domain.interfaces import ConfigStore
from mcp_manager.domain.models import Client, ConfigVersion, ServerConfig


class FileSystemConfigStore(ConfigStore):
    """Store configurations on the local file system using JSON files."""

    def __init__(self, base_path: Path | str) -> None:
        """Initialize the file system config store.

        Args:
            base_path: Base directory for storing configurations.
        """
        self._base_path = Path(base_path)
        self._servers_path = self._base_path / "servers"
        self._clients_path = self._base_path / "clients"
        self._versions_path = self._base_path / "versions"

        # Ensure directories exist
        self._servers_path.mkdir(parents=True, exist_ok=True)
        self._clients_path.mkdir(parents=True, exist_ok=True)
        self._versions_path.mkdir(parents=True, exist_ok=True)

    async def save_server_config(self, config: ServerConfig) -> None:
        """Save a server configuration."""
        file_path = self._servers_path / f"{config.id}.json"
        async with aiofiles.open(file_path, "w") as f:
            await f.write(config.model_dump_json(indent=2))

    async def get_server_config(self, config_id: UUID) -> ServerConfig | None:
        """Get a server configuration by ID."""
        file_path = self._servers_path / f"{config_id}.json"
        if not file_path.exists():
            return None

        async with aiofiles.open(file_path, "r") as f:
            data = await f.read()
            return ServerConfig.model_validate_json(data)

    async def list_server_configs(self) -> list[ServerConfig]:
        """List all server configurations."""
        configs = []
        for file_path in self._servers_path.glob("*.json"):
            async with aiofiles.open(file_path, "r") as f:
                data = await f.read()
                configs.append(ServerConfig.model_validate_json(data))
        return configs

    async def delete_server_config(self, config_id: UUID) -> None:
        """Delete a server configuration."""
        file_path = self._servers_path / f"{config_id}.json"
        if file_path.exists() and file_path.is_file():
            file_path.unlink()

    async def save_client(self, client: Client) -> None:
        """Save a client."""
        file_path = self._clients_path / f"{client.id}.json"
        async with aiofiles.open(file_path, "w") as f:
            await f.write(client.model_dump_json(indent=2))

    async def get_client(self, client_id: UUID) -> Client | None:
        """Get a client by ID."""
        file_path = self._clients_path / f"{client_id}.json"
        if not file_path.exists():
            return None

        async with aiofiles.open(file_path, "r") as f:
            data = await f.read()
            return Client.model_validate_json(data)

    async def list_clients(self) -> list[Client]:
        """List all clients."""
        clients = []
        for file_path in self._clients_path.glob("*.json"):
            async with aiofiles.open(file_path, "r") as f:
                data = await f.read()
                clients.append(Client.model_validate_json(data))
        return clients

    async def save_config_version(self, version: ConfigVersion) -> None:
        """Save a configuration version."""
        # Create client-specific directory
        client_dir = self._versions_path / str(version.client.id)
        client_dir.mkdir(parents=True, exist_ok=True)

        file_path = client_dir / f"{version.id}.json"
        async with aiofiles.open(file_path, "w") as f:
            await f.write(version.model_dump_json(indent=2))

    async def get_config_history(self, client_id: UUID) -> list[ConfigVersion]:
        """Get configuration history for a client."""
        client_dir = self._versions_path / str(client_id)
        if not client_dir.exists():
            return []

        versions = []
        for file_path in client_dir.glob("*.json"):
            async with aiofiles.open(file_path, "r") as f:
                data = await f.read()
                versions.append(ConfigVersion.model_validate_json(data))

        # Sort by version number, newest first
        return sorted(versions, key=lambda v: v.version, reverse=True)

    async def get_config_version(self, version_id: UUID) -> ConfigVersion | None:
        """Get a specific configuration version."""
        # Need to search across all client directories
        for client_dir in self._versions_path.iterdir():
            if not client_dir.is_dir():
                continue

            file_path = client_dir / f"{version_id}.json"
            if file_path.exists():
                async with aiofiles.open(file_path, "r") as f:
                    data = await f.read()
                    return ConfigVersion.model_validate_json(data)

        return None
