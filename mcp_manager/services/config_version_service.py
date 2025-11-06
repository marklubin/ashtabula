"""Service for managing configuration versions and rollbacks."""

from uuid import UUID

from mcp_manager.domain.interfaces import ConfigStore
from mcp_manager.domain.models import Client, ClientConfig, ConfigVersion


class ConfigVersionService:
    """Service for tracking configuration versions and enabling rollbacks."""

    def __init__(self, config_store: ConfigStore) -> None:
        """Initialize the config version service.

        Args:
            config_store: Configuration store implementation.
        """
        self._config_store = config_store

    async def save_version(
        self, client: Client, config: ClientConfig, description: str = ""
    ) -> ConfigVersion:
        """Save a new configuration version.

        Args:
            client: Target client.
            config: Configuration to save.
            description: Version description.

        Returns:
            Created configuration version.
        """
        # Get current version number
        history = await self._config_store.get_config_history(client.id)
        version_number = len(history) + 1

        version = ConfigVersion(
            client=client,
            config=config,
            version=version_number,
            description=description,
            is_rollback=False,
        )

        await self._config_store.save_config_version(version)
        return version

    async def get_version_history(self, client_id: UUID) -> list[ConfigVersion]:
        """Get version history for a client.

        Args:
            client_id: Client ID.

        Returns:
            List of configuration versions, newest first.
        """
        return await self._config_store.get_config_history(client_id)

    async def get_version(self, version_id: UUID) -> ConfigVersion | None:
        """Get a specific configuration version.

        Args:
            version_id: Version ID.

        Returns:
            Configuration version or None if not found.
        """
        return await self._config_store.get_config_version(version_id)

    async def rollback_to_version(
        self, version_id: UUID, description: str = ""
    ) -> ConfigVersion:
        """Rollback to a previous configuration version.

        Args:
            version_id: Version ID to rollback to.
            description: Rollback description.

        Returns:
            New rollback version.

        Raises:
            ValueError: If version not found.
        """
        source_version = await self._config_store.get_config_version(version_id)
        if not source_version:
            raise ValueError(f"Version {version_id} not found")

        # Get current version number
        history = await self._config_store.get_config_history(
            source_version.client.id
        )
        version_number = len(history) + 1

        # Create rollback version
        rollback_version = ConfigVersion(
            client=source_version.client,
            config=source_version.config,
            version=version_number,
            description=description
            or f"Rollback to version {source_version.version}",
            is_rollback=True,
        )

        await self._config_store.save_config_version(rollback_version)
        return rollback_version

    async def compare_versions(
        self, version_id_1: UUID, version_id_2: UUID
    ) -> dict[str, any]:
        """Compare two configuration versions.

        Args:
            version_id_1: First version ID.
            version_id_2: Second version ID.

        Returns:
            Comparison result with differences.

        Raises:
            ValueError: If either version not found.
        """
        v1 = await self._config_store.get_config_version(version_id_1)
        v2 = await self._config_store.get_config_version(version_id_2)

        if not v1:
            raise ValueError(f"Version {version_id_1} not found")
        if not v2:
            raise ValueError(f"Version {version_id_2} not found")

        # Compare server configurations
        v1_servers = {s.id: s for s in v1.config.server_configs}
        v2_servers = {s.id: s for s in v2.config.server_configs}

        added = [s for sid, s in v2_servers.items() if sid not in v1_servers]
        removed = [s for sid, s in v1_servers.items() if sid not in v2_servers]
        modified = []

        for sid in v1_servers:
            if sid in v2_servers:
                if v1_servers[sid] != v2_servers[sid]:
                    modified.append((v1_servers[sid], v2_servers[sid]))

        return {
            "version_1": v1,
            "version_2": v2,
            "added_servers": added,
            "removed_servers": removed,
            "modified_servers": modified,
        }
