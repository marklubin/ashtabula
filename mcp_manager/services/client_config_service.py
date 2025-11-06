"""Service for managing client configurations."""

from uuid import UUID

from mcp_manager.domain.interfaces import ClientPublisher, ConfigStore
from mcp_manager.domain.models import Client, ClientConfig, ServerConfig


class ClientConfigService:
    """Service for managing and publishing client configurations."""

    def __init__(
        self, config_store: ConfigStore, publishers: list[ClientPublisher]
    ) -> None:
        """Initialize the client config service.

        Args:
            config_store: Configuration store implementation.
            publishers: List of client publisher implementations.
        """
        self._config_store = config_store
        self._publishers = {
            pub: pub for pub in publishers
        }  # Map publishers to themselves for lookup

    async def create_client(
        self, name: str, client_type: str, config_path: str, description: str = ""
    ) -> Client:
        """Create a new client.

        Args:
            name: Client name.
            client_type: Client type.
            config_path: Path to client configuration file.
            description: Client description.

        Returns:
            Created client.
        """
        client = Client(
            name=name, type=client_type, config_path=config_path, description=description
        )
        await self._config_store.save_client(client)
        return client

    async def list_clients(self) -> list[Client]:
        """List all configured clients.

        Returns:
            List of clients.
        """
        return await self._config_store.list_clients()

    async def get_client(self, client_id: UUID) -> Client | None:
        """Get a client by ID.

        Args:
            client_id: Client ID.

        Returns:
            Client or None if not found.
        """
        return await self._config_store.get_client(client_id)

    async def create_client_config(
        self, client: Client, server_configs: list[ServerConfig]
    ) -> ClientConfig:
        """Create a client configuration.

        Args:
            client: Target client.
            server_configs: Server configurations to include.

        Returns:
            Created client configuration.
        """
        config = ClientConfig(client=client, server_configs=server_configs)
        return config

    async def publish_configuration(self, config: ClientConfig) -> None:
        """Publish configuration to a client.

        Args:
            config: Client configuration to publish.

        Raises:
            ValueError: If no publisher supports the client type.
            RuntimeError: If publishing fails.
        """
        # Find appropriate publisher
        publisher = None
        for pub in self._publishers:
            if pub.supports_client_type(config.client.type):
                publisher = pub
                break

        if not publisher:
            raise ValueError(
                f"No publisher available for client type: {config.client.type}"
            )

        try:
            await publisher.publish_config(config)
        except Exception as e:
            raise RuntimeError(f"Failed to publish configuration: {e}") from e

    async def get_current_config(self, client: Client) -> ClientConfig | None:
        """Get the current published configuration for a client.

        Args:
            client: The client.

        Returns:
            Current configuration or None.
        """
        # Find appropriate publisher
        for pub in self._publishers:
            if pub.supports_client_type(client.type):
                return await pub.get_current_config(client)

        return None

    def get_supported_client_types(self) -> list[str]:
        """Get list of supported client types.

        Returns:
            List of supported client type strings.
        """
        # Collect all supported types from all publishers
        types = set()
        for pub in self._publishers:
            # We'll need to enhance the interface to get supported types
            # For now, we'll use the known types
            if pub.supports_client_type("claude_code"):
                types.add("claude_code")
            if pub.supports_client_type("claude_desktop"):
                types.add("claude_desktop")

        return list(types)
