"""Service for managing MCP server registries and discovery."""

from uuid import UUID

from mcp_manager.domain.interfaces import ServerRegistryReader
from mcp_manager.domain.models import MCPServer, ServerRegistry


class RegistryService:
    """Service for browsing and discovering MCP servers from registries."""

    def __init__(self, registry_reader: ServerRegistryReader) -> None:
        """Initialize the registry service.

        Args:
            registry_reader: Registry reader implementation.
        """
        self._registry_reader = registry_reader
        self._cache: dict[UUID, MCPServer] = {}

    async def list_available_registries(self) -> list[ServerRegistry]:
        """List all available MCP server registries.

        Returns:
            List of server registries.
        """
        return await self._registry_reader.list_registries()

    async def browse_servers(
        self, registry: ServerRegistry | None = None
    ) -> list[MCPServer]:
        """Browse available MCP servers.

        Args:
            registry: Specific registry to browse, or None for all registries.

        Returns:
            List of available MCP servers.
        """
        if registry:
            servers = await self._registry_reader.list_servers(registry)
        else:
            # Get servers from all registries
            registries = await self.list_available_registries()
            servers = []
            for reg in registries:
                reg_servers = await self._registry_reader.list_servers(reg)
                servers.extend(reg_servers)

        # Cache servers
        for server in servers:
            self._cache[server.id] = server

        return servers

    async def get_server_details(self, server_id: UUID) -> MCPServer | None:
        """Get detailed information about a specific server.

        Args:
            server_id: The server ID.

        Returns:
            Server details or None if not found.
        """
        # Check cache first
        if server_id in self._cache:
            return self._cache[server_id]

        # Fallback to registry reader
        server = await self._registry_reader.get_server(server_id)
        if server:
            self._cache[server_id] = server

        return server

    async def search_servers(self, query: str) -> list[MCPServer]:
        """Search for servers by name, description, or tags.

        Args:
            query: Search query.

        Returns:
            List of matching servers.
        """
        all_servers = await self.browse_servers()
        query_lower = query.lower()

        return [
            server
            for server in all_servers
            if query_lower in server.name.lower()
            or query_lower in server.description.lower()
            or any(query_lower in tag.lower() for tag in server.tags)
        ]

    def clear_cache(self) -> None:
        """Clear the server cache."""
        self._cache.clear()
