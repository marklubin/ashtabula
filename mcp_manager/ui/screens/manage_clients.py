"""Manage MCP clients and publish configurations screen."""

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Checkbox, DataTable, Label, Static

from mcp_manager.domain.models import Client, ServerConfig
from mcp_manager.services import ClientConfigService


class ManageClientsScreen(Container):
    """Screen for managing clients and publishing configurations."""

    def __init__(self, client_config: ClientConfigService) -> None:
        """Initialize the manage clients screen.

        Args:
            client_config: Client config service instance.
        """
        super().__init__()
        self.client_config = client_config
        self.clients: list[Client] = []
        self.selected_client: Client | None = None
        self.server_configs: list[ServerConfig] = []
        self.selected_servers: set[str] = set()

    def compose(self) -> ComposeResult:
        """Compose the manage clients UI."""
        with Vertical():
            yield Label("Manage Clients & Publish Configurations", classes="title")

            with Horizontal():
                yield Button("Add Client", id="btn-add-client", variant="primary")
                yield Button("Refresh", id="btn-refresh-clients")

            with Horizontal():
                with Container(id="clients-list"):
                    yield Label("Configured Clients")
                    table = DataTable(id="clients-table")
                    table.add_columns("Name", "Type", "Config Path")
                    yield table

                with Container(id="client-details"):
                    yield Label("Client Details")
                    yield Static(id="client-info")

            with Container(id="config-builder"):
                yield Label("Build Configuration")
                yield Label("Select servers to include:")
                yield DataTable(id="servers-checklist")
                yield Button(
                    "Publish Configuration",
                    id="btn-publish",
                    variant="success",
                )

    async def on_mount(self) -> None:
        """Load clients on mount."""
        await self._load_clients()
        await self._load_servers()

    async def _load_clients(self) -> None:
        """Load configured clients."""
        try:
            self.clients = await self.client_config.list_clients()
            await self._update_clients_table()
        except Exception as e:
            self.app.notify(f"Failed to load clients: {e}", severity="error")

    async def _update_clients_table(self) -> None:
        """Update the clients table."""
        table = self.query_one("#clients-table", DataTable)
        table.clear()

        for client in self.clients:
            table.add_row(
                client.name,
                client.type,
                client.config_path[:40],
                key=str(client.id),
            )

    async def _load_servers(self) -> None:
        """Load available server configs."""
        try:
            # Get server configs from the store
            from mcp_manager.adapters import FileSystemConfigStore
            from pathlib import Path

            config_dir = Path.home() / ".mcp-manager"
            store = FileSystemConfigStore(config_dir / "data")
            self.server_configs = await store.list_server_configs()

            await self._update_servers_checklist()
        except Exception as e:
            self.app.notify(f"Failed to load servers: {e}", severity="error")

    async def _update_servers_checklist(self) -> None:
        """Update the servers checklist."""
        checklist = self.query_one("#servers-checklist", DataTable)
        checklist.clear()
        checklist.add_columns("Select", "Server", "Status")

        for config in self.server_configs:
            is_selected = "☑" if str(config.id) in self.selected_servers else "☐"
            checklist.add_row(
                is_selected,
                config.name,
                config.status.value,
                key=str(config.id),
            )

    @on(Button.Pressed, "#btn-refresh-clients")
    async def refresh_clients(self) -> None:
        """Handle refresh button press."""
        await self._load_clients()
        await self._load_servers()
        self.app.notify("Clients refreshed", severity="information")

    @on(DataTable.RowSelected, "#clients-table")
    async def client_selected(self, event: DataTable.RowSelected) -> None:
        """Handle client selection."""
        from uuid import UUID

        client_id = UUID(str(event.row_key))
        self.selected_client = await self.client_config.get_client(client_id)
        await self._update_client_info()

    async def _update_client_info(self) -> None:
        """Update client info panel."""
        if not self.selected_client:
            return

        info_widget = self.query_one("#client-info", Static)

        info = f"""
**{self.selected_client.name}**

**Type:** {self.selected_client.type}
**Config Path:** {self.selected_client.config_path}

{self.selected_client.description}
"""
        info_widget.update(info)

    @on(DataTable.RowSelected, "#servers-checklist")
    async def toggle_server_selection(self, event: DataTable.RowSelected) -> None:
        """Handle server selection toggle."""
        server_id = str(event.row_key)

        if server_id in self.selected_servers:
            self.selected_servers.remove(server_id)
        else:
            self.selected_servers.add(server_id)

        await self._update_servers_checklist()

    @on(Button.Pressed, "#btn-publish")
    async def publish_configuration(self) -> None:
        """Handle publish button press."""
        if not self.selected_client:
            self.app.notify("No client selected", severity="warning")
            return

        if not self.selected_servers:
            self.app.notify("No servers selected", severity="warning")
            return

        try:
            # Build configuration
            from uuid import UUID

            selected_configs = [
                config
                for config in self.server_configs
                if str(config.id) in self.selected_servers
            ]

            client_config = await self.client_config.create_client_config(
                self.selected_client, selected_configs
            )

            # Publish
            await self.client_config.publish_configuration(client_config)

            self.app.notify(
                f"Configuration published to {self.selected_client.name}",
                severity="information",
            )

        except Exception as e:
            self.app.notify(f"Failed to publish configuration: {e}", severity="error")
