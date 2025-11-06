"""Browse and discover MCP servers screen."""

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, DataTable, Input, Label, Static

from mcp_manager.domain.models import MCPServer
from mcp_manager.services import RegistryService, ServerManagementService


class BrowseServersScreen(Container):
    """Screen for browsing available MCP servers from registries."""

    def __init__(
        self,
        registry_service: RegistryService,
        server_management: ServerManagementService,
    ) -> None:
        """Initialize the browse servers screen.

        Args:
            registry_service: Registry service instance.
            server_management: Server management service instance.
        """
        super().__init__()
        self.registry_service = registry_service
        self.server_management = server_management
        self.servers: list[MCPServer] = []
        self.selected_server: MCPServer | None = None

    def compose(self) -> ComposeResult:
        """Compose the browse servers UI."""
        with Vertical():
            yield Label("Browse MCP Servers", classes="title")
            with Horizontal():
                yield Input(placeholder="Search servers...", id="search-input")
                yield Button("Search", id="btn-search", variant="primary")
                yield Button("Refresh", id="btn-refresh")

            with Horizontal():
                with Container(id="server-list"):
                    yield Label("Available Servers")
                    table = DataTable(id="servers-table")
                    table.add_columns("Name", "Description", "Tags")
                    yield table

                with Container(id="server-details"):
                    yield Label("Server Details")
                    yield Static(id="details-content")
                    yield Button(
                        "Install & Configure", id="btn-install", variant="success"
                    )

    async def on_mount(self) -> None:
        """Load servers on mount."""
        await self._load_servers()

    async def _load_servers(self) -> None:
        """Load servers from registries."""
        try:
            self.servers = await self.registry_service.browse_servers()
            await self._update_table()
        except Exception as e:
            self.app.notify(f"Failed to load servers: {e}", severity="error")

    async def _update_table(self, filter_text: str = "") -> None:
        """Update the servers table.

        Args:
            filter_text: Optional filter text for search.
        """
        table = self.query_one("#servers-table", DataTable)
        table.clear()

        filtered_servers = self.servers
        if filter_text:
            filtered_servers = [
                s
                for s in self.servers
                if filter_text.lower() in s.name.lower()
                or filter_text.lower() in s.description.lower()
            ]

        for server in filtered_servers:
            tags = ", ".join(server.tags[:3])  # Show first 3 tags
            table.add_row(server.name, server.description[:50], tags, key=str(server.id))

    @on(Button.Pressed, "#btn-search")
    async def search_servers(self) -> None:
        """Handle search button press."""
        search_input = self.query_one("#search-input", Input)
        await self._update_table(search_input.value)

    @on(Button.Pressed, "#btn-refresh")
    async def refresh_servers(self) -> None:
        """Handle refresh button press."""
        self.registry_service.clear_cache()
        await self._load_servers()
        self.app.notify("Servers refreshed", severity="information")

    @on(DataTable.RowSelected)
    async def row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection in servers table."""
        server_id = event.row_key
        if server_id:
            from uuid import UUID

            self.selected_server = await self.registry_service.get_server_details(
                UUID(str(server_id))
            )
            await self._update_details()

    async def _update_details(self) -> None:
        """Update server details panel."""
        if not self.selected_server:
            return

        details_widget = self.query_one("#details-content", Static)

        details = f"""
**{self.selected_server.name}**

{self.selected_server.description}

**Repository:** {self.selected_server.repository}
**Version:** {self.selected_server.version}

**Install Command:**
{self.selected_server.install_command}

**Run Command:**
{self.selected_server.run_command}

**Environment Variables:**
"""
        for env_var in self.selected_server.environment_variables:
            req = "Required" if env_var.required else "Optional"
            details += f"\n- {env_var.name} ({req}): {env_var.description}"

        details += "\n\n**Available Tools:**"
        for tool in self.selected_server.tools:
            details += f"\n- {tool.name}: {tool.description}"

        details_widget.update(details)

    @on(Button.Pressed, "#btn-install")
    async def install_server(self) -> None:
        """Handle install button press."""
        if not self.selected_server:
            self.app.notify("No server selected", severity="warning")
            return

        # Navigate to configuration screen
        # For now, show a placeholder
        self.app.notify(
            f"Installing {self.selected_server.name}...", severity="information"
        )
        # TODO: Navigate to configuration wizard
