"""Manage configured MCP servers screen."""

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, DataTable, Label, Static

from mcp_manager.domain.models import ServerConfig, ServerStatus
from mcp_manager.services import ServerExecutionService, ServerManagementService


class MyServersScreen(Container):
    """Screen for managing configured MCP servers."""

    def __init__(
        self,
        server_management: ServerManagementService,
        server_execution: ServerExecutionService,
    ) -> None:
        """Initialize the my servers screen.

        Args:
            server_management: Server management service instance.
            server_execution: Server execution service instance.
        """
        super().__init__()
        self.server_management = server_management
        self.server_execution = server_execution
        self.servers: list[ServerConfig] = []
        self.selected_server: ServerConfig | None = None

    def compose(self) -> ComposeResult:
        """Compose the my servers UI."""
        with Vertical():
            yield Label("My Configured Servers", classes="title")
            with Horizontal():
                yield Button("Refresh", id="btn-refresh-servers")
                yield Button("Add New Server", id="btn-add-server", variant="primary")

            with Horizontal():
                with Container(id="servers-list"):
                    yield Label("Servers")
                    table = DataTable(id="my-servers-table")
                    table.add_columns("Name", "Status", "Server Type")
                    yield table

                with Container(id="server-control"):
                    yield Label("Server Control")
                    yield Static(id="server-info")
                    with Horizontal():
                        yield Button("Start", id="btn-start", variant="success")
                        yield Button("Stop", id="btn-stop", variant="error")
                        yield Button("Restart", id="btn-restart")
                    yield Button("Delete", id="btn-delete", variant="warning")

    async def on_mount(self) -> None:
        """Load servers on mount."""
        await self._load_servers()

    async def _load_servers(self) -> None:
        """Load configured servers."""
        try:
            self.servers = await self.server_management.list_server_configs()
            await self._update_table()
        except Exception as e:
            self.app.notify(f"Failed to load servers: {e}", severity="error")

    async def _update_table(self) -> None:
        """Update the servers table."""
        table = self.query_one("#my-servers-table", DataTable)
        table.clear()

        for server in self.servers:
            status_icon = {
                ServerStatus.RUNNING: "🟢",
                ServerStatus.STOPPED: "⚫",
                ServerStatus.STARTING: "🟡",
                ServerStatus.ERROR: "🔴",
            }.get(server.status, "⚪")

            table.add_row(
                server.name,
                f"{status_icon} {server.status.value}",
                server.server.name,
                key=str(server.id),
            )

    @on(Button.Pressed, "#btn-refresh-servers")
    async def refresh_servers(self) -> None:
        """Handle refresh button press."""
        await self._load_servers()
        self.app.notify("Servers refreshed", severity="information")

    @on(DataTable.RowSelected)
    async def row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection."""
        from uuid import UUID

        server_id = UUID(str(event.row_key))
        self.selected_server = await self.server_management.get_server_config(
            server_id
        )
        await self._update_server_info()

    async def _update_server_info(self) -> None:
        """Update server info panel."""
        if not self.selected_server:
            return

        info_widget = self.query_one("#server-info", Static)

        info = f"""
**{self.selected_server.name}**

**Server:** {self.selected_server.server.name}
**Status:** {self.selected_server.status.value}

**Environment:**
"""
        for key, value in self.selected_server.environment.items():
            # Mask sensitive values
            display_value = value if len(value) < 20 else f"{value[:10]}..."
            info += f"\n- {key}: {display_value}"

        if self.selected_server.error_message:
            info += f"\n\n**Error:** {self.selected_server.error_message}"

        info_widget.update(info)

    @on(Button.Pressed, "#btn-start")
    async def start_server(self) -> None:
        """Handle start button press."""
        if not self.selected_server:
            self.app.notify("No server selected", severity="warning")
            return

        try:
            await self.server_execution.start_server(self.selected_server.id)
            await self._load_servers()
            self.app.notify(
                f"{self.selected_server.name} started", severity="information"
            )
        except Exception as e:
            self.app.notify(f"Failed to start server: {e}", severity="error")

    @on(Button.Pressed, "#btn-stop")
    async def stop_server(self) -> None:
        """Handle stop button press."""
        if not self.selected_server:
            self.app.notify("No server selected", severity="warning")
            return

        try:
            await self.server_execution.stop_server(self.selected_server.id)
            await self._load_servers()
            self.app.notify(
                f"{self.selected_server.name} stopped", severity="information"
            )
        except Exception as e:
            self.app.notify(f"Failed to stop server: {e}", severity="error")

    @on(Button.Pressed, "#btn-restart")
    async def restart_server(self) -> None:
        """Handle restart button press."""
        if not self.selected_server:
            self.app.notify("No server selected", severity="warning")
            return

        try:
            await self.server_execution.restart_server(self.selected_server.id)
            await self._load_servers()
            self.app.notify(
                f"{self.selected_server.name} restarted", severity="information"
            )
        except Exception as e:
            self.app.notify(f"Failed to restart server: {e}", severity="error")

    @on(Button.Pressed, "#btn-delete")
    async def delete_server(self) -> None:
        """Handle delete button press."""
        if not self.selected_server:
            self.app.notify("No server selected", severity="warning")
            return

        try:
            await self.server_management.delete_server_config(self.selected_server.id)
            await self._load_servers()
            self.selected_server = None
            self.app.notify("Server deleted", severity="information")
        except Exception as e:
            self.app.notify(f"Failed to delete server: {e}", severity="error")
