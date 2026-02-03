"""Configuration history and rollback screen."""

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, DataTable, Label, Select, Static

from mcp_manager.domain.interfaces import ConfigStore
from mcp_manager.domain.models import Client, ConfigVersion
from mcp_manager.services import ClientConfigService, ConfigVersionService


class ConfigHistoryScreen(Container):
    """Screen for viewing configuration history and performing rollbacks."""

    def __init__(
        self,
        config_version: ConfigVersionService,
        client_config: ClientConfigService,
        config_store: ConfigStore,
    ) -> None:
        """Initialize the config history screen.

        Args:
            config_version: Config version service instance.
            client_config: Client config service instance.
            config_store: Config store instance.
        """
        super().__init__()
        self.config_version = config_version
        self.client_config = client_config
        self.config_store = config_store
        self.clients: list[Client] = []
        self.selected_client: Client | None = None
        self.versions: list[ConfigVersion] = []
        self.selected_version: ConfigVersion | None = None

    def compose(self) -> ComposeResult:
        """Compose the config history UI."""
        with Vertical():
            yield Label("Configuration History & Rollback", classes="title")

            with Horizontal():
                yield Label("Select Client:")
                yield Select(
                    [(c.name, str(c.id)) for c in self.clients],
                    id="client-select",
                    allow_blank=False,
                )
                yield Button("Refresh", id="btn-refresh-history")

            with Horizontal():
                with Container(id="history-list"):
                    yield Label("Version History")
                    table = DataTable(id="versions-table")
                    table.add_columns("Version", "Date", "Description", "Rollback")
                    yield table

                with Container(id="version-details"):
                    yield Label("Version Details")
                    yield Static(id="version-info")
                    yield Button(
                        "Rollback to This Version",
                        id="btn-rollback",
                        variant="warning",
                    )

            with Container(id="comparison"):
                yield Label("Compare Versions")
                with Horizontal():
                    yield Label("Version 1:")
                    yield Select([], id="compare-v1")
                    yield Label("Version 2:")
                    yield Select([], id="compare-v2")
                    yield Button("Compare", id="btn-compare")
                yield Static(id="comparison-results")

    async def on_mount(self) -> None:
        """Load clients on mount."""
        await self._load_clients()

    async def _load_clients(self) -> None:
        """Load configured clients."""
        try:
            self.clients = await self.client_config.list_clients()

            # Update client select
            client_select = self.query_one("#client-select", Select)
            client_select.set_options([(c.name, str(c.id)) for c in self.clients])

        except Exception as e:
            self.app.notify(f"Failed to load clients: {e}", severity="error")

    @on(Select.Changed, "#client-select")
    async def client_selected(self, event: Select.Changed) -> None:
        """Handle client selection."""
        if event.value is None:
            return

        from uuid import UUID

        client_id = UUID(str(event.value))
        self.selected_client = await self.config_store.get_client(client_id)

        if self.selected_client:
            await self._load_history()

    async def _load_history(self) -> None:
        """Load version history for selected client."""
        if not self.selected_client:
            return

        try:
            self.versions = await self.config_version.get_version_history(
                self.selected_client.id
            )
            await self._update_versions_table()
            await self._update_compare_selects()

        except Exception as e:
            self.app.notify(f"Failed to load history: {e}", severity="error")

    async def _update_versions_table(self) -> None:
        """Update the versions table."""
        table = self.query_one("#versions-table", DataTable)
        table.clear()

        for version in self.versions:
            rollback_icon = "↩️" if version.is_rollback else ""
            date_str = version.created_at.strftime("%Y-%m-%d %H:%M")
            table.add_row(
                f"v{version.version}",
                date_str,
                version.description[:40],
                rollback_icon,
                key=str(version.id),
            )

    async def _update_compare_selects(self) -> None:
        """Update comparison select widgets."""
        options = [(f"v{v.version}", str(v.id)) for v in self.versions]

        compare_v1 = self.query_one("#compare-v1", Select)
        compare_v2 = self.query_one("#compare-v2", Select)

        compare_v1.set_options(options)
        compare_v2.set_options(options)

    @on(Button.Pressed, "#btn-refresh-history")
    async def refresh_history(self) -> None:
        """Handle refresh button press."""
        await self._load_clients()
        if self.selected_client:
            await self._load_history()
        self.app.notify("History refreshed", severity="information")

    @on(DataTable.RowSelected, "#versions-table")
    async def version_selected(self, event: DataTable.RowSelected) -> None:
        """Handle version selection."""
        from uuid import UUID

        version_id = UUID(str(event.row_key))
        self.selected_version = await self.config_version.get_version(version_id)
        await self._update_version_details()

    async def _update_version_details(self) -> None:
        """Update version details panel."""
        if not self.selected_version:
            return

        info_widget = self.query_one("#version-info", Static)

        info = f"""
**Version {self.selected_version.version}**

**Date:** {self.selected_version.created_at.strftime("%Y-%m-%d %H:%M:%S")}
**Rollback:** {"Yes" if self.selected_version.is_rollback else "No"}

**Description:**
{self.selected_version.description}

**Servers in this configuration:**
"""
        for server_config in self.selected_version.config.server_configs:
            info += f"\n- {server_config.name} ({server_config.server.name})"

        info_widget.update(info)

    @on(Button.Pressed, "#btn-rollback")
    async def rollback_version(self) -> None:
        """Handle rollback button press."""
        if not self.selected_version:
            self.app.notify("No version selected", severity="warning")
            return

        try:
            new_version = await self.config_version.rollback_to_version(
                self.selected_version.id,
                description=f"Rollback to v{self.selected_version.version}",
            )

            # Publish the rolled-back configuration
            await self.client_config.publish_configuration(new_version.config)

            await self._load_history()
            self.app.notify(
                f"Rolled back to version {self.selected_version.version}",
                severity="information",
            )

        except Exception as e:
            self.app.notify(f"Failed to rollback: {e}", severity="error")

    @on(Button.Pressed, "#btn-compare")
    async def compare_versions(self) -> None:
        """Handle compare button press."""
        v1_select = self.query_one("#compare-v1", Select)
        v2_select = self.query_one("#compare-v2", Select)

        if v1_select.value is None or v2_select.value is None:
            self.app.notify("Select two versions to compare", severity="warning")
            return

        try:
            from uuid import UUID

            v1_id = UUID(str(v1_select.value))
            v2_id = UUID(str(v2_select.value))

            comparison = await self.config_version.compare_versions(v1_id, v2_id)

            # Display comparison results
            results_widget = self.query_one("#comparison-results", Static)

            results = f"""
**Comparison: v{comparison['version_1'].version} vs v{comparison['version_2'].version}**

**Added Servers ({len(comparison['added_servers'])}):**
"""
            for server in comparison["added_servers"]:
                results += f"\n+ {server.name}"

            results += f"\n\n**Removed Servers ({len(comparison['removed_servers'])}):**"
            for server in comparison["removed_servers"]:
                results += f"\n- {server.name}"

            results += f"\n\n**Modified Servers ({len(comparison['modified_servers'])}):**"
            for old, new in comparison["modified_servers"]:
                results += f"\n~ {old.name}"

            results_widget.update(results)

        except Exception as e:
            self.app.notify(f"Failed to compare versions: {e}", severity="error")
