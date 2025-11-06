"""Main Textual application for MCP Manager."""

from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Footer, Header, TabbedContent, TabPane

from mcp_manager.adapters import (
    ClaudeCodePublisher,
    ClaudeDesktopPublisher,
    FileSystemConfigStore,
    GitHubRegistryReader,
    LocalServerExecutor,
)
from mcp_manager.services import (
    ClientConfigService,
    ConfigVersionService,
    RegistryService,
    ServerExecutionService,
    ServerManagementService,
)
from mcp_manager.ui.screens.browse_servers import BrowseServersScreen
from mcp_manager.ui.screens.config_history import ConfigHistoryScreen
from mcp_manager.ui.screens.manage_clients import ManageClientsScreen
from mcp_manager.ui.screens.my_servers import MyServersScreen
from mcp_manager.ui.screens.test_tools import TestToolsScreen


class MCPManagerApp(App[None]):
    """MCP Manager TUI Application."""

    CSS = """
    Screen {
        background: $surface;
    }

    TabbedContent {
        height: 1fr;
    }

    TabPane {
        padding: 1 2;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "toggle_dark", "Toggle Dark Mode"),
    ]

    def __init__(self, config_dir: Path | None = None) -> None:
        """Initialize the MCP Manager app.

        Args:
            config_dir: Configuration directory. Defaults to ~/.mcp-manager
        """
        super().__init__()
        self.title = "MCP Manager"
        self.sub_title = "Manage MCP Servers Across Multiple Clients"

        # Set config directory
        if config_dir is None:
            config_dir = Path.home() / ".mcp-manager"
        config_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir = config_dir

        # Initialize adapters
        self.config_store = FileSystemConfigStore(config_dir / "data")
        self.registry_reader = GitHubRegistryReader()
        self.server_executor = LocalServerExecutor()
        self.publishers = [ClaudeCodePublisher(), ClaudeDesktopPublisher()]

        # Initialize services
        self.registry_service = RegistryService(self.registry_reader)
        self.server_management = ServerManagementService(self.config_store)
        self.server_execution = ServerExecutionService(
            self.server_executor, self.config_store
        )
        self.client_config = ClientConfigService(self.config_store, self.publishers)
        self.config_version = ConfigVersionService(self.config_store)

    def compose(self) -> ComposeResult:
        """Compose the UI layout."""
        yield Header()
        with TabbedContent():
            with TabPane("Browse Servers", id="tab-browse"):
                yield BrowseServersScreen(
                    self.registry_service, self.server_management
                )
            with TabPane("My Servers", id="tab-my-servers"):
                yield MyServersScreen(self.server_management, self.server_execution)
            with TabPane("Test Tools", id="tab-test"):
                yield TestToolsScreen(self.server_management, self.server_execution)
            with TabPane("Manage Clients", id="tab-clients"):
                yield ManageClientsScreen(self.client_config)
            with TabPane("Config History", id="tab-history"):
                yield ConfigHistoryScreen(
                    self.config_version, self.client_config, self.config_store
                )
        yield Footer()

    async def on_mount(self) -> None:
        """Handle app mount event."""
        # Set dark mode by default
        self.theme = "nord"

    async def action_toggle_dark(self) -> None:
        """Toggle dark mode."""
        self.dark = not self.dark

    async def on_unmount(self) -> None:
        """Handle app unmount event."""
        # Cleanup running servers
        await self.server_executor.cleanup()
