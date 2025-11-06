"""Test MCP server tools interactively screen."""

import json

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, DataTable, Label, Select, Static, TextArea

from mcp_manager.domain.models import ServerConfig, Tool
from mcp_manager.services import ServerExecutionService, ServerManagementService


class TestToolsScreen(Container):
    """Screen for testing MCP server tools interactively."""

    def __init__(
        self,
        server_management: ServerManagementService,
        server_execution: ServerExecutionService,
    ) -> None:
        """Initialize the test tools screen.

        Args:
            server_management: Server management service instance.
            server_execution: Server execution service instance.
        """
        super().__init__()
        self.server_management = server_management
        self.server_execution = server_execution
        self.servers: list[ServerConfig] = []
        self.selected_server: ServerConfig | None = None
        self.available_tools: list[Tool] = []
        self.selected_tool: Tool | None = None

    def compose(self) -> ComposeResult:
        """Compose the test tools UI."""
        with Vertical():
            yield Label("Test MCP Server Tools", classes="title")

            with Horizontal():
                with Container(id="server-selection"):
                    yield Label("Select Running Server")
                    yield Select(
                        [(s.name, str(s.id)) for s in self.servers],
                        id="server-select",
                        allow_blank=False,
                    )

                with Container(id="tool-selection"):
                    yield Label("Select Tool")
                    yield Select(
                        [(t.name, t.name) for t in self.available_tools],
                        id="tool-select",
                        allow_blank=False,
                    )

            with Horizontal():
                with Container(id="tool-params"):
                    yield Label("Tool Parameters (JSON)")
                    yield TextArea(
                        '{\n  "param": "value"\n}',
                        id="params-input",
                        language="json",
                    )
                    yield Button("Execute Tool", id="btn-execute", variant="primary")

                with Container(id="tool-results"):
                    yield Label("Execution Results")
                    yield Static(id="results-output")

            with Container(id="tool-details"):
                yield Label("Tool Details")
                yield Static(id="tool-info")

    async def on_mount(self) -> None:
        """Load servers on mount."""
        await self._load_servers()

    async def _load_servers(self) -> None:
        """Load running servers."""
        try:
            all_servers = await self.server_management.list_server_configs()
            # Filter to only running servers
            from mcp_manager.domain.models import ServerStatus

            self.servers = [
                s for s in all_servers if s.status == ServerStatus.RUNNING
            ]

            # Update server select
            server_select = self.query_one("#server-select", Select)
            server_select.set_options([(s.name, str(s.id)) for s in self.servers])

        except Exception as e:
            self.app.notify(f"Failed to load servers: {e}", severity="error")

    @on(Select.Changed, "#server-select")
    async def server_selected(self, event: Select.Changed) -> None:
        """Handle server selection."""
        if event.value is None:
            return

        from uuid import UUID

        server_id = UUID(str(event.value))
        self.selected_server = await self.server_management.get_server_config(
            server_id
        )

        if self.selected_server:
            await self._load_tools()

    async def _load_tools(self) -> None:
        """Load available tools for selected server."""
        if not self.selected_server:
            return

        try:
            self.available_tools = await self.server_execution.list_tools(
                self.selected_server.id
            )

            # Update tool select
            tool_select = self.query_one("#tool-select", Select)
            tool_select.set_options([(t.name, t.name) for t in self.available_tools])

        except Exception as e:
            self.app.notify(f"Failed to load tools: {e}", severity="error")

    @on(Select.Changed, "#tool-select")
    async def tool_selected(self, event: Select.Changed) -> None:
        """Handle tool selection."""
        if event.value is None:
            return

        tool_name = str(event.value)
        self.selected_tool = next(
            (t for t in self.available_tools if t.name == tool_name), None
        )

        if self.selected_tool:
            await self._update_tool_details()

    async def _update_tool_details(self) -> None:
        """Update tool details panel."""
        if not self.selected_tool:
            return

        tool_info = self.query_one("#tool-info", Static)

        details = f"""
**{self.selected_tool.name}**

{self.selected_tool.description}

**Parameters:**
"""
        for param in self.selected_tool.parameters:
            req = "Required" if param.required else "Optional"
            default = f" (default: {param.default})" if param.default else ""
            details += f"\n- {param.name} ({param.type}, {req}){default}: {param.description}"

        tool_info.update(details)

        # Update parameter template
        params_input = self.query_one("#params-input", TextArea)
        template = {}
        for param in self.selected_tool.parameters:
            if param.required:
                template[param.name] = param.default or f"<{param.type}>"

        params_input.text = json.dumps(template, indent=2)

    @on(Button.Pressed, "#btn-execute")
    async def execute_tool(self) -> None:
        """Handle execute button press."""
        if not self.selected_server or not self.selected_tool:
            self.app.notify("Select a server and tool first", severity="warning")
            return

        params_input = self.query_one("#params-input", TextArea)
        results_output = self.query_one("#results-output", Static)

        try:
            # Parse parameters
            params = json.loads(params_input.text)

            # Execute tool
            results_output.update("Executing...")
            result = await self.server_execution.test_tool(
                self.selected_server.id, self.selected_tool.name, params
            )

            # Display results
            results_json = json.dumps(result, indent=2)
            results_output.update(f"```json\n{results_json}\n```")

            self.app.notify("Tool executed successfully", severity="information")

        except json.JSONDecodeError as e:
            self.app.notify(f"Invalid JSON parameters: {e}", severity="error")
            results_output.update(f"Error: Invalid JSON - {e}")
        except Exception as e:
            self.app.notify(f"Tool execution failed: {e}", severity="error")
            results_output.update(f"Error: {e}")
