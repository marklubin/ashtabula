"""Local MCP server executor implementation."""

import asyncio
import json
import os
import shlex
from typing import Any
from uuid import UUID

from mcp_manager.domain.interfaces import ServerExecutor
from mcp_manager.domain.models import ServerConfig, Tool


class LocalServerExecutor(ServerExecutor):
    """Execute MCP servers locally as subprocesses."""

    def __init__(self) -> None:
        """Initialize the local server executor."""
        self._processes: dict[UUID, asyncio.subprocess.Process] = {}
        self._tools_cache: dict[UUID, list[Tool]] = {}

    async def start_server(self, config: ServerConfig) -> None:
        """Start an MCP server."""
        if config.id in self._processes:
            # Already running
            return

        try:
            # Parse the run command using shlex for proper shell-like parsing
            cmd_parts = shlex.split(config.server.run_command)

            # Create environment: inherit system environment and add configured variables
            env = os.environ.copy()
            env.update(config.environment)

            # Start the process
            process = await asyncio.create_subprocess_exec(
                *cmd_parts,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE,
            )

            self._processes[config.id] = process

            # Give it a moment to start
            await asyncio.sleep(0.5)

            # Check if it's still running
            if process.returncode is not None:
                stderr = await process.stderr.read()
                raise RuntimeError(
                    f"Server failed to start: {stderr.decode('utf-8')}"
                )

        except Exception as e:
            # Clean up if start failed
            if config.id in self._processes:
                del self._processes[config.id]
            raise RuntimeError(f"Failed to start server: {e}") from e

    async def stop_server(self, config_id: UUID) -> None:
        """Stop a running MCP server."""
        if config_id not in self._processes:
            return

        process = self._processes[config_id]

        try:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        finally:
            del self._processes[config_id]
            if config_id in self._tools_cache:
                del self._tools_cache[config_id]

    async def get_server_status(self, config_id: UUID) -> str:
        """Get the status of a server."""
        if config_id not in self._processes:
            return "stopped"

        process = self._processes[config_id]
        if process.returncode is not None:
            return "error"

        return "running"

    async def test_tool(
        self, config_id: UUID, tool_name: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Test a tool on a running server."""
        if config_id not in self._processes:
            raise RuntimeError("Server is not running")

        process = self._processes[config_id]

        # Create JSON-RPC request
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": f"tools/{tool_name}",
            "params": parameters,
        }

        try:
            # Send request to server
            request_json = json.dumps(request) + "\n"
            process.stdin.write(request_json.encode("utf-8"))
            await process.stdin.drain()

            # Read response (with timeout)
            response_line = await asyncio.wait_for(
                process.stdout.readline(), timeout=30.0
            )

            response = json.loads(response_line.decode("utf-8"))

            if "error" in response:
                raise RuntimeError(f"Tool execution error: {response['error']}")

            return response.get("result", {})

        except asyncio.TimeoutError:
            raise RuntimeError("Tool execution timed out")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON response from server: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to execute tool: {e}") from e

    async def list_tools(self, config_id: UUID) -> list[Tool]:
        """List all available tools for a running server."""
        if config_id not in self._processes:
            raise RuntimeError("Server is not running")

        # Check cache first
        if config_id in self._tools_cache:
            return self._tools_cache[config_id]

        process = self._processes[config_id]

        # Create JSON-RPC request to list tools
        request = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}

        try:
            # Send request
            request_json = json.dumps(request) + "\n"
            process.stdin.write(request_json.encode("utf-8"))
            await process.stdin.drain()

            # Read response
            response_line = await asyncio.wait_for(
                process.stdout.readline(), timeout=10.0
            )

            response = json.loads(response_line.decode("utf-8"))

            if "error" in response:
                raise RuntimeError(f"Failed to list tools: {response['error']}")

            tools_data = response.get("result", {}).get("tools", [])
            tools = [Tool.model_validate(t) for t in tools_data]

            # Cache the tools
            self._tools_cache[config_id] = tools

            return tools

        except asyncio.TimeoutError:
            raise RuntimeError("Listing tools timed out")
        except Exception as e:
            raise RuntimeError(f"Failed to list tools: {e}") from e

    async def cleanup(self) -> None:
        """Stop all running servers and cleanup."""
        for config_id in list(self._processes.keys()):
            await self.stop_server(config_id)
