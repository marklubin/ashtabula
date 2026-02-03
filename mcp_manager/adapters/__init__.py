"""Adapters for external integrations."""

from mcp_manager.adapters.claude_code_publisher import ClaudeCodePublisher
from mcp_manager.adapters.claude_desktop_publisher import ClaudeDesktopPublisher
from mcp_manager.adapters.filesystem_config_store import FileSystemConfigStore
from mcp_manager.adapters.github_registry_reader import GitHubRegistryReader
from mcp_manager.adapters.local_server_executor import LocalServerExecutor

__all__ = [
    "ClaudeCodePublisher",
    "ClaudeDesktopPublisher",
    "FileSystemConfigStore",
    "GitHubRegistryReader",
    "LocalServerExecutor",
]
