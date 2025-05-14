# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from ._mcp_utils import mcp_to_gemini_tools
from .types import FunctionCall, Tool

try:
  from mcp import ClientSession
  from mcp import types as mcp_types
except ImportError as e:
  import sys

  if sys.version_info < (3, 10):
    raise ImportError(
        "MCP Tool requires Python 3.10 or above. Please upgrade your Python"
        " version."
    ) from e
  else:
    raise e


class McpToGenAiToolAdapter:
  """Adapter for working with MCP tools in a GenAI client."""

  def __init__(
      self, session: ClientSession, list_tools_result: mcp_types.ListToolsResult
  ) -> None:
    self._mcp_session = session
    self._list_tools_result = list_tools_result

  async def call_tool(
      self, function_call: FunctionCall
  ) -> mcp_types.CallToolResult:
    """Calls a function on the MCP server."""
    name = function_call.name if function_call.name else ""
    arguments = dict(function_call.args) if function_call.args else {}
    return await self._mcp_session.call_tool(
        name=name,
        arguments=arguments,
    )

  @property
  def tools(self) -> list[Tool]:
    """Returns a list of Google GenAI tools."""
    return mcp_to_gemini_tools(self._list_tools_result.tools)
