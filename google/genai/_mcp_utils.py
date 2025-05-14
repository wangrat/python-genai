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

"""Utils for working with MCP tools."""

import typing
from typing import Any

from . import types

if typing.TYPE_CHECKING:
  from mcp.types import Tool as McpTool
else:
  McpTool: typing.Type = Any
  try:
    from mcp.types import Tool as McpTool
  except ImportError:
    McpTool = None


def mcp_to_gemini_tool(tool: McpTool) -> types.Tool:
  """Translates an MCP tool to a Google GenAI tool."""
  return types.Tool(
      function_declarations=[{
          "name": tool.name,
          "description": tool.description,
          "parameters": types.Schema.from_json_schema(
              json_schema=types.JSONSchema(
                  **_filter_to_supported_schema(tool.inputSchema)
              )
          ),
      }]
  )


def mcp_to_gemini_tools(tools: list[McpTool]) -> list[types.Tool]:
  """Translates a list of MCP tools to a list of Google GenAI tools."""
  return [mcp_to_gemini_tool(tool) for tool in tools]


def _filter_to_supported_schema(schema: dict[str, Any]) -> dict[str, Any]:
  """Filters the schema to only include fields that are supported by JSONSchema."""
  supported_fields: set[str] = set(types.JSONSchema.model_fields.keys())
  schema_field_names: tuple[str] = ("items",)  # 'additional_properties' to come
  list_schema_field_names: tuple[str] = (
      "any_of",  # 'one_of', 'all_of', 'not' to come
  )
  dict_schema_field_names: tuple[str] = ("properties",)  # 'defs' to come
  for field_name, field_value in schema.items():
    if field_name in schema_field_names:
      schema[field_name] = _filter_to_supported_schema(field_value)
    elif field_name in list_schema_field_names:
      schema[field_name] = [
          _filter_to_supported_schema(value) for value in field_value
      ]
    elif field_name in dict_schema_field_names:
      schema[field_name] = {
          key: _filter_to_supported_schema(value)
          for key, value in field_value.items()
      }
  return {
      key: value for key, value in schema.items() if key in supported_fields
  }
