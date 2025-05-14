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

import pytest
from ... import _transformers as t
from ... import types
from .. import pytest_helper

try:
  from mcp import types as mcp_types
except ImportError as e:
  import sys

  if sys.version_info < (3, 10):
    raise ImportError(
        'MCP Tool requires Python 3.10 or above. Please upgrade your Python'
        ' version.'
    ) from e
  else:
    raise e

pytestmark = pytest_helper.setup(
    file=__file__,
    globals_for_file=globals(),
    test_method='models.generate_content',
)


# Cannot be included in test_table because MCP integration is only supported
# in the Python SDK.
@pytest.mark.asyncio
async def test_mcp_tools_async(client):
  response = await client.aio.models.generate_content(
      model='gemini-2.0-flash',
      contents=t.t_contents(None, 'What is the weather in Boston?'),
      config={
          'tools': [
              mcp_types.Tool(
                  name='get_weather',
                  description='Get the weather in a city.',
                  inputSchema={
                      'type': 'object',
                      'properties': {'location': {'type': 'string'}},
                  },
              )
          ],
      },
  )
  assert response.function_calls == [
      types.FunctionCall(
          name='get_weather',
          args={'location': 'Boston'},
      )
  ]
