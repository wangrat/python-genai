# Copyright 2024 Google LLC
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


"""Tests for generate_content's speech_config flag."""

from pydantic import BaseModel, ValidationError
import pytest
from ... import _transformers as t
from ... import errors
from ... import types
from .. import pytest_helper


pytestmark = pytest_helper.setup(
    file=__file__,
    globals_for_file=globals(),
    test_method='models.generate_content',
)

pytest_plugin = ('pytest_asyncio',)


def test_generate_content_google_search_tool(client):
  contents = 'Why is the sky blue?'
  config = types.GenerateContentConfig(
      tools=[types.Tool(google_search=types.GoogleSearch())]
  )
  with pytest_helper.exception_if_vertex(client, errors.ClientError):
    client.models.generate_content(
        model='gemini-v3-s-test',
        contents=contents,
        config=config,
    )


def test_generate_content_with_config(client):
  contents = 'Produce a speech response saying "Cheese"'
  config = types.GenerateContentConfig(
      response_modalities=['audio'],
      speech_config=types.SpeechConfig(
          voice_config=types.VoiceConfig(
              prebuilt_voice_config=types.PrebuiltVoiceConfig(
                  voice_name='charon'
              )
          )
      )
  )
  # Right now the MLDev endpoint returns a 502 error, but the point of
  # this test is to verify that the SDK sends the correct request.
  with pytest_helper.exception_if_mldev(client, errors.ServerError):
    with pytest_helper.exception_if_vertex(client, errors.ClientError):
      client.models.generate_content(
          model='gemini-v3-s-test',
          contents=contents,
          config=config,
      )


def test_generate_content_string_config(client):
  contents = 'Say hello!'
  config = types.GenerateContentConfig(
      response_modalities=['audio'], speech_config='charon'
  )

  # Right now the MLDev endpoint returns a 502 error, but the point of
  # this test is to verify that the SDK transforms the string into a
  # SpeechConfig object, which is what we can see in the recorded request.
  with pytest_helper.exception_if_mldev(client, errors.ServerError):
    with pytest_helper.exception_if_vertex(client, errors.ClientError):
      client.models.generate_content(
          model='gemini-v3-s-test',
          contents=contents,
          config=config,
      )
