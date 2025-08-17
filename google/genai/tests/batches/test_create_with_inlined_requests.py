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


"""Tests for batches.create() with inlined requests."""
import base64
import datetime
import os

import pytest

from ... import types
from .. import pytest_helper

_GEMINI_MODEL = 'gemini-1.5-flash-002'
_DISPLAY_NAME = 'test_batch'

_MLDEV_GEMINI_MODEL = 'gemini-2.0-flash'
_INLINED_REQUEST = {
    'contents': [{
        'parts': [{
            'text': 'Hello!',
        }],
        'role': 'user',
    }],
}
_INLINED_TEXT_REQUEST = {
    'contents': [{
        'parts': [{
            'text': 'What is the QQQ stock price?',
        }],
        'role': 'user',
    }],
    'config': {
        'response_modalities': ['TEXT'],
    },
}
_INLINED_IMAGE_REQUEST = {
    'contents': [{
        'parts': [
            {'text': 'What is in this image?'},
            {
                'file_data': {
                    'file_uri': (
                        'https://generativelanguage.googleapis.com/v1beta/files/kje1wewvo85z'
                    ),
                    'mime_type': 'image/jpeg',
                },
            },
        ],
        'role': 'user',
    }],
    'config': {
        'temperature': 0.7,
        'top_p': 0.9,
        'top_k': 10,
    },
}
_INLINED_VIDEO_REQUEST = {
    'contents': [{
        'parts': [
            {
                'text': 'Summerize this video.',
            },
            {
                'file_data': {
                    'file_uri': (
                        'https://generativelanguage.googleapis.com/v1beta/files/tyvaih24jwje'
                    ),
                    'mime_type': 'video/mp4',
                },
                'video_metadata': {
                    'start_offset': '0s',
                    'end_offset': '5s',
                    'fps': 3,
                },
            },
        ],
        'role': 'user',
    }],
}
_IMAGE_PNG_FILE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../data/google.png')
)
with open(_IMAGE_PNG_FILE_PATH, 'rb') as image_file:
  image_bytes = image_file.read()
  image_string = base64.b64encode(image_bytes).decode('utf-8')
_INLINED_IMAGE_BLOB_REQUEST = types.InlinedRequest(
    contents=[
        types.Content(
            parts=[
                types.Part(text='What is this image about?'),
                types.Part(
                    inline_data=types.Blob(
                        data=image_string,
                        mime_type='image/png',
                    ),
                ),
            ],
            role='user',
        )
    ]
)

# All tests will be run for both Vertex and MLDev.
test_table: list[pytest_helper.TestTableItem] = [
    pytest_helper.TestTableItem(
        name='test_union_with_inlined_request',
        parameters=types._CreateBatchJobParameters(
            model=_MLDEV_GEMINI_MODEL,
            src=[_INLINED_REQUEST],
            config={
                'display_name': _DISPLAY_NAME,
            },
        ),
        exception_if_vertex='not supported in Vertex',
        has_union=True,
    ),
    pytest_helper.TestTableItem(
        name='test_with_inlined_request',
        parameters=types._CreateBatchJobParameters(
            model=_MLDEV_GEMINI_MODEL,
            src={'inlined_requests': [_INLINED_TEXT_REQUEST]},
            config={
                'display_name': _DISPLAY_NAME,
            },
        ),
        exception_if_vertex='not supported',
    ),
    pytest_helper.TestTableItem(
        name='test_with_image_file',
        parameters=types._CreateBatchJobParameters(
            model=_MLDEV_GEMINI_MODEL,
            src={'inlined_requests': [_INLINED_IMAGE_REQUEST]},
            config={
                'display_name': _DISPLAY_NAME,
            },
        ),
        exception_if_vertex='not supported',
    ),
    pytest_helper.TestTableItem(
        name='test_with_image_blob',
        parameters=types._CreateBatchJobParameters(
            model=_MLDEV_GEMINI_MODEL,
            src={'inlined_requests': [_INLINED_IMAGE_BLOB_REQUEST]},
            config={
                'display_name': _DISPLAY_NAME,
            },
        ),
        exception_if_vertex='not supported',
    ),
    pytest_helper.TestTableItem(
        name='test_with_video_file',
        parameters=types._CreateBatchJobParameters(
            model=_MLDEV_GEMINI_MODEL,
            src={'inlined_requests': [_INLINED_VIDEO_REQUEST]},
            config={
                'display_name': _DISPLAY_NAME,
            },
        ),
        exception_if_vertex='not supported',
    ),
]

pytestmark = [
    pytest.mark.usefixtures('mock_timestamped_unique_name'),
    pytest_helper.setup(
        file=__file__,
        globals_for_file=globals(),
        test_method='batches.create',
        test_table=test_table,
    ),
]


@pytest.mark.asyncio
async def test_async_create(client):
  with pytest_helper.exception_if_vertex(client, ValueError):
    batch_job = await client.aio.batches.create(
        model=_GEMINI_MODEL,
        src=[_INLINED_REQUEST],
    )
    assert batch_job.name.startswith('batches/')
    assert (
        batch_job.model == 'models/' + _GEMINI_MODEL
    )  # Converted to Gemini full name.
