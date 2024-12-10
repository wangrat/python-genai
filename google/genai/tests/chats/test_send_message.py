import os

import PIL.Image
from pydantic import BaseModel, ValidationError
import pytest

from ... import _transformers as t
from ... import errors
from ... import types
from .. import pytest_helper


IMAGE_FILE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../data/google.jpg')
)
image = PIL.Image.open(IMAGE_FILE_PATH)


pytestmark = pytest_helper.setup(
    file=__file__,
    globals_for_file=globals(),
)
pytest_plugins = ('pytest_asyncio',)


def test_text(client):
  chat = client.chats.create(model='gemini-1.5-flash')
  chat.send_message(
      'tell me a story in 100 words',
  )


def test_part(client):
  chat = client.chats.create(model='gemini-1.5-flash')
  chat.send_message(
      types.Part.from_text('tell me a story in 100 words'),
  )


def test_parts(client):
  chat = client.chats.create(model='gemini-1.5-flash')
  chat.send_message(
      [
          types.Part.from_text('tell me a US city'),
          types.Part.from_text('the city is in west coast'),
      ],
  )


def test_send_2_messages(client):
  chat = client.chats.create(model='gemini-1.5-flash')
  chat.send_message('write a python function to check if a year is a leap year')
  chat.send_message('write a unit test for the function')


def test_image(client):
  chat = client.chats.create(model='gemini-1.5-flash')
  chat.send_message(
      [
          'what is the image about?',
          image,
      ],
  )


def test_google_cloud_storage_uri(client):
  chat = client.chats.create(model='gemini-1.5-flash')
  with pytest_helper.exception_if_mldev(client, errors.ClientError):
    chat.send_message(
        [
            'what is the image about?',
            types.Part.from_uri(
                'gs://unified-genai-dev/imagen-inputs/google_small.png',
                'image/png',
            ),
        ],
    )


def test_uploaded_file_uri(client):
  chat = client.chats.create(model='gemini-1.5-flash')
  with pytest_helper.exception_if_vertex(client, errors.ClientError):
    chat.send_message(
        [
            'what is the image about?',
            types.Part.from_uri(
                'https://generativelanguage.googleapis.com/v1beta/files/9w04rxmcgsp8',
                'image/png',
            ),
        ],
    )


def test_history(client):
  history = [
      types.Content(
          role='model',
          parts=[types.Part.from_text('Hello there! how can I help you?')],
      ),
      types.Content(
          role='user', parts=[types.Part.from_text('define a=5, b=10')]
      ),
  ]
  chat = client.chats.create(model='gemini-1.5-flash', history=history)
  chat.send_message('what is a + b?')


@pytest.mark.asyncio
async def test_async_text(client):
  chat = client.aio.chats.create(model='gemini-1.5-flash')
  await chat.send_message('tell me a story in 100 words')


@pytest.mark.asyncio
async def test_async_part(client):
  chat = client.aio.chats.create(model='gemini-1.5-flash')
  await chat.send_message(types.Part.from_text('tell me a story in 100 words'))


@pytest.mark.asyncio
async def test_async_parts(client):
  chat = client.aio.chats.create(model='gemini-1.5-flash')
  await chat.send_message(
      [
          types.Part.from_text('tell me a US city'),
          types.Part.from_text('the city is in west coast'),
      ],
  )


@pytest.mark.asyncio
async def test_async_history(client):
  history = [
      types.Content(
          role='model',
          parts=[types.Part.from_text('Hello there! how can I help you?')],
      ),
      types.Content(
          role='user', parts=[types.Part.from_text('define a=5, b=10')]
      ),
  ]
  chat = client.aio.chats.create(model='gemini-1.5-flash', history=history)
  await chat.send_message('what is a + b?')
