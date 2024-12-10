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


"""Tests for client behavior when issuing requests."""

import http

from ... import _api_client as api_client
from ... import Client


def build_test_client(monkeypatch):
  monkeypatch.setenv('GOOGLE_API_KEY', 'google_api_key')
  return Client()


def test_build_request_sets_library_version_headers(monkeypatch):
  request_client = build_test_client(monkeypatch).models.api_client
  request = request_client._build_request('GET', 'test/path', {'key': 'value'})
  assert 'google-genai-sdk/' in request.headers['user-agent']
  assert 'gl-python/' in request.headers['user-agent']
  assert 'google-genai-sdk/' in request.headers['x-goog-api-client']
  assert 'gl-python/' in request.headers['x-goog-api-client']


def test_build_request_appends_to_user_agent_headers(monkeypatch):
  request_client = build_test_client(monkeypatch).models.api_client
  request = request_client._build_request(
      'GET',
      'test/path',
      {'key': 'value'},
      api_client.HttpOptions(
          url='test/url',
          api_version='1',
          headers={'user-agent': 'test-user-agent'},
          response_payload=None,
      ),
  )
  assert 'test-user-agent' in request.headers['user-agent']
  assert 'google-genai-sdk/' in request.headers['user-agent']
  assert 'gl-python/' in request.headers['user-agent']
  assert 'google-genai-sdk/' in request.headers['x-goog-api-client']


def test_build_request_appends_to_goog_api_client_headers(monkeypatch):
  request_client = build_test_client(monkeypatch).models.api_client
  request = request_client._build_request(
      'GET',
      'test/path',
      {'key': 'value'},
      api_client.HttpOptions(
          url='test/url',
          api_version='1',
          headers={'x-goog-api-client': 'test-goog-api-client'},
          response_payload=None,
      ),
  )
  assert 'google-genai-sdk/' in request.headers['user-agent']
  assert 'test-goog-api-client' in request.headers['x-goog-api-client']
  assert 'google-genai-sdk/' in request.headers['x-goog-api-client']
  assert 'gl-python/' in request.headers['x-goog-api-client']


def test_build_request_keeps_sdk_version_headers(monkeypatch):
  headers_to_inject = {}
  api_client._append_library_version_headers(headers_to_inject)
  assert 'google-genai-sdk/' in headers_to_inject['user-agent']
  request_client = build_test_client(monkeypatch).models.api_client
  request = request_client._build_request(
      'GET',
      'test/path',
      {'key': 'value'},
      api_client.HttpOptions(
          url='test/url',
          api_version='1',
          headers=headers_to_inject,
          response_payload=None,
      ),
  )
  assert 'google-genai-sdk/' in request.headers['user-agent']
  assert 'gl-python/' in request.headers['x-goog-api-client']
  assert 'google-genai-sdk/' in request.headers['x-goog-api-client']
  assert 'gl-python/' in request.headers['x-goog-api-client']