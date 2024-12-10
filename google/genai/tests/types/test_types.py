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


from ... import types


class SubPart(types.Part):
  pass


def test_factory_method_from_uri_part():

  my_part = SubPart.from_uri(
      'gs://generativeai-downloads/images/scones.jpg', 'image/jpeg'
  )
  assert (
      my_part.file_data.file_uri
      == 'gs://generativeai-downloads/images/scones.jpg'
  )
  assert my_part.file_data.mime_type == 'image/jpeg'
  assert isinstance(my_part, SubPart)


def test_factory_method_from_text_part():
  my_part = SubPart.from_text('What is your name?')
  assert my_part.text == 'What is your name?'
  assert isinstance(my_part, SubPart)


def test_factory_method_from_bytes_part():
  my_part = SubPart.from_bytes(b'123', 'text/plain')
  assert my_part.inline_data.data == b'123'
  assert my_part.inline_data.mime_type == 'text/plain'
  assert isinstance(my_part, SubPart)


def test_factory_method_from_function_call_part():
  my_part = SubPart.from_function_call('func', {'arg': 'value'})
  assert my_part.function_call.name == 'func'
  assert my_part.function_call.args == {'arg': 'value'}
  assert isinstance(my_part, SubPart)


def test_factory_method_from_function_response_part():
  my_part = SubPart.from_function_response('func', {'response': 'value'})
  assert my_part.function_response.name == 'func'
  assert my_part.function_response.response == {'response': 'value'}
  assert isinstance(my_part, SubPart)


def test_factory_method_from_video_metadata_part():
  my_part = SubPart.from_video_metadata('10s', '20s')
  assert my_part.video_metadata.end_offset == '10s'
  assert my_part.video_metadata.start_offset == '20s'
  assert isinstance(my_part, SubPart)


def test_factory_method_from_executable_code_part():
  my_part = SubPart.from_executable_code('print("hello")', 'PYTHON')
  assert my_part.executable_code.code == 'print("hello")'
  assert my_part.executable_code.language == 'PYTHON'
  assert isinstance(my_part, SubPart)


def test_factory_method_from_code_execution_result_part():
  my_part = SubPart.from_code_execution_result('OUTCOME_OK', 'print("hello")')
  assert my_part.code_execution_result.outcome == 'OUTCOME_OK'
  assert my_part.code_execution_result.output == 'print("hello")'
  assert isinstance(my_part, SubPart)
