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

"""Backward compatibility shim for tests.

This module re-exports all testing utilities from the new public location
at google.adk.testing. Tests should continue to import from here for now,
but new code should import from google.adk.testing directly.
"""

# Re-export commonly used types that tests access via testing_utils
from google.adk.agents.run_config import RunConfig
from google.adk.events.event import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions.session import Session
# Re-export everything from the new public testing module
from google.adk.testing import append_user_content
from google.adk.testing import create_invocation_context
from google.adk.testing import create_test_agent
from google.adk.testing import END_OF_AGENT
from google.adk.testing import get_user_content
from google.adk.testing import InMemoryRunner
from google.adk.testing import MockLlmConnection
from google.adk.testing import MockModel
from google.adk.testing import ModelContent
from google.adk.testing import simplify_content
from google.adk.testing import simplify_contents
from google.adk.testing import simplify_events
from google.adk.testing import simplify_resumable_app_events
from google.adk.testing import TestInMemoryRunner
from google.adk.testing import UserContent

__all__ = [
    'MockModel',
    'MockLlmConnection',
    'InMemoryRunner',
    'TestInMemoryRunner',
    'create_test_agent',
    'create_invocation_context',
    'append_user_content',
    'simplify_events',
    'simplify_resumable_app_events',
    'simplify_contents',
    'simplify_content',
    'get_user_content',
    'UserContent',
    'ModelContent',
    'END_OF_AGENT',
    # Commonly used types
    'RunConfig',
    'Event',
    'LlmRequest',
    'LlmResponse',
    'Session',
]
