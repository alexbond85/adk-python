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

"""Testing utilities for ADK users.

This module provides utilities for testing agents built with the Agent
Development Kit (ADK). It includes mock models, test runners, and helper
functions to make testing agents easier without requiring real LLM API calls.

Example:
  Basic usage with MockModel:

  >>> from google.adk.testing import MockModel, InMemoryRunner, create_test_agent
  >>> from google.adk import Agent
  >>>
  >>> # Create a test agent with a mock model
  >>> agent = Agent(
  ...     name="test_agent",
  ...     model=MockModel.create(responses=["Hello, I'm a test response!"]),
  ...     instruction="You are a helpful assistant."
  ... )
  >>>
  >>> # Run the agent in a test environment
  >>> runner = InMemoryRunner(root_agent=agent)
  >>> events = runner.run("Hi there!")
  >>> assert len(events) > 0

  Testing with multiple responses:

  >>> mock_model = MockModel.create(responses=[
  ...     "First response",
  ...     "Second response",
  ...     "Third response"
  ... ])
  >>> # Each call to the model will return the next response in order

  Using helper functions for assertions:

  >>> from google.adk.testing import simplify_events
  >>> events = runner.run("Test message")
  >>> simplified = simplify_events(events)
  >>> # Makes it easier to assert on event content
"""

from __future__ import annotations

from ..agents.run_config import RunConfig
from ..events.event import Event
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..sessions.session import Session
from .testing_utils import append_user_content
from .testing_utils import create_invocation_context
from .testing_utils import create_test_agent
from .testing_utils import END_OF_AGENT
from .testing_utils import get_user_content
from .testing_utils import InMemoryRunner
from .testing_utils import MockLlmConnection
from .testing_utils import MockModel
from .testing_utils import ModelContent
from .testing_utils import simplify_content
from .testing_utils import simplify_contents
from .testing_utils import simplify_events
from .testing_utils import simplify_resumable_app_events
from .testing_utils import TestInMemoryRunner
from .testing_utils import UserContent

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
