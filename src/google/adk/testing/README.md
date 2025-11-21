# ADK Testing Utilities

This module provides testing utilities for developers building agents with the Agent Development Kit (ADK). These tools make it easy to write unit tests for your agents without requiring real LLM API calls.

## Quick Start

```python
from google.adk import Agent
from google.adk.testing import MockModel, InMemoryRunner

# Create an agent with a mock model
agent = Agent(
    name="test_agent",
    model=MockModel.create(responses=["Hello, I'm a test response!"]),
    instruction="You are a helpful assistant."
)

# Run the agent in a test environment
runner = InMemoryRunner(root_agent=agent)
events = runner.run("Hi there!")

# Assert on the results
assert len(events) > 0
```

## Key Components

### MockModel

A mock LLM that returns pre-defined responses instead of calling a real API. This makes tests fast, deterministic, and doesn't require API keys.

```python
from google.adk.testing import MockModel

# Simple text responses
mock = MockModel.create(responses=["Response 1", "Response 2"])

# Multiple responses for a conversation
mock = MockModel.create(responses=[
    "First response",
    "Second response", 
    "Third response"
])

# Mock an error
mock = MockModel.create(responses=[], error=ValueError("API Error"))
```

### InMemoryRunner

A test runner that uses in-memory services for fast, isolated tests.

```python
from google.adk.testing import InMemoryRunner

runner = InMemoryRunner(root_agent=agent)

# Synchronous execution
events = runner.run("Test message")

# Async execution
events = await runner.run_async("Test message")

# Access the session
session = runner.session
```

### Helper Functions

#### simplify_events()

Simplify events for easier assertions in tests:

```python
from google.adk.testing import simplify_events

events = runner.run("Hello")
simplified = simplify_events(events)

# Returns list of (author, simplified_content) tuples
# Easier to assert: [("user", "Hello"), ("test_agent", "Hi there!")]
```

#### create_test_agent()

Create a basic test agent quickly:

```python
from google.adk.testing import create_test_agent

agent = create_test_agent(name="my_test")
```

#### create_invocation_context()

Create a test invocation context for testing agent components:

```python
from google.adk.testing import create_invocation_context

context = await create_invocation_context(
    agent=agent,
    user_content="Test message",
    run_config=RunConfig(),
    plugins=[]
)
```

## Complete Example

```python
import pytest
from google.adk import Agent
from google.adk.testing import MockModel, InMemoryRunner, simplify_events

@pytest.fixture
def mock_agent():
    """Create a test agent with mock responses."""
    mock_model = MockModel.create(responses=[
        "I can help with that!",
        "Here's the information you requested.",
        "Is there anything else?"
    ])
    
    return Agent(
        name="test_assistant",
        model=mock_model,
        instruction="You are a helpful assistant."
    )

def test_basic_conversation(mock_agent):
    """Test a basic conversation flow."""
    runner = InMemoryRunner(root_agent=mock_agent)
    
    # First message
    events = runner.run("Can you help me?")
    simplified = simplify_events(events)
    
    assert len(simplified) >= 2
    assert simplified[0][0] == "user"
    assert simplified[0][1] == "Can you help me?"
    
    # Second message in same session
    events = runner.run("Thanks!")
    simplified = simplify_events(events)
    
    assert len(simplified) >= 2

@pytest.mark.asyncio
async def test_async_execution(mock_agent):
    """Test async agent execution."""
    runner = InMemoryRunner(root_agent=mock_agent)
    
    events = await runner.run_async("Hello!")
    assert len(events) > 0

def test_error_handling():
    """Test error handling with mock models."""
    error_model = MockModel.create(
        responses=[], 
        error=ValueError("API unavailable")
    )
    
    agent = Agent(name="error_agent", model=error_model)
    runner = InMemoryRunner(root_agent=agent)
    
    with pytest.raises(ValueError, match="API unavailable"):
        runner.run("This will fail")
```

## API Reference

### MockModel

- `MockModel.create(responses, error=None)` - Create a mock model with pre-defined responses
- `model.requests` - List of all LlmRequest objects sent to the mock
- `model.responses` - List of LlmResponse objects the mock will return
- `model.response_index` - Current position in the responses list

### InMemoryRunner

- `InMemoryRunner(root_agent, plugins=[], app=None)` - Create a test runner
- `runner.run(message)` - Run agent synchronously, returns list of Events
- `runner.run_async(message, invocation_id=None)` - Run agent asynchronously
- `runner.session` - Access the current test session

### Helper Functions

- `create_test_agent(name)` - Create a simple test agent
- `create_invocation_context(agent, user_content, run_config, plugins)` - Create test context
- `simplify_events(events)` - Simplify events for assertions
- `simplify_content(content)` - Simplify content for assertions
- `append_user_content(context, parts)` - Add user content to context

## Best Practices

1. **Use MockModel for unit tests** - Fast, deterministic, no API calls needed
2. **Use InMemoryRunner** - Isolated test environment with in-memory services
3. **Use simplify_events() for assertions** - Makes test assertions cleaner
4. **Test both sync and async paths** - If your agent supports both
5. **Test error handling** - Use MockModel.create() with error parameter
6. **Keep tests isolated** - Each test should use its own runner instance

## Integration Testing

For integration tests with real LLM APIs, use the regular `Runner` instead:

```python
from google.adk import Agent, Runner
from google.adk.sessions import InMemorySessionService

# Real agent for integration testing
agent = Agent(
    name="real_agent",
    model="gemini-2.0-flash-exp",
    instruction="You are a helpful assistant."
)

runner = Runner(
    agent=agent,
    session_service=InMemorySessionService()
)

# This will make real API calls
events = list(runner.run(
    user_id="test_user",
    session_id="test_session",
    new_message="Hello!"
))
```

## Migration Guide

If you were previously using internal testing utilities from `tests/unittests/testing_utils.py`, you can now import from the public module:

```python
# Old (internal)
from tests.unittests.testing_utils import MockModel, InMemoryRunner

# New (public)
from google.adk.testing import MockModel, InMemoryRunner
```

All functionality remains the same - this just makes the utilities officially supported and available via the PyPI package.

