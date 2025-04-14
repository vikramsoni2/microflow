import pytest
import asyncio
from typing import Dict, Any, AsyncGenerator
from microflow import WorkflowManager, WorkflowEvent

async def simple_handler(event: WorkflowEvent, ctx: Dict[str, Any]) -> AsyncGenerator[WorkflowEvent, None]:
    """A simple handler that yields a completed event."""
    yield WorkflowEvent(name="completed", data={"result": "success"})

async def progress_handler(event: WorkflowEvent, ctx: Dict[str, Any]) -> AsyncGenerator[WorkflowEvent, None]:
    """A handler that yields progress and then a completed event."""
    yield WorkflowEvent.progress("step1", "Starting")
    yield WorkflowEvent(name="completed", data={"result": "success"})

async def error_handler(event: WorkflowEvent, ctx: Dict[str, Any]) -> AsyncGenerator[WorkflowEvent, None]:
    """A handler that raises an exception."""
    raise ValueError("Test error")

async def error_event_handler(event: WorkflowEvent, ctx: Dict[str, Any]) -> AsyncGenerator[WorkflowEvent, None]:
    """A handler that yields an error event."""
    yield WorkflowEvent(name="error", error="Something went wrong")

async def chained_handler(event: WorkflowEvent, ctx: Dict[str, Any]) -> AsyncGenerator[WorkflowEvent, None]:
    """A handler that yields an event to trigger another handler."""
    yield WorkflowEvent(name="next_step", data={"value": event.data.get("value", 0) + 1})

async def context_handler(event: WorkflowEvent, ctx: Dict[str, Any]) -> AsyncGenerator[WorkflowEvent, None]:
    """A handler that uses the context."""
    api_key = ctx.get("api_key", "default")
    yield WorkflowEvent(name="completed", data={"api_key": api_key})

@pytest.fixture
def workflow():
    """Create a basic workflow manager."""
    return WorkflowManager()

@pytest.fixture
def workflow_with_context():
    """Create a workflow manager with context."""
    return WorkflowManager(ctx={"api_key": "test_key"})

@pytest.mark.asyncio
async def test_simple_workflow(workflow):
    """Test a simple workflow with one handler."""
    workflow.register("start", simple_handler)
    
    events = []
    async for event in workflow.process(WorkflowEvent(name="start")):
        events.append(event)
    
    assert len(events) == 2
    assert events[0].name == "start"
    assert events[1].name == "completed"
    assert events[1].data == {"result": "success"}

@pytest.mark.asyncio
async def test_progress_events(workflow):
    """Test handling of progress events."""
    workflow.register("start", progress_handler)
    
    events = []
    async for event in workflow.process(WorkflowEvent(name="start")):
        events.append(event)
    
    assert len(events) == 3
    assert events[0].name == "start"
    assert events[1].name == "progress"
    assert events[1].data == {"step": "step1", "description": "Starting"}
    assert events[2].name
