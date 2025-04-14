import pytest
from microflow import WorkflowEvent

def test_workflow_event_creation():
    """Test basic WorkflowEvent creation and properties."""
    event = WorkflowEvent(name="test_event", data={"key": "value"}, metadata={"user": "123"})
    
