
# Microflow API Reference

## WorkflowEvent

```python
from dataclasses import dataclass
from typing import Any, Optional, Dict

@dataclass
class WorkflowEvent:
    name: str 
    data: Dict[str, Any] = None  
    metadata: Dict[str, Any] = None  
    error: Optional[str] = None  
```

The `WorkflowEvent` class represents events that flow through the workflow system.

### Parameters

- `name` (str): The event type identifier
- `data` (Dict[str, Any], optional): The event payload
- `metadata` (Dict[str, Any], optional): Context information
- `error` (Optional[str], optional): Error information if applicable

### Methods

#### `progress(step_name: str, description: str, metadata: Dict[str, Any] = None) -> WorkflowEvent`

Creates a progress event.

- `step_name` (str): The name of the step reporting progress
- `description` (str): A description of the progress
- `metadata` (Dict[str, Any], optional): Additional metadata

Returns a `WorkflowEvent` with name="progress" and appropriate data.

## WorkflowManager

```python
from typing import Dict, List, Callable, AsyncGenerator, Any

EventHandler = Callable[[WorkflowEvent, Dict[str, Any]], AsyncGenerator[WorkflowEvent, None]]

class WorkflowManager:
    def __init__(self, ctx: Dict[str, Any] = None):
        ...
    
    def register(self, event_name: str, handler: EventHandler):
        ...
    
    async def process(self, initial_event: WorkflowEvent) -> AsyncGenerator[WorkflowEvent, None]:
        ...
```

The `WorkflowManager` class orchestrates the flow of events through registered handlers.

### Parameters

- `ctx` (Dict[str, Any], optional): A context dictionary accessible to all handlers

### Methods

#### `register(event_name: str, handler: EventHandler) -> WorkflowManager`

Registers a handler function for a specific event type.

- `event_name` (str): The event type to handle
- `handler` (EventHandler): The handler function

Returns the `WorkflowManager` instance for chaining.

#### `process(initial_event: WorkflowEvent) -> AsyncGenerator[WorkflowEvent, None]`

Processes an initial event through the workflow.

- `initial_event` (WorkflowEvent): The event to start the workflow with

Yields all events produced during the workflow, including the initial event, progress events, and any error events.

## EventHandler

```python
EventHandler = Callable[[WorkflowEvent, Dict[str, Any]], AsyncGenerator[WorkflowEvent, None]]
```

An event handler is an async function that takes a `WorkflowEvent` and a context dictionary, and yields one or more events.

### Parameters

- `event` (WorkflowEvent): The event to handle
- `ctx` (Dict[str, Any]): The workflow context

### Returns

An async generator that yields `WorkflowEvent` instances.



