This directory contains the BaxiCHAT Agentic workflow. Which is driving the entire BaciCHAT inference flow.
To see how it works start with the file

```
config.py
```

for more information about the Workflow system, read below:


# Event-Driven Workflow System

This system implements an event-driven workflow architecture where multiple handlers can process events in sequence, providing a flexible way to build complex processing pipelines.

## Core Components

the entire framework is just 80 lines of python code, located in this repository in two files.

- src/workflow/manager.py
- src/workflow/event.py

### WorkflowEvent

The fundamental data structure that flows through the system:

```python
@dataclass
class WorkflowEvent:
    name: str                      # Event type identifier
    data: Dict[str, Any] = None    # Event payload
    metadata: Dict[str, Any] = None  # Context information
    error: Optional[str] = None    # Error information if applicable
```

Special events include:
- `progress` events: For reporting status updates
- `error` events: For handling failures

### WorkflowManager

Orchestrates the flow of events through registered handlers:

```python
class WorkflowManager:
    def register(self, event_name: str, handler: EventHandler) -> Self
    async def process(self, initial_event: WorkflowEvent) -> AsyncGenerator[WorkflowEvent, None]
```

## How It Works

### Registering Handlers

Handlers are registered for specific event types:

```python
workflow = WorkflowManager()

# Register a handler for the "start" event
workflow.register("start", start_handler)

# Register multiple handlers for the same event (they'll run in sequence)
workflow.register("process_data", validation_handler)
workflow.register("process_data", transformation_handler)

# Chain registration is supported
workflow.register("event1", handler1).register("event2", handler2)
```

### Creating Event Handlers

An event handler is an async function that takes a `WorkflowEvent` and yields one or more events:

```python
async def search_handler(event: WorkflowEvent) -> AsyncGenerator[WorkflowEvent, None]:
    # Report progress
    yield WorkflowEvent.progress("search", "Searching for information...")
    
    # Perform work
    search_results = await perform_search(event.data["query"])
    
    # Return results with a new event type
    yield WorkflowEvent(
        name="search_completed",
        data={"search_results": search_results},
        metadata=event.metadata
    )
```

### Chaining Handlers

Handlers chain together by yielding events that trigger other handlers:

1. Handler A processes an event and yields a new event with name="process_data"
2. WorkflowManager sees this event and finds handlers registered for "process_data"
3. Those handlers run in sequence, potentially yielding more events

### Progress Reporting

Special handling for progress events allows for status updates without breaking the chain:

```python
async def complex_handler(event: WorkflowEvent):
    # Report progress without changing workflow direction
    yield WorkflowEvent.progress("step1", "Starting processing...")
    
    # Do some work
    await some_operation()
    
    yield WorkflowEvent.progress("step2", "Halfway done...")
    
    # Do more work
    await another_operation()
    
    # Continue the workflow with a new event
    yield WorkflowEvent(name="next_step", data={"result": "success"})
```

### Error Handling

Errors can be propagated through the workflow:

```python
async def risky_handler(event: WorkflowEvent):
    try:
        result = await risky_operation()
        yield WorkflowEvent(name="success", data={"result": result})
    except Exception as e:
        yield WorkflowEvent(name="error", error=str(e), metadata=event.metadata)
```

## Example Workflow

```python
import asyncio
from src.workflow.event  import WorkflowManager
from src.workflow.event import WorkflowEvent
from typing import AsyncGenerator

# Create a workflow
workflow = WorkflowManager()

# Define some simple handlers
async def greeting_handler(event: WorkflowEvent) -> AsyncGenerator[WorkflowEvent, None]:
    print(f"Greeting: Hello, {event.data['name']}!")
    
    # Report progress
    yield WorkflowEvent.progress("greeting", "Sending welcome message")
    
    # Continue to the next step
    yield WorkflowEvent(
        name="process_user",
        data={"name": event.data["name"], "action": "welcome"},
        metadata=event.metadata
    )

async def process_user_handler(event: WorkflowEvent) -> AsyncGenerator[WorkflowEvent, None]:
    print(f"Processing user: {event.data['name']} with action: {event.data['action']}")
    
    # Report progress
    yield WorkflowEvent.progress("processing", "Updating user records")
    
    # Complete the workflow
    yield WorkflowEvent(
        name="completed",
        data={"result": f"User {event.data['name']} processed successfully"},
        metadata=event.metadata
    )

# Register the handlers
workflow.register("start", greeting_handler)
workflow.register("process_user", process_user_handler)

# Start the workflow with an initial event
initial_event = WorkflowEvent(
    name="start",
    data={"name": "Alice"},
    metadata={"session_id": "12345"}
)

# Process the workflow and handle events
async def main():
    async for event in workflow.process(initial_event):
        if event.name == "progress":
            print(f"Progress update: {event.data['step']} - {event.data['description']}")
        elif event.name == "completed":
            print(f"Workflow completed: {event.data['result']}")
        elif event.name == "error":
            print(f"Error occurred: {event.error}")


# Run the main function as a stanalone script:
if __name__ == "__main__":
    asyncio.run(main())
```

## Testing Workflow Framework
```sh
pytest -xvs ./tests/test_workflow.py
```

## UV Environment Setup
For testing use:
```sh
uv sync
```

For deployment use. Will not start with testing dependencies.
```sh
uv sync --no-dev
```


## Benefits

- **Decoupling**: Components communicate through events without direct dependencies
- **Extensibility**: Easy to add new handlers or modify workflow without changing existing code
- **Observability**: Progress events provide visibility into the workflow state
- **Error Handling**: Centralized error management through error events

