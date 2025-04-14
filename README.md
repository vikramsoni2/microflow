# Microflow

A lightweight event-driven workflow system for building AI agents and processing pipelines in just 60 lines of Python.

## Installation

```bash
pip install microflow
```

## Quick Start

```python
import asyncio
from microflow import WorkflowManager, WorkflowEvent

# Create a workflow manager
workflow = WorkflowManager()

# Define a handler
async def greeting_handler(event, ctx):
    name = event.data.get("name", "World")
    yield WorkflowEvent.progress("greeting", f"Processing greeting for {name}")
    yield WorkflowEvent(name="completed", data={"message": f"Hello, {name}!"})

# Register the handler
workflow.register("greet", greeting_handler)

# Run the workflow
async def main():
    initial_event = WorkflowEvent(name="greet", data={"name": "Alice"})
    
    async for event in workflow.process(initial_event):
        if event.name == "progress":
            print(f"Progress: {event.data['step']} - {event.data['description']}")
        elif event.name == "completed":
            print(f"Result: {event.data['message']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Features

- **Event-Driven Architecture**: Chain operations through events
- **Progress Reporting**: Built-in support for progress updates
- **Error Handling**: Graceful error propagation
- **Flexible Workflows**: Easily modify and extend workflows
- **Minimal Dependencies**: No external dependencies required

## Core Components

The entire framework is just 60 lines of Python code, consisting of two main components:

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
async def search_handler(event: WorkflowEvent, ctx: Dict[str, Any]) -> AsyncGenerator[WorkflowEvent, None]:
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
async def complex_handler(event: WorkflowEvent, ctx: Dict[str, Any]):
    # Report progress without changing workflow direction
    yield WorkflowEvent.progress("step1", "Starting processing...")
    
    # Do some work...
    
    yield WorkflowEvent.progress("step2", "Halfway done...")
    
    # Continue the workflow with a new event
    yield WorkflowEvent(name="next_step", data={"result": "success"})
```

### Error Handling

Errors can be propagated through the workflow:

```python
async def risky_handler(event: WorkflowEvent, ctx: Dict[str, Any]):
    try:
        result = await risky_operation()
        yield WorkflowEvent(name="success", data={"result": result})
    except Exception as e:
        yield WorkflowEvent(name="error", error=str(e), metadata=event.metadata)
```

## Example Workflow

For more complex examples, see the [examples directory](https://github.com/vikramsoni2/microflow/tree/main/microflow/examples) which includes:

1. A simple agent workflow that processes queries, searches for information, and generates responses
2. A weather assistant that analyzes queries, fetches weather data, and generates streaming responses

## Benefits

- **Decoupling**: Components communicate through events without direct dependencies
- **Extensibility**: Easy to add new handlers or modify workflow without changing existing code
- **Observability**: Progress events provide visibility into the workflow state
- **Error Handling**: Centralized error management through error events

## License

MIT
