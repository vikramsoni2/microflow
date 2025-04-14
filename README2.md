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

## Documentation

For more detailed examples and documentation, see the [examples directory](https://github.com/yourusername/microflow/tree/main/microflow/examples).

## License

MIT