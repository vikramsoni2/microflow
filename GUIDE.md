
# Microflow User Guide

## Installation

```bash
pip install microflow
```

## Basic Usage

### Creating a Simple Workflow

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

### Chaining Handlers

Handlers can be chained together by yielding events that trigger other handlers:

```python
# Define handlers
async def query_handler(event, ctx):
    query = event.data["query"]
    yield WorkflowEvent.progress("query", f"Processing query: {query}")
    yield WorkflowEvent(name="search", data={"query": query})

async def search_handler(event, ctx):
    query = event.data["query"]
    yield WorkflowEvent.progress("search", f"Searching for: {query}")
    results = ["Result 1", "Result 2", "Result 3"]
    yield WorkflowEvent(name="generate", data={"query": query, "results": results})

async def generate_handler(event, ctx):
    query = event.data["query"]
    results = event.data["results"]
    yield WorkflowEvent.progress("generate", "Generating response")
    response = f"Answer to '{query}' based on {len(results)} results"
    yield WorkflowEvent(name="completed", data={"response": response})

# Register handlers
workflow = WorkflowManager()
workflow.register("query", query_handler)
workflow.register("search", search_handler)
workflow.register("generate", generate_handler)
```

### Using Context

The workflow manager can maintain a shared context accessible to all handlers:

```python
# Create a workflow with shared context
workflow = WorkflowManager(ctx={
    "api_key": "secret_key",
    "user_preferences": {"language": "en", "units": "metric"}
})

async def api_handler(event, ctx):
    # Access the API key from context
    api_key = ctx["api_key"]
    # Use the API key...
    yield WorkflowEvent(name="completed", data={"result": "API call successful"})

workflow.register("api_call", api_handler)
```

### Error Handling

Errors can be handled in two ways:

1. By catching exceptions and yielding error events:

```python
async def risky_handler(event, ctx):
    try:
        result = await risky_operation()
        yield WorkflowEvent(name="success", data={"result": result})
    except Exception as e:
        yield WorkflowEvent(name="error", error=str(e), metadata=event.metadata)
```

2. The workflow manager automatically catches exceptions from handlers and converts them to error events.

### Progress Reporting

Special handling for progress events allows for status updates without breaking the chain:

```python
async def complex_handler(event, ctx):
    # Report progress without changing workflow direction
    yield WorkflowEvent.progress("step1", "Starting processing...")
    
    # Do some work...
    
    yield WorkflowEvent.progress("step2", "Halfway done...")
    
    # Continue the workflow with a new event
    yield WorkflowEvent(name="next_step", data={"result": "success"})
```

## Advanced Usage

### Multiple Handlers Per Event

You can register multiple handlers for the same event type:

```python
async def validation_handler(event, ctx):
    # Validate data
    if not event.data.get("value"):
        yield WorkflowEvent(name="error", error="Missing value")
        return
    
    # Pass the event through unchanged
    yield event

async def transformation_handler(event, ctx):
    # Transform data
    transformed_value = event.data["value"].upper()
    
    # Create a new event with transformed data
    yield WorkflowEvent(
        name="transformed",
        data={"value": transformed_value},
        metadata=event.metadata
    )

# Register both handlers for the same event
workflow.register("process_data", validation_handler)
workflow.register("process_data", transformation_handler)
```

### Streaming Responses

For handling streaming responses (like from OpenAI's API):

```python
async def streaming_handler(event, ctx):
    # Set up a streaming API client
    client = AsyncClient(api_key=ctx["api_key"])
    
    # Create a streaming response
    stream = await client.create_stream(prompt=event.data["prompt"])
    
    # Pass the stream object in an event
    yield WorkflowEvent(
        name="stream",
        data={"stream": stream},
        metadata=event.metadata
    )

# Handle the stream in your main function
async def main():
    async for event in workflow.process(initial_event):
        if event.name == "stream":
            stream = event.data["stream"]
            async for chunk in stream:
                print(chunk, end="", flush=True)