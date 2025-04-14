# Building Event-Driven Agentic Workflows with Just 60 Lines of Python

## TL;DR
Learn how to build a lightweight yet powerful event-driven workflow system for AI agents using just two Python classes. This approach enables flexible, maintainable agent workflows with minimal code while supporting progress reporting, error handling, and streaming responses.

## Introduction

Have you ever tried to build a complex AI agent that needs to perform multiple steps in sequence? Perhaps you've found yourself writing spaghetti code with deeply nested functions, or creating a complex state machine that's difficult to maintain. I faced this exact challenge when building a multi-step AI assistant that needed to:

- Process user queries
- Decide which tools to use
- Call external APIs
- Generate responses based on retrieved information
- Report progress throughout the process

Traditional approaches led to tightly coupled components that were hard to modify and extend. After several iterations, I discovered an elegant solution: a simple event-driven workflow system that requires just 60 lines of Python code.

In this post, I'll show you how to build this system from scratch and demonstrate how it can power sophisticated AI agents with clean, maintainable code.

## Table of Contents

1. [The Problem](#the-problem)
2. [The Solution: Event-Driven Workflows](#the-solution-event-driven-workflows)
3. [Building the Core Components](#building-the-core-components)
   - [The WorkflowEvent Class](#the-workflowevent-class)
   - [The WorkflowManager Class](#the-workflowmanager-class)
4. [Basic Usage Example](#basic-usage-example)
5. [Key Features](#key-features)
6. [Advanced Example: Weather Assistant](#advanced-example-weather-assistant-with-streaming-responses)
7. [How This Example Works](#how-this-example-works)
8. [The Power of This Approach](#the-power-of-this-approach)
9. [Extending the System](#extending-the-system)
10. [Conclusion](#conclusion)

## The Problem

When building AI agents or complex processing systems, you typically need to:

- Chain multiple operations together in a specific sequence
- Allow for conditional branching based on intermediate results
- Handle errors gracefully at each step
- Report progress throughout the process
- Maintain flexibility to modify the workflow as requirements change

Traditional approaches often result in:

- **Deeply nested functions** that are hard to understand and maintain
- **Tightly coupled components** where changing one part affects many others
- **Complex state machines** with numerous edge cases
- **Rigid architectures** that resist modification and extension

What if we could solve these problems with a simple, flexible design pattern?

## The Solution: Event-Driven Workflows

The solution I've found most effective is an event-driven workflow system built around two simple components:

1. **WorkflowEvent** - A data structure representing events flowing through the system
2. **WorkflowManager** - An orchestrator that routes events to appropriate handlers

This approach offers several advantages:

- **Loose coupling** - Components communicate only through events
- **Flexibility** - Easy to add, remove, or modify workflow steps
- **Visibility** - Clear tracking of progress and state
- **Simplicity** - Minimal code required to implement

Let's see how to build this system from the ground up.

## Building the Core Components

### The WorkflowEvent Class

At the heart of our system is the `WorkflowEvent` class - a simple data container that carries information between workflow steps:

```python src/workflow/event.py
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class WorkflowEvent:
    name: str                      # Event type identifier
    data: Dict[str, Any] = None    # Event payload
    metadata: Dict[str, Any] = None  # Context information
    error: Optional[str] = None    # Error information if applicable
    
    @classmethod
    def progress(cls, step_name: str, description: str, metadata: Dict[str, Any] = None):
        """Create a progress event"""
        return cls(
            name="progress",
            data={
                "step": step_name,
                "description": description
            },
            metadata=metadata
        )
```

This simple dataclass serves as the message format for all communication in our workflow. The special `progress` method creates standardized progress update events, making it easy to report status throughout the process.

> **Key Point**: By standardizing on a single event format, we create a consistent interface between all components in our system.

### The WorkflowManager Class

The `WorkflowManager` orchestrates the flow of events through our system:

```python src/workflow/manager.py
import logging
from typing import Dict, List, Any, AsyncGenerator, Callable, Optional

from .event import WorkflowEvent

# Type definition for event handlers
EventHandler = Callable[[WorkflowEvent, Dict[str, Any]], AsyncGenerator[WorkflowEvent, None]]

class WorkflowManager:
    def __init__(self, ctx: Dict[str, Any] = None):
        self.ctx = ctx or {}
        self.handlers: Dict[str, List[EventHandler]] = {}
        self.logger = logging.getLogger("WorkflowManager")
    
    def register(self, event_name: str, handler: EventHandler):
        """Register a handler for a specific event type"""
        if event_name not in self.handlers:
            self.handlers[event_name] = []
        self.handlers[event_name].append(handler)
        return self  # Allow chaining
    
    async def process(self, initial_event: WorkflowEvent) -> AsyncGenerator[WorkflowEvent, None]:
        """Process an event through the workflow"""
        events_to_process = [initial_event]
        
        while events_to_process:
            current_event = events_to_process.pop(0)
            
            # Always yield the current event so the caller can see it
            yield current_event
            
            # Skip further processing for progress and error events
            if current_event.name in ["progress", "error"]:
                continue
                
            # Find handlers for this event type
            handlers = self.handlers.get(current_event.name, [])
            if not handlers:
                self.logger.warning(f"No handlers registered for event: {current_event.name}")
                continue
                
            # Process the event through each handler
            for handler in handlers:
                try:
                    # Call the handler and collect new events
                    async for new_event in handler(current_event, self.ctx):
                        events_to_process.append(new_event)
                except Exception as e:
                    # Convert exceptions to error events
                    error_event = WorkflowEvent(
                        name="error",
                        error=f"Handler error: {str(e)}",
                        metadata=current_event.metadata
                    )
                    events_to_process.append(error_event)
                    self.logger.exception(f"Error in handler for {current_event.name}")
```

The manager maintains a registry of handlers for different event types and processes events by finding and executing the appropriate handlers. It also handles errors gracefully by converting exceptions into error events.

> **Checkpoint**: At this point, we have our two core components - a data structure for events and a manager to route them. Now let's see how to use them.

## Basic Usage Example

Let's build a simple AI agent workflow that:
1. Receives a user query
2. Searches for information
3. Generates a response

```python src/examples/simple_agent.py
import asyncio
from typing import Dict, Any, AsyncGenerator

from workflow.manager import WorkflowManager
from workflow.event import WorkflowEvent

# Create our workflow manager
workflow = WorkflowManager()

# Define handlers
async def query_handler(event: WorkflowEvent, ctx: Dict[str, Any]) -> AsyncGenerator[WorkflowEvent, None]:
    query = event.data["query"]
    
    # Report progress
    yield WorkflowEvent.progress("query", f"Processing query: {query}")
    
    # Continue to search step
    yield WorkflowEvent(
        name="search",
        data={"query": query},
        metadata=event.metadata
    )

async def search_handler(event: WorkflowEvent, ctx: Dict[str, Any]) -> AsyncGenerator[WorkflowEvent, None]:
    query = event.data["query"]
    
    yield WorkflowEvent.progress("search", f"Searching for: {query}")
    
    # Simulate search operation
    search_results = ["Result 1", "Result 2", "Result 3"]
    
    yield WorkflowEvent(
        name="generate",
        data={"query": query, "search_results": search_results},
        metadata=event.metadata
    )

async def generate_handler(event: WorkflowEvent, ctx: Dict[str, Any]) -> AsyncGenerator[WorkflowEvent, None]:
    query = event.data["query"]
    search_results = event.data["search_results"]
    
    yield WorkflowEvent.progress("generate", "Generating response")
    
    # Simulate response generation
    response = f"Based on {len(search_results)} results, the answer to '{query}' is..."
    
    yield WorkflowEvent(
        name="completed",
        data={"query": query, "response": response},
        metadata=event.metadata
    )

# Register handlers
workflow.register("query", query_handler)
workflow.register("search", search_handler)
workflow.register("generate", generate_handler)

# Run the workflow
async def main():
    initial_event = WorkflowEvent(
        name="query",
        data={"query": "How does photosynthesis work?"},
        metadata={"user_id": "user123"}
    )
    
    async for event in workflow.process(initial_event):
        if event.name == "progress":
            print(f"Progress: {event.data['step']} - {event.data['description']}")
        elif event.name == "completed":
            print(f"Final response: {event.data['response']}")
        elif event.name == "error":
            print(f"Error: {event.error}")

if __name__ == "__main__":
    asyncio.run(main())
```

This example demonstrates a simple three-step workflow. Each handler:
1. Reports progress
2. Performs its specific task
3. Yields a new event to continue the workflow

> **Try it yourself**: Run this example and observe how events flow through the system. Try adding a deliberate error in one of the handlers to see how error handling works.

## Key Features

Now that you've seen a basic example, let's explore the key features that make this system powerful.

### 1. Event-Driven Architecture

The system is entirely event-driven. Each handler processes an event and produces new events, creating a natural flow of operations:

```
query → search → generate → completed
```

This creates a clean separation of concerns, where each handler focuses on a specific task.

### 2. Progress Reporting

Special handling for `progress` events allows for detailed status updates without disrupting the workflow:

```python
async def complex_handler(event: WorkflowEvent, ctx: Dict[str, Any]) -> AsyncGenerator[WorkflowEvent, None]:
    # Report progress without changing workflow direction
    yield WorkflowEvent.progress("step1", "Starting processing...")
    
    # Do some work...
    
    yield WorkflowEvent.progress("step2", "Halfway done...")
    
    # Continue the workflow with a new event
    yield WorkflowEvent(name="next_step", data={"result": "success"})
```

Progress events are passed through to the caller but don't trigger additional handlers, making it easy to provide visibility into the workflow.

### 3. Error Handling

Errors are propagated through the workflow as special events:

```python
async def risky_handler(event: WorkflowEvent, ctx: Dict[str, Any]) -> AsyncGenerator[WorkflowEvent, None]:
    try:
        # Risky operation...
        result = "operation result"
        yield WorkflowEvent(name="success", data={"result": result})
    except Exception as e:
        yield WorkflowEvent(name="error", error=str(e), metadata=event.metadata)
```

The workflow manager also catches exceptions from handlers and converts them to error events, ensuring that errors don't crash the entire workflow.

### 4. Multiple Handlers Per Event

You can register multiple handlers for the same event type, and they'll run in sequence:

```python
# Register both handlers for the same event
workflow.register("process_data", validation_handler)
workflow.register("process_data", transformation_handler)
```

This allows you to separate concerns like validation and transformation while maintaining a clean workflow.

### 5. Shared Context

The workflow manager can maintain a shared context accessible to all handlers:

```python
# Create a workflow with shared context
workflow = WorkflowManager(ctx={
    "api_key": "secret_key",
    "user_preferences": {"language": "en", "units": "metric"}
})
```

This context can store configuration, credentials, or any other data that needs to be shared across handlers.

> **Key Point**: These features combine to create a system that's both simple and powerful, capable of handling complex workflows with minimal code.

## Advanced Example: Weather Assistant with Streaming Responses

Let's build a more practical example that demonstrates how this system can handle real-world requirements:

1. Analyzes a user query
2. Fetches real weather data from OpenWeatherMap API
3. Generates a streaming response using OpenAI's API

```python src/examples/weather_assistant.py
import asyncio
import aiohttp
import os
from typing import Dict, Any, AsyncGenerator
from openai import AsyncOpenAI
from workflow.manager import WorkflowManager
from workflow.event import WorkflowEvent

# Weather API client - simplified
async def fetch_weather(location: str, api_key: str) -> Dict[str, Any]:
    """Fetch weather data from OpenWeatherMap API"""
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": location, "appid": api_key, "units": "metric"}
    
    async with aiohttp.ClientSession() as session:
        async with session.get(base_url, params=params) as response:
            if response.status != 200:
                raise Exception(f"Weather API error: {response.status}")
            
            data = await response.json()
            return {
                "location": data["name"],
                "country": data["sys"]["country"],
                "temperature": data["main"]["temp"],
                "condition": data["weather"][0]["description"],
                "humidity": data["main"]["humidity"],
                "wind_speed": data["wind"]["speed"]
            }

# Define workflow handlers
async def query_analyzer(event: WorkflowEvent, ctx: Dict[str, Any]) -> AsyncGenerator[WorkflowEvent, None]:
    """Analyzes the query to determine if it's weather-related"""
    query = event.data["query"].lower()
    
    yield WorkflowEvent.progress("analysis", "Analyzing query")
    
    # Simple keyword-based routing
    weather_keywords = ["weather", "temperature", "forecast", "rain", "sunny"]
    
    if any(word in query for word in weather_keywords):
        # Extract location (simplified)
        words = query.split()
        location = "London"  # Default
        
        for i, word in enumerate(words):
            if word in ["in", "for"] and i < len(words) - 1:
                location = words[i+1].rstrip(",.!?")
                break
        
        yield WorkflowEvent(
            name="weather_lookup",
            data={"query": event.data["query"], "location": location},
            metadata=event.metadata
        )
    else:
        # Direct to response generation
        yield WorkflowEvent(
            name="generate_response",
            data={"query": event.data["query"], "context": {}},
            metadata=event.metadata
        )

async def weather_handler(event: WorkflowEvent, ctx: Dict[str, Any]) -> AsyncGenerator[WorkflowEvent, None]:
    """Fetches weather data and passes to response generator"""
    location = event.data["location"]
    
    yield WorkflowEvent.progress("weather", f"Fetching weather for {location}")
    
    try:
        weather_data = await fetch_weather(location, ctx["openweathermap_api_key"])
        
        yield WorkflowEvent(
            name="generate_response",
            data={
                "query": event.data["query"],
                "context": {"weather": weather_data}
            },
            metadata=event.metadata
        )
    except Exception as e:
        yield WorkflowEvent(name="error", error=str(e), metadata=event.metadata)

async def response_generator(event: WorkflowEvent, ctx: Dict[str, Any]) -> AsyncGenerator[WorkflowEvent, None]:
    """Generates a streaming response using OpenAI API"""
    query = event.data["query"]
    context = event.data["context"]
    
    yield WorkflowEvent.progress("generate", "Creating response")
    
    # Initialize OpenAI client
    client = AsyncOpenAI(api_key=ctx["openai_api_key"])
    
    # Prepare messages based on context
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    
    if "weather" in context:
        w = context["weather"]
        messages.append({
            "role": "user", 
            "content": (
                f"Query: {query}\n\n"
                f"Weather in {w['location']}, {w['country']}:\n"
                f"Temperature: {w['temperature']}°C\n"
                f"Condition: {w['condition']}\n"
                f"Humidity: {w['humidity']}%\n"
                f"Wind: {w['wind_speed']} m/s\n\n"
                f"Please answer the query using this weather information."
            )
        })
    else:
        messages.append({"role": "user", "content": query})
    
    try:
        # Create the streaming completion
        stream = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True
        )
        
        # Yield a single event with the stream object
        yield WorkflowEvent(
            name="stream",
            data={
                "query": query,
                "stream": stream,  # Pass the stream object directly
                "context_type": "weather" if "weather" in context else "none"
            },
            metadata=event.metadata
        )
        
    except Exception as e:
        yield WorkflowEvent(name="error", error=str(e), metadata=event.metadata)

# Create our workflow manager with API keys
workflow = WorkflowManager(ctx={
    "openweathermap_api_key": os.environ.get("OPENWEATHERMAP_API_KEY", "your_weather_api_key"),
    "openai_api_key": os.environ.get("OPENAI_API_KEY", "your_openai_api_key")
})

# Register handlers
workflow.register("query", query_analyzer)
workflow.register("weather_lookup", weather_handler)
workflow.register("generate_response", response_generator)

# Run the workflow
async def main():
    # Process a weather query
    query = "What's the weather like in Paris today?"
    
    initial_event = WorkflowEvent(
        name="query",
        data={"query": query},
        metadata={"user_id": "user123"}
    )
    
    print(f"Processing: '{query}'")
    print("=" * 50)
    
    async for event in workflow.process(initial_event):
        if event.name == "progress":
            print(f"Progress: {event.data['step']} - {event.data['description']}")
        
        elif event.name == "stream":
            print("\nStreaming response:")
            print("-" * 50)
            
            # Handle the stream directly here
            full_response = ""
            stream = event.data["stream"]
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    print(content, end="", flush=True)
            
            print("\n" + "-" * 50)
            print(f"Context type: {event.data['context_type']}")
            
        elif event.name == "error":
            print(f"\nError: {event.error}")

if __name__ == "__main__":
    asyncio.run(main())
```

> **Checkpoint**: This example demonstrates how our simple workflow system can handle complex real-world requirements like API calls, conditional routing, and streaming responses.

## How This Example Works

Let's break down how our weather assistant workflow operates:

### 1. The Event Flow

Our workflow follows a clear sequence of events:

```
query → weather_lookup (if weather-related) → generate_response → stream
```

Each step produces events that trigger the next handler in the chain, creating a seamless flow of operations.

### 2. Query Analysis

The workflow begins with the `query_analyzer` handler, which:
- Takes the user's question as input
- Determines if it's weather-related by checking for keywords
- Extracts a location if present
- Routes to either `weather_lookup` or directly to `generate_response`

This demonstrates how our workflow can implement conditional branching based on the content of events.

### 3. Weather Data Retrieval

If the query is weather-related, the `weather_handler`:
- Makes an API call to OpenWeatherMap
- Formats the weather data into a clean structure
- Passes this data as context to the response generator
- Handles any API errors gracefully

This shows how our workflow can integrate with external services while maintaining clean error handling.

### 4. Response Generation

The `response_generator` handler:
- Takes the query and any context (weather data)
- Constructs appropriate messages for the OpenAI API
- Creates a streaming completion request
- Yields a single `stream` event containing the stream object

This demonstrates how our workflow can handle streaming data sources.

### 5. Stream Consumption

The consumer (our `main()` function) then:
- Receives the `stream` event
- Extracts the stream object
- Processes the chunks directly
- Displays the content as it arrives

This approach gives the consumer full control over how to handle the streaming data.

## The Power of This Approach

What makes this design particularly elegant is how it combines:

### 1. Separation of Concerns

Each handler has a single, well-defined responsibility:
- Query analyzer: Determine intent and route accordingly
- Weather handler: Fetch and format weather data
- Response generator: Create the streaming response

This makes the code easier to understand, test, and maintain.

### 2. Flexible Data Flow

The workflow can adapt based on the query:
- Weather queries follow the complete path
- Other queries skip the weather lookup step
- Additional tools could be easily added as new branches

This flexibility allows the system to grow and evolve over time.

### 3. Dynamic Routing

Since it's completely event-driven, it provides dynamic routing capability:
- Handlers can call any other handler in any sequence
- Two or more handlers can talk back and forth until a certain condition is met
- The workflow can branch and merge as needed

This allows for complex decision-making without complex code.

### 4. Progress Reporting

Each step reports its progress, providing visibility into the workflow:
- Users can see what's happening at each stage
- Debugging is easier with clear progress indicators
- Long-running operations can provide feedback

This improves both the user experience and developer experience.

### 5. Context Preservation

The original query and metadata are preserved throughout the entire process:
- User information stays attached to all events
- The original query is available at every step
- Additional context can be added as the workflow progresses

This ensures that all handlers have access to the information they need.

## Extending the System

The beauty of this design is how easily it can be extended:

### Add New Tools

Create handlers for additional APIs or services:
```python
async def news_handler(event: WorkflowEvent, ctx: Dict[str, Any]) -> AsyncGenerator[WorkflowEvent, None]:
    # Fetch news related to the query
    # ...
    yield WorkflowEvent(name="generate_response", data={"context": {"news": news_data}})
```

### Enhance Analysis

Improve the query analyzer with more sophisticated NLP:
```python
async def nlp_analyzer(event: WorkflowEvent, ctx: Dict[str, Any]) -> AsyncGenerator[WorkflowEvent, None]:
    # Use a language model to determine intent
    # ...
    yield WorkflowEvent(name=detected_intent, data=event.data)
```

### Add Memory

Store conversation history in the context:
```python
async def memory_handler(event: WorkflowEvent, ctx: Dict[str, Any]) -> AsyncGenerator[WorkflowEvent, None]:
    # Add conversation history to context
    if "history" not in ctx:
        ctx["history"] = []
    
    ctx["history"].append({"query": event.data["query"]})
    
    # Continue with original event
    yield event
```

> **Try it yourself**: Pick one of these extensions and implement it in the weather assistant example. How does it change the workflow?

## Conclusion

With just 60 lines of Python code, we've created a flexible, event-driven workflow system that can power complex AI agent interactions. This approach offers several key benefits:

- **Simplicity**: The core system is just two classes
- **Flexibility**: Easy to add, remove, or modify workflow steps
- **Visibility**: Clear tracking of progress and state
- **Resilience**: Graceful error handling throughout the workflow
- **Extensibility**: Simple to add new capabilities

This pattern works especially well for AI applications where operations are often asynchronous, real-time streaming is important, and the workflow might evolve over time.

The next time you find yourself building a complex AI agent or processing pipeline, consider this event-driven approach. It might just save you from a tangled web of nested functions and state management headaches.

### What's Next?

- Try implementing this pattern in your own projects
- Experiment with different types of workflows and handlers
- Consider how you might extend the system for your specific needs

Remember, sometimes the simplest solutions are the most powerful!