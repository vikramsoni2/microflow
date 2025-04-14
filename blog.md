# Building Event-Driven Agentic Workflows with Just 60 Lines of Python

In the world of AI agents and complex processing pipelines, orchestrating the flow of operations can quickly become challenging. Today, I want to introduce a lightweight yet powerful workflow management system that enables event-driven processing with minimal code.

## The Problem

When building AI agents or complex processing systems, you often need to:
- Chain multiple operations together
- Handle errors gracefully
- Report progress throughout the process
- Maintain flexibility to modify the workflow

Traditional approaches often lead to tightly coupled components or complex state machines. But what if we could solve this with just two simple classes?

## Enter the Event-Driven Workflow System

This system consists of just two components:
1. `WorkflowEvent` - A data structure representing events flowing through the system
2. `WorkflowManager` - An orchestrator that routes events to appropriate handlers

Let's see how it works.

## The WorkflowEvent

```python
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

This simple dataclass carries all the information needed between workflow steps. The special `progress` method creates standardized progress update events.

## The WorkflowManager

```python
class WorkflowManager:
    def __init__(self, ctx: Dict[str, Any] = None):
        self.ctx = ctx
        self.handlers: Dict[str, List[EventHandler]] = {}
        self.logger = logging.getLogger("WorkflowManager")
    
    def register(self, event_name: str, handler: EventHandler):
        if event_name not in self.handlers:
            self.handlers[event_name] = []
        self.handlers[event_name].append(handler)
        return self  # Allow chaining
    
    async def process(self, initial_event: WorkflowEvent) -> AsyncGenerator[WorkflowEvent, None]:
        # Implementation details...
```

The manager maintains a registry of handlers for different event types and processes events by finding and executing the appropriate handlers.

## How to Use It: A Simple Example

Let's build a simple AI agent workflow that:
1. Receives a user query
2. Searches for information
3. Generates a response

```python
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

## Key Features

### 1. Event-Driven Architecture

The system is entirely event-driven. Each handler processes an event and produces new events, creating a natural flow of operations.

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

The workflow manager also catches exceptions from handlers and converts them to error events.

### 4. Multiple Handlers Per Event

You can register multiple handlers for the same event type, and they'll run in sequence:

```python
async def validation_handler(event: WorkflowEvent, ctx: Dict[str, Any]) -> AsyncGenerator[WorkflowEvent, None]:
    # Validate the data
    if not event.data.get("text"):
        yield WorkflowEvent(name="error", error="Missing text field")
        return
    
    # If valid, pass through the same event for the next handler
    yield event

async def transformation_handler(event: WorkflowEvent, ctx: Dict[str, Any]) -> AsyncGenerator[WorkflowEvent, None]:
    # Transform the data
    transformed_text = event.data["text"].upper()
    
    # Continue with transformed data
    yield WorkflowEvent(
        name="transformed",
        data={"original": event.data["text"], "transformed": transformed_text},
        metadata=event.metadata
    )

# Register both handlers for the same event
workflow.register("process_data", validation_handler)
workflow.register("process_data", transformation_handler)
```

### 5. Shared Context

The workflow manager can maintain a shared context accessible to all handlers:

```python
# Create a workflow with shared context
workflow = WorkflowManager(ctx={
    "api_key": "secret_key",
    "user_preferences": {"language": "en", "units": "metric"}
})

async def handler(event: WorkflowEvent, ctx: Dict[str, Any]) -> AsyncGenerator[WorkflowEvent, None]:
    # Access shared context
    api_key = ctx["api_key"]
    language = ctx["user_preferences"]["language"]
    
    # Use the context values in processing
    yield WorkflowEvent.progress("api_call", f"Making API call with language: {language}")
    
    # Continue workflow
    yield WorkflowEvent(name="next_step", data={"processed": True})
```

## Real-World Applications

This simple system is surprisingly versatile for building AI agent workflows:

1. **RAG (Retrieval-Augmented Generation)**: Chain together query understanding, retrieval, and generation steps
2. **Multi-agent systems**: Let different specialized agents handle different parts of a task
3. **Complex processing pipelines**: Break down complex tasks into manageable steps
4. **Workflow orchestration**: Coordinate calls to multiple microservices while maintaining a clean, event-driven architecture.



## Advanced Example: Weather Assistant with Streaming Responses

Let's build a practical example that:
1. Analyzes a user query
2. Fetches real weather data from OpenWeatherMap API
3. Generates a streaming response using OpenAI's API

```python
import asyncio
import aiohttp
import os
from typing import Dict, Any, AsyncGenerator
from datetime import datetime
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
        # Create the streaming completion - KEY SIMPLIFICATION:
        # Instead of yielding each chunk as a separate event,
        # we yield a single event with the stream object
        stream = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True
        )
        
        # Yield a single event with the stream object
        # (you can also yield every chunk directly from this handler)
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

## How This Example Works

Let's break down how our streamlined workflow operates, step by step:

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

This routing logic is simple but effective - in a production system, you might use more sophisticated NLP techniques to understand user intent.

### 3. Weather Data Retrieval

If the query is weather-related, the `weather_handler`:
- Makes an API call to OpenWeatherMap
- Formats the weather data into a clean structure
- Passes this data as context to the response generator
- Handles any API errors gracefully

This demonstrates how our workflow can integrate with external services while maintaining clean error handling.

### 4. Response Generation

The `response_generator` handler:
- Takes the query and any context (weather data)
- Constructs appropriate messages for the OpenAI API
- Creates a streaming completion request
- Yields a single `stream` event containing the stream object

This is where our simplification shines - instead of yielding multiple events for the streaming process, we yield just one event with the stream object itself.

### 5. Stream Consumption

The consumer (our `main()` function) then:
- Receives the `stream` event
- Extracts the stream object
- Processes the chunks directly
- Displays the content as it arrives

This approach gives the consumer full control over how to handle the streaming data, whether that's displaying it, processing it, or storing it.

## The Power of This Approach

What makes this design particularly elegant is how it combines:

### 1. Separation of Concerns

Each handler has a single, well-defined responsibility:
- Query analyzer: Determine intent and route accordingly
- Weather handler: Fetch and format weather data
- Response generator: Create the streaming response

business logic can be separated from handlers if needed.

### 2. Flexible Data Flow

The workflow can adapt based on the query:
- Weather queries follow the complete path
- Other queries skip the weather lookup step
- Additional tools could be easily added as new branches

### 3. Dynamic Routing

Since its completely event driven, it provides dynamic routing capability:
- Handlers can call any other handler in any sequence.
- Two more more handlers can talk back and forth until certain condition met.




### 4. Streaming Responses: 
The workflow handles streaming content from OpenAI, breaking it into manageable chunks.

### 5. Progress Reporting: 
Each step reports its progress, providing visibility into the workflow.


### 6. Context Preservation: 
The original query and metadata are preserved throughout the entire process.




## Extending the System

The beauty of this design is how easily it can be extended:

1. **Add new tools**: Create handlers for additional APIs (news, stocks, translations)
2. **Enhance analysis**: Improve the query analyzer with more sophisticated NLP
3. **Add memory**: Store conversation history in the context


## Conclusion

With just 60 lines of Python code, we've created a flexible, event-driven workflow system that can power complex AI agent interactions. The beauty of this approach is its simplicity and flexibility - you can easily extend it with new handlers, modify the flow, or add new features without disrupting existing code.

This pattern works especially well for AI applications where operations are often asynchronous, real-time streaming, progress reporting is important, and the workflow might evolve over time.

The weather and search agent example demonstrates how you can build sophisticated agentic workflows by:
1. Breaking down complex tasks into discrete steps
2. Using events to coordinate between different specialized tools
3. Maintaining context throughout the workflow
4. Providing detailed progress updates
5. Generating comprehensive responses based on multiple data sources

Give it a try in your next project - sometimes the simplest solutions are the most powerful!