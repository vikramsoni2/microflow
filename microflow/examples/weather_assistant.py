import asyncio
import aiohttp
import os
from typing import Dict, Any, AsyncGenerator
from openai import AsyncOpenAI
from microflow import WorkflowManager, WorkflowEvent

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