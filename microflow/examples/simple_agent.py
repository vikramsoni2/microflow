import asyncio
from typing import Dict, Any, AsyncGenerator

from microflow import WorkflowManager, WorkflowEvent

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