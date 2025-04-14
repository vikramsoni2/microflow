# services/workflow/manager.py
from typing import Dict, List, Callable, AsyncGenerator, Any
from .event import WorkflowEvent
import logging

EventHandler = Callable[[WorkflowEvent, Dict[str, Any]], AsyncGenerator[WorkflowEvent, None]]

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
        current_event = initial_event
        yield current_event  # Yield the initial event
        
        while True:
            if current_event.name not in self.handlers:
                self.logger.debug(f"No handlers for event: {current_event.name}")
                break
            handlers = self.handlers[current_event.name]
            try:
                for handler in handlers:
                    async for next_event in handler(current_event, self.ctx):
                        yield next_event
                        if next_event.name == "progress":
                            continue
                        current_event = next_event
                        if next_event.error:
                            break
            except Exception as e:
                self.logger.exception(f"Error processing event {current_event.name}")
                yield WorkflowEvent(
                    name="error",
                    error=str(e),
                    metadata=current_event.metadata
                )
                break