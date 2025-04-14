"""
Microflow: A lightweight event-driven workflow system for building AI agents and processing pipelines.
"""

from .event import WorkflowEvent
from .manager import WorkflowManager, EventHandler

__all__ = ["WorkflowEvent", "WorkflowManager", "EventHandler"]
__version__ = "0.1.0"