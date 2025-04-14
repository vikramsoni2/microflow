from dataclasses import dataclass
from typing import Any, Optional, Dict

@dataclass
class WorkflowEvent:
    """Event passed between workflow components"""
    name: str 
    data: Dict[str, Any] = None  
    metadata: Dict[str, Any] = None  
    error: Optional[str] = None  
    
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