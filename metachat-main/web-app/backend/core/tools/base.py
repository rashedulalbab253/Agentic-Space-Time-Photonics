from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass

@dataclass
class ToolCall:
    tool_name: str
    input_data: Any
    output_data: Any
    metadata: Optional[Dict] = None

class BaseTool(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    async def execute(self, input_data: Any) -> Any:
        """Execute the tool with given input"""
        pass
    
    def format_input(self, input_data: Any) -> str:
        """Format input for LLM consumption"""
        return str(input_data)
    
    def format_output(self, output_data: Any) -> str:
        """Format output for LLM consumption"""
        return str(output_data)