from typing import List, Dict, Any, Optional, Union
import openai
from openai import OpenAI
from .base import BaseModel, LLMResponse
import os
import base64

DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL_NAME", "gpt-5.2-2025-12-11")

class OpenAIModel(BaseModel):
    def __init__(self, model_name: str = DEFAULT_OPENAI_MODEL, api_key: Optional[str] = None):
        super().__init__(model_name)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
    
    async def generate(self, messages: List[Dict[str, Any]], temperature: float = 1.0) -> str:
        """
        Generate a response using the OpenAI API
        
        Args:
            messages: List of message dictionaries with role and content
            temperature: Controls randomness in the output
            
        Returns:
            Generated text response
        """
        try:
            # Process messages to handle images if present
            processed_messages = []
            
            for message in messages:
                # Check if the message contains image data
                if 'images' in message:
                    # Create a multimodal message
                    content = []
                    
                    # Add text if present
                    if 'content' in message and message['content']:
                        content.append({
                            "type": "text", 
                            "text": message['content']
                        })
                    
                    # Add images
                    for image_data in message['images']:
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            }
                        })
                    
                    processed_messages.append({
                        "role": message["role"],
                        "content": content
                    })
                else:
                    # Regular text message
                    processed_messages.append(message)

            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=processed_messages,
                temperature=temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}"    
    def count_tokens(self, text: str) -> int:
        # Implement token counting logic
        pass
