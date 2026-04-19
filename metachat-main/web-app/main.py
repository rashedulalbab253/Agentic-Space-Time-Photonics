from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from typing import Optional, AsyncGenerator, List, Dict, Any
from pydantic import BaseModel
import dotenv
from pathlib import Path
from sse_starlette.sse import EventSourceResponse
import asyncio
import json
import tiktoken  # Add this import for token counting
import uuid  # Add this import for UUID generation
from datetime import datetime
from fastapi.responses import FileResponse
import subprocess  # Add this import for the ping endpoint

# Load environment variables
dotenv.load_dotenv()

DEFAULT_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-5.2-2025-12-11")
DESIGN_BASE_PATH = os.getenv("DESIGN_BASE_PATH", "/media/tmp2/metachat-app/backend/tools/design")
MATERIAL_DB_PATH = os.getenv("MATERIAL_DB_PATH", "/media/tmp2/metachat-app/backend/tools/material_db/materials.db")

# Import your agent components
from backend.agent.base import Agent
from backend.core.models.openai import OpenAIModel
from backend.tools.design.api import NeuralDesignAPI
from backend.tools.solvers.scientific_compute import ScientificCompute
from backend.tools.solvers.symbolic_solver import SymbolicSolver
from backend.tools.material_db.query_materials import MaterialDatabaseCLI

# Create FastAPI app
app = FastAPI()

# Configure logging - must be done before uvicorn starts
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True  # Force reconfiguration even if logging was already configured
)
logger = logging.getLogger(__name__)

# Also configure uvicorn's access logger
uvicorn_logger = logging.getLogger("uvicorn.access")
uvicorn_logger.setLevel(logging.INFO)

# Log startup message
logger.info("Starting MetaChat...")

# Enable CORS for development (update `allow_origins` for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific domains for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI model
model = OpenAIModel(
    model_name=DEFAULT_MODEL_NAME,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize MaterialDatabaseCLI
materials_cli = MaterialDatabaseCLI(
    db_path=MATERIAL_DB_PATH,
    model=model,
    debug=True
)

# Initialize agent
from backend.agent.cot_iterative_tools_materials import IterativeAgentToolsMaterials
from backend.agent.standard_agent_tools import StandardAgentTools
# 
agent = IterativeAgentToolsMaterials(model=model)
# agent = StandardAgentTools(model=model)

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

# Add this function to count tokens
def count_tokens(text: str, model: Optional[str] = None) -> int:
    model_to_use = model or DEFAULT_MODEL_NAME
    encoding_model = "gpt-5" if model_to_use.startswith("gpt-5") else model_to_use
    encoding = tiktoken.encoding_for_model(encoding_model)
    return len(encoding.encode(text))

# Add this class to manage conversation history
class ConversationManager:
    def __init__(self, max_tokens: int = 100000):
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        self.max_tokens = max_tokens
        
    def add_message(self, conversation_id: str, role: str, content: str, message_type: str = "chat") -> None:
        """
        Add a message to the conversation history.
        Args:
            conversation_id: The ID of the conversation
            role: The role of the sender (user, assistant, system)
            content: The message content
            message_type: Type of message ("chat" or "internal")
        """
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
            
        message = {
            "role": role,
            "content": content,
            "type": message_type,
            "timestamp": datetime.now().isoformat()
        }
        self.conversations[conversation_id].append(message)
        self._trim_conversation(conversation_id)
    
    def get_conversation(self, conversation_id: str, include_internal: bool = True) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        Args:
            conversation_id: The ID of the conversation
            include_internal: Whether to include internal thoughts
        """
        messages = self.conversations.get(conversation_id, [])
        if not include_internal:
            return [msg for msg in messages if msg.get("type", "chat") == "chat"]
        return messages
    
    def _trim_conversation(self, conversation_id: str) -> None:
        """
        Trim conversation history to stay within token limit.
        Preserves both chat messages and internal thoughts, but trims oldest first.
        """
        messages = self.conversations[conversation_id]
        total_tokens = sum(count_tokens(msg["content"]) for msg in messages)
        
        while total_tokens > self.max_tokens and len(messages) > 1:
            # Always keep the system message if it exists
            start_idx = 1 if messages[0]["role"] == "system" else 0
            removed_msg = messages.pop(start_idx)
            total_tokens -= count_tokens(removed_msg["content"])

# Initialize conversation manager
conversation_manager = ConversationManager()

async def generate_events(agent, message: str, conversation_id: Optional[str] = None) -> AsyncGenerator[dict, None]:
    try:
        # Generate conversation ID if not provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            logger.info(f"Generated new conversation_id: {conversation_id}")
        else:
            logger.info(f"Using existing conversation_id: {conversation_id}")
            
        # Add user message to conversation history
        conversation_manager.add_message(conversation_id, "user", message, "chat")
        logger.debug(f"Added user message to conversation {conversation_id}")
        
        # Get conversation history - include internal thoughts for the agent
        conversation_history = conversation_manager.get_conversation(conversation_id, include_internal=True)
        
        async for status in agent.solve_with_status(
            problem=message,
            problem_id=conversation_id,
            conversation_history=conversation_history
        ):
            if isinstance(status, dict):
                if "status" in status:
                    # Only yield status updates that aren't thinking messages
                    if status["status"] != "Thinking...":
                        yield {
                            "event": "message",
                            "data": json.dumps({"status": status["status"]})
                        }
                elif "type" in status and status["type"] == "plots":
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "plots",
                            "data": status["data"]
                        })
                    }
                elif "solution" in status:
                    # Add assistant's response to conversation history
                    conversation_manager.add_message(conversation_id, "assistant", status["solution"], "chat")
                    logger.info(f"Solution generated for conversation {conversation_id}")
                    
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "solution": status["solution"],
                            "metadata": status.get("metadata", {}),
                            "conversation_id": conversation_id
                        })
                    }
                elif "thinking" in status:
                    # Add internal thought to conversation history
                    conversation_manager.add_message(conversation_id, "assistant", status["thinking"], "internal")
                    
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "internal",
                            "content": status["thinking"]
                        })
                    }
                elif "error" in status:
                    yield {
                        "event": "message",
                        "data": json.dumps({"error": status["error"]})
                    }
            else:
                yield {
                    "event": "message",
                    "data": json.dumps({"status": str(status)})
                }
                
    except Exception as e:  
        logger.error(f"Error processing chat: {str(e)}")
        yield {
            "event": "message",
            "data": json.dumps({"error": str(e)})
        }

@app.get("/chat")
async def chat_get(message: str, conversation_id: Optional[str] = None):
    try:
        logger.info(f"Chat request received - conversation_id: {conversation_id}, message length: {len(message)}")
        return EventSourceResponse(
            generate_events(agent, message, conversation_id)
        )
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_post(request: ChatRequest):
    try:
        logger.info(f"Chat POST request received - conversation_id: {request.conversation_id}, message length: {len(request.message)}")
        return EventSourceResponse(
            generate_events(agent, request.message, request.conversation_id)
        )
    except Exception as e:
        logger.error(f"Error processing chat POST: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint for testing
@app.get("/")
async def read_root():
    logger.info("Root endpoint hit")
    return {"message": "Hello, World!"}

# Endpoint to trigger Docker Hello World
@app.get("/ping")
async def ping(name: str = "Anonymous"):
    logger.info(f"Ping endpoint hit with name: {name}")
    
    try:
        # Run the Docker "Hello World" container
        result = subprocess.run(
            ["docker", "run", "hello-world"],
            check=True,  # Ensures an exception is raised if the command fails
            capture_output=True,  # Captures stdout and stderr
            text=True  # Decodes output to string
        )
        
        # Log and return the output from Docker
        docker_output = result.stdout
        logger.info(f"Docker output: {docker_output}")
        return {
            "message": f"Hello, {name}!",
            "docker_output": docker_output,
        }
    except subprocess.CalledProcessError as e:
        # Log and return error output if Docker command fails
        logger.error(f"Docker command failed: {e.stderr}")
        return {
            "message": f"Hello, {name}!",
            "error": "Failed to run Docker container",
            "details": e.stderr,
        }

# Add a download endpoint for GDS files
@app.get("/download/{path:path}")
async def download_file(path: str):
    # Base path where design files are stored
    base_path = DESIGN_BASE_PATH
    file_path = os.path.join(base_path, path)
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Return the file as a download
    return FileResponse(
        path=file_path, 
        filename=os.path.basename(file_path),
        media_type="application/octet-stream"
    )

# Run server when script is executed directly
if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI app on all available IPs, port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
