from typing import Dict, Any, List, Optional, AsyncGenerator
from .base import Agent
from ..tools.solvers.scientific_compute import ScientificCompute
from ..tools.solvers.symbolic_solver import SymbolicSolver
from ..tools.design.api import NeuralDesignAPI
from ..tools.material_db.query_materials import MaterialDatabaseCLI
from pathlib import Path
import json
from datetime import datetime
import uuid
import os

class StandardAgentTools(Agent):
    """Simple one-shot agent that solves problems in a single model call."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add scientific computing tool
        self.tools['scientific_compute'] = ScientificCompute()
        # Add symbolic solving tool
        self.tools['symbolic_solve'] = SymbolicSolver()
        # Add neural design tools
        # Read GPU IDs from environment variable (comma-separated, e.g., "1,2,3,4,5,6,7")
        gpu_ids_env = os.getenv("GPU_IDS")
        gpu_ids = None
        if gpu_ids_env:
            try:
                gpu_ids = [int(x.strip()) for x in gpu_ids_env.split(",")]
            except ValueError:
                raise ValueError(f"Invalid GPU_IDS format: {gpu_ids_env}. Expected comma-separated integers (e.g., '0,1,2,3')")
        self.tools['neural_design'] = NeuralDesignAPI(gpu_ids=gpu_ids)
        material_db_path = os.getenv("MATERIAL_DB_PATH", "/media/tmp2/metachat-app/backend/tools/material_db/materials.db")
        # Initialize the materials CLI with the same model
        self.materials_cli = MaterialDatabaseCLI(
            db_path=material_db_path,
            model=self.model,  # Pass the model instance directly
            debug=False,
            log_dir="logs/materials_chat"
        )
        
        # Add logging configuration
        self.log_dir = Path(kwargs.get('log_dir', "logs/self_chat")) / self.model.model_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.debug = kwargs.get('debug', False)
        
        if not self.system_prompt:
            self.system_prompt = """You are an expert in optics and photonics with access to scientific computing, symbolic mathematics, neural network-based metalens and deflector design APIs, and a materials database expert you can chat with. 

When solving problems:
0. Think step by step. Break down complex problems into steps and plan your approach before solving.
1. If calculations are needed, write Python numpy code between <tool>scientific_compute</tool> tags
2. If symbolic manipulation is needed to e.g. solve for a variable or rearrange an equation because you are unsure, write Python sympy code between <tool>symbolic_solve</tool> tags
3. If you need to design a metalens or deflector, use these neural network tools:
   - For single wavelength metalenses, use: <tool>neural_design
   design_single_wavelength_metalens(refractive_index, lens_diameter [m], focal_length [m], focal_x_offset [m], thickness [m], operating_wavelength [m])
   </tool>
   - For dual wavelength metalenses, use: <tool>neural_design
   design_dual_wavelength_metalens(refractive_index, lens_diameter [m], focal_length_1 [m], focal_length_2 [m], focal_x_offset_1 [m], focal_x_offset_2 [m], thickness [m], operating_wavelength_1 [m], operating_wavelength_2 [m])
   </tool>
   - For deflectors, use: <tool>neural_design
   design_deflector(refractive_index, length [m], incident_angle [deg], deflection_angle [deg], thickness [m], operating_wavelength [m])
   </tool>
   Note that the focal_x_offset is the x-offset of the focal point from the center of the lens.
4. If you need information about materials or their properties, you can chat with the materials expert:
   <tool>materials_chat
   Your question or message to the materials expert
   </tool>
5. If no calculations are needed, simply state the answer directly
6. You can only use ONE type of tag per message
7. Make sure to convert intermediate results to the correct units before using them to prevent multiplication or function unit mismatch errors
8. After using a tool, analyze its output before proceeding. If there is an error, think carefully why it occured and fix the code to try again.

Example responses:

For materials chat:
<tool>materials_chat
What materials would you recommend for a high-power laser mirror operating at 1064 nm?
</tool>

<tool>materials_chat
What is the refractive index of TiO2 film at 800 nm wavelength?
</tool>

For neural network design:
<tool>neural_design
design_metalens(refractive_index=2.7, lens_diameter=100e-6, focal_length=200e-6, thickness=500e-9, operating_wavelength=800e-9)
</tool>

For a numerical calculation:
<tool>scientific_compute
import numpy as np
from scipy import constants

wavelength = 500e-9  # 500 nm
freq = constants.c / wavelength
print(f'Answer: {freq:.2e} Hz')
</tool>

For symbolic manipulation:
<tool>symbolic_solve
import sympy as sp

# Solve n1 * sin(theta1) = n2 * sin(theta2) for theta2
n1, n2, theta1, theta2 = sp.symbols('n1 n2 theta1 theta2')
eq = sp.Eq(n1 * sp.sin(theta1), n2 * sp.sin(theta2))
solution = sp.solve(eq, theta2)[0]
print(f'theta2 = {solution}')
</tool>

For a direct answer:
Answer: 1.55 μm"""

    def _format_messages(self, problem: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
        """Format messages for the model including system prompt and conversation history."""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)
            
        # Add current problem
        messages.append({"role": "user", "content": problem})
        
        return messages
    
    async def _log_conversation(self, problem_id: str, messages: List[Dict[str, str]]):
        """Log conversation to a JSON file."""
        log_file = self.log_dir / f"{problem_id}.json"
        
        # Load existing log if it exists
        existing_log = []
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    existing_log = json.load(f)
            except json.JSONDecodeError:
                existing_log = []
                
        # Convert any SymPy objects to strings before serializing
        def convert_sympy_objects(obj):
            if isinstance(obj, dict):
                return {k: convert_sympy_objects(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_sympy_objects(item) for item in obj]
            elif str(type(obj).__module__).startswith('sympy'):
                return str(obj)
            return obj
        
        # Convert SymPy objects in messages
        messages = convert_sympy_objects(messages)
        
        # Append new messages
        existing_log.extend(messages)
        
        # Write updated log
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(existing_log, f, indent=2, default=str, ensure_ascii=False)
        except Exception as e:
            print(f"Error writing to log file: {e}")

    async def solve(self, problem: str, problem_id: Optional[str] = None, temperature: float = 1.0, disable_cache: bool = False) -> Dict[str, Any]:
        # Generate problem_id if not provided
        if not problem_id:
            problem_id = str(uuid.uuid4())
            
        # Log initial problem
        await self._log_conversation(problem_id, [{
            "role": "user",
            "content": problem,
            "timestamp": datetime.now().isoformat()
        }])

        if disable_cache:
            timestamp = datetime.now().isoformat()
            problem = f"Date submitted: {timestamp}\n\n{problem}"

        messages = self._format_messages(problem)
        iteration_count = 0
        max_iterations = 20
        conversation = []

        while True:  # Let it run until natural completion or max iterations
            iteration_count += 1
            
            # Check if max iterations exceeded
            if iteration_count > max_iterations:
                # Log error if max iterations reached
                await self._log_conversation(problem_id, [{
                    "role": "error",
                    "content": "Maximum iterations reached",
                    "timestamp": datetime.now().isoformat(),
                    "iteration": iteration_count
                }])

                yield {
                    "status": "error",
                    "error": "Maximum iterations reached",
                    "conversation": conversation,
                    "problem_id": problem_id
                }

                return
            
            if self.debug:
                print(f"\n=== Iteration {iteration_count} ===")

            current_response = await self._call_model(messages, temperature=temperature)

            # Log the model's response
            await self._log_conversation(problem_id, [{
                "role": "assistant",
                "content": current_response,
                "timestamp": datetime.now().isoformat(),
                "iteration": iteration_count
            }])

            # Process tool calls and thoughts
            processed_solution = current_response
            tool_used = False

            # Handle materials chat with logging
            if "<tool>materials_chat" in processed_solution:
                tool_used = True
                start = processed_solution.find("<tool>materials_chat")
                end = processed_solution.find("</tool>", start)
                if end != -1:
                    message = processed_solution[start + len("<tool>materials_chat"):end].strip()
                    
                    # Log materials chat query
                    await self._log_conversation(problem_id, [{
                        "role": "tool_call",
                        "tool": "materials_chat",
                        "query": message,
                        "timestamp": datetime.now().isoformat(),
                        "iteration": iteration_count
                    }])
                    
                    results = await self.materials_cli.query_with_id(message, conversation_id=problem_id)
                    
                    # Log materials chat response
                    await self._log_conversation(problem_id, [{
                        "role": "tool_response",
                        "tool": "materials_chat",
                        "content": results['message'],
                        "timestamp": datetime.now().isoformat(),
                        "iteration": iteration_count
                    }])
                    
                    messages.append({"role": "assistant", "content": current_response})
                    messages.append({"role": "user", "content": f"Materials expert response: {results['message']}"})

            # Similar logging for other tools
            for tool_name in ['scientific_compute', 'symbolic_solve', 'neural_design']:
                if f"<tool>{tool_name}" in processed_solution:
                    tool_used = True
                    start = processed_solution.find(f"<tool>{tool_name}")
                    end = processed_solution.find("</tool>", start)
                    if end != -1:
                        code_block = processed_solution[start + len(f"<tool>{tool_name}"):end].strip()
                        
                        # Log tool call
                        await self._log_conversation(problem_id, [{
                            "role": "tool_call",
                            "tool": tool_name,
                            "code": code_block,
                            "timestamp": datetime.now().isoformat(),
                            "iteration": iteration_count
                        }])
                        
                        result = await self.tools[tool_name].execute(code_block)
                        
                        # Separate the plots from the message for the LLM
                        plots = None
                        if isinstance(result, dict):
                            plots = result.pop('plots', None)
                            # Log the result without plots
                            await self._log_conversation(problem_id, [{
                                "role": "tool_response",
                                "tool": tool_name,
                                "result": result,
                                "timestamp": datetime.now().isoformat(),
                                "iteration": iteration_count
                            }])
                        
                        # Send plots to frontend if available
                        if plots:
                            yield {
                                "type": "plots",
                                "data": plots
                            }
                        
                        # Only send the success/error message to the LLM
                        messages.append({"role": "assistant", "content": current_response})
                        messages.append({"role": "user", "content": f"Tool output: {result}"})

            # If no tool was used, treat as direct response
            if not tool_used:
                yield {
                    "solution": processed_solution,
                    "metadata": {
                        "method": "iterative",
                        "num_iterations": iteration_count,
                        "conversation": conversation,
                        "problem_id": problem_id
                    },
                    "tool_calls": self.tool_calls
                }
                return

            # Update conversation history
            conversation.append({
                "iteration": iteration_count,
                "input": current_response,
                "output": processed_solution,
                "timestamp": datetime.now().isoformat()
            })

    async def solve_with_status(
        self, 
        problem: str, 
        problem_id: Optional[str] = None, 
        temperature: float = 1.0,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        # Generate problem_id if not provided
        if not problem_id:
            problem_id = str(uuid.uuid4())
        
        # Initialize messages with system prompt and conversation history
        messages = self._format_messages(problem, conversation_history)
        
        # Log initial problem
        await self._log_conversation(problem_id, [{
            "role": "user",
            "content": problem,
            "timestamp": datetime.now().isoformat()
        }])
        
        iteration_count = 0
        max_iterations = 20
        conversation = []

        while True:  # Let it run until natural completion or max iterations
            iteration_count += 1
            
            # Check if max iterations exceeded
            if iteration_count > max_iterations:
                # Log error if max iterations reached
                await self._log_conversation(problem_id, [{
                    "role": "error",
                    "content": "Maximum iterations reached",
                    "timestamp": datetime.now().isoformat(),
                    "iteration": iteration_count
                }])
                
                yield {"error": "Maximum iterations reached"}
                return

            if self.debug:
                print(f"\n=== Iteration {iteration_count} ===")

            yield {"status": "Thinking..."}
            current_response = await self._call_model(messages, temperature=temperature)

            # Log the model's response
            await self._log_conversation(problem_id, [{
                "role": "assistant",
                "content": current_response,
                "timestamp": datetime.now().isoformat(),
                "iteration": iteration_count
            }])

            # Handle final response to user
            if "<response>" in current_response:
                response = current_response.split("<response>")[1].split("</response>")[0].strip()
                
                # Log final response
                await self._log_conversation(problem_id, [{
                    "role": "response",
                    "content": response,
                    "timestamp": datetime.now().isoformat(),
                    "iteration": iteration_count
                }])
                
                yield {"solution": response, "metadata": {
                    "method": "iterative",
                    "num_iterations": iteration_count,
                    "conversation": conversation,
                    "problem_id": problem_id
                }}
                return

            # Process tool calls and thoughts
            processed_solution = current_response
            
            # If no tags present, treat as thinking/planning
            if not any(tag in processed_solution for tag in ["<tool>", "<response>"]):
                # Log thinking
                await self._log_conversation(problem_id, [{
                    "role": "thinking",
                    "content": processed_solution,
                    "timestamp": datetime.now().isoformat(),
                    "iteration": iteration_count
                }])
                
                messages.append({"role": "assistant", "content": processed_solution})
                messages.append({"role": "user", "content": "Continue with your approach. If you need to reply to the user with an answer or need clarification, respond with <response> tags."})
                continue

            # Handle materials chat with logging
            if "<tool>materials_chat" in processed_solution:
                start = processed_solution.find("<tool>materials_chat")
                end = processed_solution.find("</tool>", start)
                if end != -1:
                    message = processed_solution[start + len("<tool>materials_chat"):end].strip()
                    
                    yield {"status": "Consulting materials expert..."}
                    
                    # Log materials chat query
                    await self._log_conversation(problem_id, [{
                        "role": "tool_call",
                        "tool": "materials_chat",
                        "query": message,
                        "timestamp": datetime.now().isoformat(),
                        "iteration": iteration_count
                    }])
                    
                    results = await self.materials_cli.query_with_id(message, conversation_id=problem_id)
                    
                    # Log materials chat response
                    await self._log_conversation(problem_id, [{
                        "role": "tool_response",
                        "tool": "materials_chat",
                        "content": results['message'],
                        "timestamp": datetime.now().isoformat(),
                        "iteration": iteration_count
                    }])
                    
                    messages.append({"role": "assistant", "content": current_response})
                    messages.append({"role": "user", "content": f"Materials expert response: {results['message']}"})

            # Handle other tools with logging
            for tool_name in ['scientific_compute', 'symbolic_solve', 'neural_design']:
                if f"<tool>{tool_name}" in processed_solution:
                    start = processed_solution.find(f"<tool>{tool_name}")
                    end = processed_solution.find("</tool>", start)
                    if end != -1:
                        code_block = processed_solution[start + len(f"<tool>{tool_name}"):end].strip()
                        
                        yield {"status": f"Using {tool_name.replace('_', ' ')}..."}
                        
                        # Log tool call
                        await self._log_conversation(problem_id, [{
                            "role": "tool_call",
                            "tool": tool_name,
                            "code": code_block,
                            "timestamp": datetime.now().isoformat(),
                            "iteration": iteration_count
                        }])

                        async for result in self.tools[tool_name].execute(code_block):
                            if isinstance(result, dict) and "status" in result:
                                yield {"status": result["status"]}
                            else:
                                # Separate the plots from the message for the LLM
                                plots = None
                                if isinstance(result, dict):
                                    plots = result.pop('plots', None)
                                    # Log the result without plots
                                    await self._log_conversation(problem_id, [{
                                        "role": "tool_response",
                                        "tool": tool_name,
                                        "result": result,
                                        "timestamp": datetime.now().isoformat(),
                                        "iteration": iteration_count
                                    }])
                                
                                # Send plots to frontend if available
                                if plots:
                                    yield {
                                        "type": "plots",
                                        "data": plots
                                    }
                                
                                # Only send the success/error message to the LLM
                                messages.append({"role": "assistant", "content": current_response})
                                messages.append({"role": "user", "content": f"Tool output: {result}"})

                    # Update conversation history
                    conversation.append({
                        "iteration": iteration_count,
                        "input": current_response,
                        "output": processed_solution,
                        "timestamp": datetime.now().isoformat()
                    })
