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

class IterativeAgentToolsMaterials(Agent):
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

        # Backend base URL for download links (e.g., http://localhost:8000)
        self.download_base_url = os.getenv("BACKEND_BASE_URL", "http://localhost:8000").rstrip("/")
        
        if not self.system_prompt:
            self.system_prompt = """You are a conversational expert in optics and photonics engaging in a continuous conversation to help users.
You communicate directly with users using <chat> tags for most interactions, and only use internal thinking for complex problem-solving steps.
You have iterative access to numpy, sympy, neural network-based design APIs, and a materials database expert. 
You can talk to yourself and have an internal monologue if you are in problem-solving mode by not using <chat> tags. Plan out tool use to use information gathered from the tools at subsequent iterations.

Two key modes:
1. CONVERSATION MODE (default, wrap your response in <chat> tags): Respond directly to simple questions, greetings, clarifications, or follow-ups using <chat> tags.
2. PROBLEM-SOLVING MODE (use only when absolutely necessary to solve a complex technical problem, not for conversation): For complex technical questions, use internal thinking followed by appropriate tools.

If you find yourself caught in a loop, the only way you can solve this problem is by breaking free using tags: whether that's <neural_design>, <chat>, or whatever is most appropriate. Never tell the user you were caught in a loop.
IMPORTANT: Never tell the user you will do something and just leave it at that. Instead, use non-chat tags to complete the action. If you know you have to e.g., design something with parameters you know, use the <tool>neural_design</tool> tag directly.

Remember: if you need the user to see your response, wrap it in <chat> tags.

Guidelines:
0. Default to CONVERSATION MODE unless a complex technical problem is presented. Don't overthink simple exchanges.
1. Think step by step. Break down complex problems into steps and plan your approach before solving.
2. If you need to ask the user for more information or missing details/parameters, use <chat> tags.
3. If you have to do any sort of calculation, make sure to write Python numpy code between <tool>scientific_compute</tool> tags so you can be certain of the answer.
4. The ONLY way to design a metalens or deflector is to use these neural network tools:
   IMPORTANT: Before beginning, check each parameter needed for the API call. If any info is missing, determine how you will figure it out. If you need user input, ask the user using <chat> up-front before continuing with anything else. DO NOT MAKE ASSUMPTIONS. ASK USING <chat>.
   If a value is not provided, your next response must ask the user for it using <chat> tags. Do not make assumptions based on common values.
   Beyond asking to narrow down materials where human designer input is absolutely necessary, do not ask the user for material properties: use the materials expert to figure these out.
   For prototyping, ask the user if they want to use a fast but more poor quality draft design.
   The substrate and immersion media are fixed. You do not have control over these and do not ask the user for them.
   - For metalenses, use: <tool>neural_design design_metalens(refractive_indices [list], lens_diameter [m], focal_lengths [list, m], focal_x_offsets [list, m], thickness [m], operating_wavelengths [list, m], min_feature_size [m, optional], draft=False [bool, optional])</tool>
   Note: The focal_x_offset is the x-offset of the focal point from the center of the lens. All list parameters must have the same length.
   - For deflectors, use: <tool>neural_design design_deflector(refractive_indices [list], length [m], incident_angles [list, deg], deflection_angles [list, deg], thickness [m], operating_wavelength [m], min_feature_size [m, optional], draft=False [bool, optional])</tool>
   Remember to use the <tool> tags. If you neglect this you won't actually design anything and you will confuse the user.
   DO NOT make up function inputs beyond the ones specified and make sure that neural design tool calls are written on a single line without newline characters. DO NOT predict specific efficiencies, performance etc. if you make a tool call: you will see the result in a subsequent query and will be able to make more informed conclusions.
5. If you need information about materials or their properties, you can chat with the materials expert:
   <tool>materials_chat
   Your question or message to the materials expert
   </tool>
   Remember, if the user asks a materials question that you asked the materials expert, pass the answer back to the user using <chat> tags after you analyze the answer.
6. Don't ever tell the user something like <chat>I will work on it</chat>, <chat>Let's initiate the design process</chat>, or <chat>I will consult the expert...</chat> because you have to make the call to the API or tool. Otherwise you will confuse the user. Instead, make the call to the API or tool: <tool>neural_design</tool>, <tool>materials_chat</tool>, etc.
7. Return the final answer wrapped in <chat> tags
8. You can only use ONE type of tag per message. e.g., Do not include both <chat> and <tool>. Decide which one needs to be executed on this step and save the other tags for future conversation steps.
9. After using a tool, analyze its output before proceeding. If there is an error, think carefully why it occured and fix the code to try again.
10. Do not bother the user with specifics about how to use the APIs (e.g., requirements regarding lists and lengths of lists for API calls), because the user does not know these details of how you call things in the backend.

Examples:

1. Simple question - respond directly:
   <chat>
   A metalens shapes the wavefront of incident light to focus it at a specific point. Did you have a specific application in mind?
   </chat>

2. Chat with materials expert:
User query: I want to use TiO2 at 650nm. What refractive index does that correspond to?

Initial response: Let's find out the refractive index of TiO2 at 650 nm. I'll consult the materials expert for accurate data.\n\n<tool>materials_chat\nCould you provide the refractive index of TiO2 at a wavelength of 650 nm?\n</tool>"

Tool response: The estimated refractive index of TiO2 at a wavelength of 650 nm (0.65 µm) is approximately 2.5774.

NOTE THE USE OF <chat> TAGS RIGHT AWAY TO RESPOND TO THE USER:
Final response to user: <chat>Great! The refractive index of TiO2 at 650 nm is approximately 2.5774. \n\nWith this information, we can proceed with the design of your deflector. Could you please provide the remaining details for the design:\n1. The length of the deflector.\n2. The incident angles of the light in degrees.\n3. The deflection angles in degrees.\n4. The thickness of the deflector.\n\nOnce I have these details, I'll be able to help you further!</chat>


3. Calculation problem - think then use tools:
   I need to calculate the focal length based on these parameters.
   <tool>scientific_compute
   import numpy as np
   lens_diameter = 100e-6
   focal_length = 80e-6
   NA = lens_diameter / (2 * focal_length)
   print(f'NA: {NA}')
   </tool>

4. For neural network design (return the text you receive wrapped in <tool>neural_design </tool> tags):
<tool>neural_design
design_metalens(refractive_indices=[2.7], lens_diameter=[100e-6], focal_lengths=[200e-6], focal_x_offsets=[0], thickness=[500e-9], operating_wavelengths=[800e-9])
</tool>

5. Fix errors:
The error indicates that the solution was returned in a different format than expected. I have to first access the dictonary in the list.

6. Provide final answer:
<chat>
Based on my analysis, I recommend using a TiO2 metalens with these parameters:
- Diameter: 200 μm
- Focal length: 500 μm
- Thickness: 600 nm

Would you like me to help you design a metalens with these parameters?
</chat>

CONVERSATION GUIDELINES:
- Use <chat> tags for ALL user-facing responses, clarification, and missing details/parameters
- Maintain an engaging, natural conversational style
- Only switch to detailed step-by-step thinking for complex technical problems--but remember to switch back to conversation mode when done thinking!
- Remember that without <chat> tags, the user cannot see your response

IMPORTANT: Any text not wrapped in tags will be treated as your internal thoughts and planning. Only text within <chat> tags will be shown to the user.
Make sure your response makes sense to the user based on their last message.

The available tools are:
- Scientific computing: <tool>scientific_compute</tool>
- Symbolic solving: <tool>symbolic_solve</tool>
- Neural design: <tool>neural_design</tool>
- Materials expert chat: <tool>materials_chat</tool>
"""

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

            # Handle final response to user
            if "<chat>" in current_response:
                response = current_response.split("<chat>")[1].split("</chat>")[0].strip()
                
                # Log final response
                await self._log_conversation(problem_id, [{
                    "role": "response",
                    "content": response,
                    "timestamp": datetime.now().isoformat(),
                    "iteration": iteration_count
                }])

                yield {
                    "solution": response,
                    "metadata": {
                        "method": "iterative",
                        "num_iterations": iteration_count,
                        "conversation": conversation,
                        "problem_id": problem_id
                    },
                    "tool_calls": self.tool_calls
                }

                return

            # Process tool calls and thoughts
            processed_solution = current_response
            
            # If no tags present, treat as thinking/planning
            if not any(tag in processed_solution for tag in ["<tool>", "<chat>"]):
                # Log thinking
                await self._log_conversation(problem_id, [{
                    "role": "thinking",
                    "content": processed_solution,
                    "timestamp": datetime.now().isoformat(),
                    "iteration": iteration_count
                }])
                
                messages.append({"role": "assistant", "content": processed_solution})
                messages.append({"role": "user", "content": "Continue with your approach. If you need to reply to the user with an answer or need clarification, respond with <chat> tags."})

                continue

            # Handle materials chat with logging
            if "<tool>materials_chat" in processed_solution:
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

            # Update conversation history
            conversation.append({
                "iteration": iteration_count,
                "input": current_response,
                "output": processed_solution,
                "timestamp": datetime.now().isoformat()
            })
            
            # messages.append({"role": "assistant", "content": current_response})
            # messages.append({"role": "user", "content": f"Tool output: {processed_solution}"})

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

        # Track farfield images to include in next message to LLM
        farfield_plots = []
        
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
            
            # If we have farfield plots, create a temporary copy of messages with the image
            current_messages = messages.copy()
            
            # If we have farfield images to analyze, add them to the temporary message list
            if farfield_plots:
                # prompt for image analysis
                image_message = {
                    "role": "user", 
                    "content": f"""This message is from the optimization assistant; your response will be to the user.
Here are the plots from the design. 

If the design is of a deflector:
Critically analyze the radial far field intensity plot. For assessing the angle, note that you do not have the capability to accurately precisely determine the deflection angle, so if it looks approximately correct, do not fixate on this (don't tell the user that you are not over-fixating on this detail though, just don't do it).
Pay careful attention to unintended scattering and lobe thickness.

If the design is of a focusing metalens:
Critically analyze these farfield plots and provide feedback on the focusing quality, efficiency, and any artifacts or issues you observe.
Pay careful attention to the z axis numbers to make sure that the focal lengths (if applicable) are where they are expected to be.
Note that focus requires light to actually be concentrated around the desired point, instead of being merely present through scattering in the vicinity of where the focal point should be. This looks like a bright spot in that region. If you do not see a bright spot, there is no focus.
Note that negative deflection angles should be to the left, and positive deflection angles should be to the right.
If the design is poor:
-If it makes sense for their parameters, you can suggest to increase the diameter of the device and/or any other parameters that you think might help.
If it is a final design:
-Be very careful and critical but fair in your comments to get the best possible design.
-If the design is a draft and of decent quality (i.e., close enough to the desired functionality), offer to optimize again using more computational resources for the best possible result.
Respond to the user using <chat> tags and make sure to include the GDS file in the response as a link.
For download links, use the explicit backend base URL, e.g., [GDS File]({self.download_base_url}/download/results_2131726/device_20251221_121717_66d40807.gds and ensure the displayed link is human readable (e.g., "GDS File"). Do not print out the Deisgn ID, as the user doesn't need this.
""",
                    "images": farfield_plots
                }
                current_messages.append(image_message)
                yield {"status": "Analyzing results..."}
                
                # Use the temporary message list with images for this call only
                current_response = await self._call_model(current_messages, temperature=temperature)
                
                # Clear the plots after using them
                farfield_plots = []
            else:
                # Normal call without images
                current_response = await self._call_model(messages, temperature=temperature)

            # Log the model's response
            await self._log_conversation(problem_id, [{
                "role": "assistant",
                "content": current_response,
                "timestamp": datetime.now().isoformat(),
                "iteration": iteration_count
            }])

            # Handle final response to user
            if "<chat>" in current_response:
                response = current_response.split("<chat>")[1].split("</chat>")[0].strip()
                
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
            if not any(tag in processed_solution for tag in ["<tool>", "<chat>"]):
                # Log thinking
                await self._log_conversation(problem_id, [{
                    "role": "thinking",
                    "content": processed_solution,
                    "timestamp": datetime.now().isoformat(),
                    "iteration": iteration_count
                }])
                
                # Emit internal thought as status and add to messages
                yield {"thinking": processed_solution}
                messages.append({"role": "assistant", "content": processed_solution, "type": "internal"})
                messages.append({"role": "user", "content": "Continue with your approach. If you need to reply to the user with an answer or need clarification, respond with <chat> tags."})
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
                                    
                                    # Store only farfield plots for next message to LLM
                                    if tool_name == 'neural_design' and plots:
                                        # Extract only the farfield plots
                                        if 'farfield_plot' in plots:
                                            farfield_plots.append(plots['farfield_plot'])
                                        if 'farfield_plot_1' in plots:
                                            farfield_plots.append(plots['farfield_plot_1'])
                                        if 'farfield_plot_2' in plots:
                                            farfield_plots.append(plots['farfield_plot_2'])
                                
                                # Add the assistant's response and tool result to permanent message history
                                # (without the images)
                                messages.append({"role": "assistant", "content": current_response})
                                messages.append({"role": "user", "content": f"Tool output: {result}"})

                    # Update conversation history
                    conversation.append({
                        "iteration": iteration_count,
                        "input": current_response,
                        "output": processed_solution,
                        "timestamp": datetime.now().isoformat()
                    })
