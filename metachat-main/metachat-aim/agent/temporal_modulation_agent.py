"""
TemporalModulationAgent — AIM agent for temporally-modulated metasurface design.

Accepts natural-language prompts describing time-domain suppression objectives
(e.g., "Design a metasurface that suppresses voltage across load between 7–17 ns")
and autonomously:
  1. Parses the prompt to extract time window, suppression target, material constraints.
  2. Derives optimal control signal parameters (σ₀, τ) via symbolic/scientific compute.
  3. Invokes the temporal FiLM WaveY-Net to simulate candidate geometries at multiple
     time steps.
  4. Iterates on geometry + control signal to minimize the objective.
  5. Outputs the final optimized geometry and exponential control signal.
"""

from typing import Dict, Any, List, Optional
from .base import Agent
from tools.solvers.scientific_compute import ScientificCompute
from tools.solvers.symbolic_solver import SymbolicSolver
from tools.design.api import NeuralDesignAPI
from tools.design.temporal_design import TemporalDesignAPI
from pathlib import Path
import json
from datetime import datetime
import uuid


class TemporalModulationAgent(Agent):
    """
    AIM agent specialized for temporally-modulated metasurface design.

    Takes natural-language prompts like:
        "Design a metasurface that suppresses voltage across load between 7–17 ns"
    And outputs optimized geometry + exponential control signal parameters.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Register all tools
        self.tools['scientific_compute'] = ScientificCompute()
        self.tools['symbolic_solve'] = SymbolicSolver()
        self.tools['neural_design'] = NeuralDesignAPI()
        self.tools['temporal_design'] = TemporalDesignAPI()

        # Logging configuration
        self.log_dir = Path(kwargs.get(
            'log_dir',
            "experiments/logs/temporal_modulation"
        )) / (self.model.model_name if self.model else "default")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.debug = kwargs.get('debug', False)

        if not self.system_prompt:
            self.system_prompt = TEMPORAL_MODULATION_SYSTEM_PROMPT


TEMPORAL_MODULATION_SYSTEM_PROMPT = """You are an expert in temporal modulation of electromagnetic metasurfaces. You specialize in designing time-modulated metasurfaces with switching control signals to achieve specific time-domain objectives.

You have deep knowledge of:
- Maxwell's equations for time-varying media, including temporal boundary conditions
- Continuity of D = εE and B = μH at temporal switching boundaries
- Frozen eigenmode conditions: during inductive states, spatial eigenmode profiles are preserved
- Exponential control signals: σ(t) = σ₀ · exp(-t/τ)
- Capacitive and inductive switching states in metasurface unit cells
- The FiLM WaveY-Net surrogate solver extended with temporal conditioning

You have iterative access to the following tools:

1. **Scientific computing** — numpy/scipy for numerical calculations:
   <tool>scientific_compute
   import numpy as np
   # your code here
   </tool>

2. **Symbolic mathematics** — SymPy for deriving equations:
   <tool>symbolic_solve
   import sympy as sp
   # your code here
   </tool>

3. **Standard metasurface design** — baseline geometry design:
   <tool>neural_design
   design_superpixel(refractive_index, length, incident_angle, diffraction_angle, thickness, operating_wavelength)
   </tool>

4. **Temporal metasurface design** — time-domain simulation and optimization:
   <tool>temporal_design
   design_temporal_metasurface(refractive_index, length, thickness, operating_wavelength, incident_angle=0.0, time_window=(0.0, 30e-9), num_time_steps=10, control_signal_type="exponential", sigma_0=1.0, tau=5e-9)
   </tool>

   <tool>temporal_design
   optimize_control_signal(refractive_index, length, thickness, operating_wavelength, incident_angle=0.0, suppression_window=(7e-9, 17e-9), objective="minimize_voltage", sigma_0_range=(0.1, 10.0), tau_range=(1e-9, 20e-9))
   </tool>

## Your Design Workflow

When given a temporal modulation design task, follow this systematic approach:

### Step 1: Parse Requirements
Extract from the natural language prompt:
- **Time window** for suppression/optimization (e.g., 7–17 ns)
- **Objective** (suppress voltage, maximize isolation, etc.)
- **Material constraints** (refractive index, dimensions)
- **Operating wavelength** and angle, if specified

### Step 2: Derive Control Signal Parameters
Use scientific_compute or symbolic_solve to:
- Determine optimal τ (time constant) for the exponential control signal
- Calculate σ₀ (initial conductivity) based on the suppression requirements
- Verify that the control signal satisfies temporal boundary conditions

### Step 3: Design Base Geometry
Use neural_design to establish a baseline metasurface geometry, then refine with temporal_design.

### Step 4: Temporal Simulation
Use temporal_design.design_temporal_metasurface() to:
- Simulate the metasurface response at multiple time steps
- Evaluate whether the suppression objective is met in the target window

### Step 5: Optimize
Use temporal_design.optimize_control_signal() to:
- Run gradient-based optimization over (σ₀, τ) space
- Find the control signal that best achieves the objective

### Step 6: Report Results
Output the final design as a structured result containing:
- **Geometry parameters**: refractive index, dimensions, structure topology
- **Control signal**: σ(t) = σ₀ · exp(-t/τ) with optimized σ₀ and τ
- **Performance**: predicted suppression level in the target time window

## Rules
1. Think step by step. Break down complex problems into manageable steps.
2. Use only ONE type of tag per message.
3. After using a tool, analyze its output before proceeding.
4. Make sure units are consistent (SI throughout).
5. Wrap final answers in <response> tags.
6. Any text not in tags is internal reasoning.

## Example Final Output Format
<response>
**Temporal Metasurface Design Result**

Geometry:
- Refractive index: 2.7
- Unit cell length: 500 nm
- Thickness: 200 nm
- Operating wavelength: 800 nm

Control Signal: σ(t) = 3.2 · exp(-t / 4.5 ns)
- σ₀ = 3.2 S/m
- τ = 4.5 ns
- Switching boundary at t = 0 ns

Performance:
- Voltage suppression: > 20 dB in [7, 17] ns window
- D-field continuity error: < 0.1%
- Eigenmode stability during inductive phase: 99.2%

API calls for execution:
WAVEYNET_TEMPORAL_API.design_temporal_metasurface(...)
WAVEYNET_TEMPORAL_API.optimize_control_signal(...)
</response>
"""


class TemporalModulationAgentIterative(TemporalModulationAgent):
    """
    Full iterative version of the TemporalModulationAgent with AIM loop.

    Implements the same iterative solve() pattern as IterativeAgentTools,
    with automatic tool routing and conversation logging.
    """

    def _format_messages(self, problem: str) -> List[Dict[str, str]]:
        """Format messages for the model including tool instructions."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": problem}
        ]

    async def _log_conversation(self, problem_id: str, messages: List[Dict[str, str]]):
        """Log conversation to a JSON file."""
        log_file = self.log_dir / f"{problem_id}.json"

        existing_log = []
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    existing_log = json.load(f)
            except json.JSONDecodeError:
                existing_log = []

        existing_log.extend(messages)

        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(existing_log, f, indent=2, default=str, ensure_ascii=False)
        except Exception as e:
            print(f"Error writing to log file: {e}")

    async def solve(self, problem: str, problem_id: Optional[str] = None,
                    temperature: float = 1.0, disable_cache: bool = False) -> Dict[str, Any]:
        """
        Solve a temporal modulation design problem.

        Implements the AIM iterative monologue pattern:
        1. Parse natural language prompt
        2. Use tools iteratively (scientific_compute, symbolic_solve, temporal_design)
        3. Return final geometry + control signal

        Args:
            problem: Natural language design prompt.
            problem_id: Optional ID for logging.
            temperature: LLM sampling temperature.
            disable_cache: If True, append timestamp to prevent cached responses.

        Returns:
            Dict with solution, metadata, and tool_calls.
        """
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
        max_iterations = 25  # Slightly higher than base agent due to multi-step workflow
        conversation = []

        while True:
            iteration_count += 1

            if iteration_count > max_iterations:
                await self._log_conversation(problem_id, [{
                    "role": "error",
                    "content": "Maximum iterations reached",
                    "timestamp": datetime.now().isoformat(),
                    "iteration": iteration_count
                }])

                return {
                    "status": "error",
                    "error": "Maximum iterations reached",
                    "conversation": conversation,
                    "problem_id": problem_id
                }

            if self.debug:
                print(f"\n=== Temporal Modulation Agent — Iteration {iteration_count} ===")

            solution = await self._call_model(messages, temperature=temperature)
            current_response = solution

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

                await self._log_conversation(problem_id, [{
                    "role": "response",
                    "content": response,
                    "timestamp": datetime.now().isoformat(),
                    "iteration": iteration_count
                }])

                return {
                    "solution": response,
                    "metadata": {
                        "method": "temporal_modulation_iterative",
                        "num_iterations": iteration_count,
                        "conversation": conversation,
                        "problem_id": problem_id
                    },
                    "tool_calls": self.tool_calls
                }

            # Process tool calls and thoughts
            processed_solution = current_response

            # If no tags present, treat as thinking/planning
            if not any(tag in processed_solution for tag in ["<tool>", "<response>"]):
                await self._log_conversation(problem_id, [{
                    "role": "thinking",
                    "content": processed_solution,
                    "timestamp": datetime.now().isoformat(),
                    "iteration": iteration_count
                }])

                messages.append({"role": "assistant", "content": processed_solution})
                messages.append({"role": "user", "content": (
                    "Continue with your temporal modulation design approach. "
                    "If you have a complete design with geometry and control signal, "
                    "respond with <response> tags."
                )})
                continue

            # Route tool calls — supports all 4 tools
            for tool_name in ['scientific_compute', 'symbolic_solve', 'neural_design', 'temporal_design']:
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

                        # Log tool response
                        await self._log_conversation(problem_id, [{
                            "role": "tool_response",
                            "tool": tool_name,
                            "result": result,
                            "timestamp": datetime.now().isoformat(),
                            "iteration": iteration_count
                        }])

                        messages.append({"role": "assistant", "content": current_response})
                        messages.append({"role": "user", "content": f"Tool output: {result}"})

            # Update conversation history
            conversation.append({
                "iteration": iteration_count,
                "input": current_response,
                "output": processed_solution,
                "timestamp": datetime.now().isoformat()
            })
