"""
Temporal Design API tool for AIM agents.

Provides neural-network-based temporal metasurface design capabilities:
- design_temporal_metasurface: Invoke the temporal FiLM WaveY-Net to predict
  fields at multiple time steps for a given geometry and control signal.
- optimize_control_signal: Gradient-based optimization of exponential control
  signal parameters to achieve a target suppression objective.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class TemporalDesignAPI:
    """Tool for designing temporally-modulated metasurfaces via FiLM WaveY-Net."""

    def design_temporal_metasurface(
        self,
        refractive_index: float,
        length: float,
        thickness: float,
        operating_wavelength: float,
        incident_angle: float = 0.0,
        time_window: Tuple[float, float] = (0.0, 30e-9),
        num_time_steps: int = 10,
        control_signal_type: str = "exponential",
        sigma_0: float = 1.0,
        tau: float = 5e-9,
    ) -> Dict[str, Any]:
        """
        Design a temporally-modulated metasurface and predict fields across time.

        Uses the temporal FiLM WaveY-Net surrogate solver to predict electromagnetic
        fields at multiple time steps, allowing evaluation of the metasurface
        response under a time-varying control signal.

        Args:
            refractive_index: Material refractive index of the metasurface.
            length: Length/period of the metasurface unit cell (meters).
            thickness: Thickness of the nanostructure (meters).
            operating_wavelength: Design wavelength (meters).
            incident_angle: Angle of incidence (degrees).
            time_window: Tuple (t_start, t_end) in seconds for the simulation.
            num_time_steps: Number of time steps to evaluate within the window.
            control_signal_type: Type of control signal ("exponential", "step", "linear").
            sigma_0: Initial conductivity/amplitude of the control signal.
            tau: Time constant for exponential decay (seconds).

        Returns:
            Dict containing the design parameters and API call string.
        """
        # Validate required parameters
        params = {
            'refractive_index': refractive_index,
            'length': length,
            'thickness': thickness,
            'operating_wavelength': operating_wavelength,
        }

        missing = [name for name, value in params.items() if value is None]
        if missing:
            return f"Error: Missing required parameters: {', '.join(missing)}"

        non_numeric = [name for name, value in params.items()
                       if not isinstance(value, (int, float)) or isinstance(value, bool)]
        if non_numeric:
            return f"Error: Non-numeric parameters detected: {', '.join(non_numeric)}"

        if control_signal_type not in ("exponential", "step", "linear"):
            return f"Error: Unsupported control_signal_type '{control_signal_type}'. Use 'exponential', 'step', or 'linear'."

        # Generate time steps
        t_start, t_end = time_window
        time_steps = np.linspace(t_start, t_end, num_time_steps).tolist()

        # Generate control signal values at each time step
        if control_signal_type == "exponential":
            control_values = [float(sigma_0 * np.exp(-(t - t_start) / tau)) for t in time_steps]
        elif control_signal_type == "step":
            t_mid = (t_start + t_end) / 2
            control_values = [float(sigma_0) if t < t_mid else 0.0 for t in time_steps]
        elif control_signal_type == "linear":
            control_values = [float(sigma_0 * (1.0 - (t - t_start) / (t_end - t_start))) for t in time_steps]

        api_call = (
            f"WAVEYNET_TEMPORAL_API.design_temporal_metasurface("
            f"refractive_index={refractive_index}, "
            f"length={length}, "
            f"thickness={thickness}, "
            f"operating_wavelength={operating_wavelength}, "
            f"incident_angle={incident_angle}, "
            f"time_window=({t_start}, {t_end}), "
            f"num_time_steps={num_time_steps}, "
            f"control_signal_type='{control_signal_type}', "
            f"sigma_0={sigma_0}, "
            f"tau={tau})"
        )

        return (
            f"API called successfully. The temporal FiLM WaveY-Net will evaluate "
            f"the metasurface at {num_time_steps} time steps from {t_start*1e9:.1f} ns "
            f"to {t_end*1e9:.1f} ns with {control_signal_type} control signal "
            f"(sigma_0={sigma_0}, tau={tau*1e9:.1f} ns).\n\n"
            f"Control signal values: {[f'{v:.4f}' for v in control_values]}\n\n"
            f"Return this API string to the user: {api_call}"
        )

    def optimize_control_signal(
        self,
        refractive_index: float,
        length: float,
        thickness: float,
        operating_wavelength: float,
        incident_angle: float = 0.0,
        suppression_window: Tuple[float, float] = (7e-9, 17e-9),
        objective: str = "minimize_voltage",
        sigma_0_range: Tuple[float, float] = (0.1, 10.0),
        tau_range: Tuple[float, float] = (1e-9, 20e-9),
    ) -> Dict[str, Any]:
        """
        Optimize the exponential control signal parameters to achieve a target objective
        (e.g., minimize voltage across load) within a specified time window.

        This uses the temporal FiLM WaveY-Net as a differentiable forward model
        and performs gradient-based optimization over the control signal parameters.

        Args:
            refractive_index: Material refractive index.
            length: Metasurface unit cell length (meters).
            thickness: Structure thickness (meters).
            operating_wavelength: Design wavelength (meters).
            incident_angle: Angle of incidence (degrees).
            suppression_window: Tuple (t_start, t_end) in seconds — the time window
                               where the objective should be optimized.
            objective: Optimization objective. One of:
                      "minimize_voltage" — suppress voltage across load.
                      "maximize_isolation" — maximize field isolation.
            sigma_0_range: Search range for initial conductivity (S/m).
            tau_range: Search range for time constant (seconds).

        Returns:
            Dict containing optimized parameters and API call string.
        """
        params = {
            'refractive_index': refractive_index,
            'length': length,
            'thickness': thickness,
            'operating_wavelength': operating_wavelength,
        }

        missing = [name for name, value in params.items() if value is None]
        if missing:
            return f"Error: Missing required parameters: {', '.join(missing)}"

        t_start, t_end = suppression_window

        api_call = (
            f"WAVEYNET_TEMPORAL_API.optimize_control_signal("
            f"refractive_index={refractive_index}, "
            f"length={length}, "
            f"thickness={thickness}, "
            f"operating_wavelength={operating_wavelength}, "
            f"incident_angle={incident_angle}, "
            f"suppression_window=({t_start}, {t_end}), "
            f"objective='{objective}', "
            f"sigma_0_range=({sigma_0_range[0]}, {sigma_0_range[1]}), "
            f"tau_range=({tau_range[0]}, {tau_range[1]}))"
        )

        return (
            f"API called successfully. Gradient-based optimization will run over "
            f"the temporal FiLM WaveY-Net to find optimal (sigma_0, tau) that "
            f"{objective.replace('_', ' ')} in the window "
            f"[{t_start*1e9:.1f} ns, {t_end*1e9:.1f} ns].\n\n"
            f"Search ranges: sigma_0 ∈ [{sigma_0_range[0]}, {sigma_0_range[1]}] S/m, "
            f"tau ∈ [{tau_range[0]*1e9:.1f}, {tau_range[1]*1e9:.1f}] ns\n\n"
            f"Return this API string to the user: {api_call}"
        )

    async def execute(self, code_block: str) -> Dict[str, Any]:
        """
        Execute the temporal design code block and route to the appropriate function.

        Args:
            code_block: String containing the Python function call to execute.

        Returns:
            Dict containing success status and result/error.
        """
        try:
            local_ns = {
                'self': self,
                'design_temporal_metasurface': self.design_temporal_metasurface,
                'optimize_control_signal': self.optimize_control_signal,
            }

            modified_code = f"_result = {code_block}"
            exec(modified_code, globals(), local_ns)
            result = local_ns.get('_result')

            return {
                'success': True,
                'result': result
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
