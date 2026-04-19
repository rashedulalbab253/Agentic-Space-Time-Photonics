from typing import Dict, Any, AsyncGenerator, Optional, List
import numpy as np
import base64
import subprocess
import os
import json
import tempfile
from pathlib import Path
import asyncio
import ast
import gdspy
import uuid
from datetime import datetime
# from .superpixel_optimization_gpu_pared import singleWavelengthDeflector, singleWavelengthMetalens, dualWavelengthsMetalens
from .design_database import DesignDatabase

class NeuralDesignAPI:
    def __init__(self, gpu_ids: Optional[List[int]] = None):
        """
        Initialize NeuralDesignAPI.
        
        Args:
            gpu_ids: List of GPU IDs to use. If None, defaults to [0].
        """
        self.gpu_ids = gpu_ids if gpu_ids is not None else [0]
        self.docker_image = "rclupoiu/waveynet:metachat"
        self.media_mount = os.getenv("MEDIA_MOUNT", "/media:/media")
        self.base_path = os.getenv("DESIGN_BASE_PATH", "/media/tmp2/metachat-app/backend/tools/design")
        self.checkpoint_directory_multisrc = os.getenv("CHECKPOINT_DIRECTORY_MULTISRC")
        self.pytorch_cuda_alloc_conf = os.getenv("PYTORCH_CUDA_ALLOC_CONF")
        self.feature_size_meter = 4e-8
        
        # Initialize the design database
        self.db = DesignDatabase(os.path.join(self.base_path, "designs.db"))
        
    async def _run_in_docker(self, script_content: str, results_dir: str, design_type: str = "deflector") -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run optimization script in Docker container and return results
        Args:
            script_content: The Python script to run
            results_dir: Directory for results
            design_type: Type of design ("deflector", "single_wavelength", "dual_wavelength")
        """
        try:
            # Create temporary directory for results
            os.makedirs(results_dir, exist_ok=True)
            
            # Create temporary Python script
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script_content)
                script_path = f.name
                print(f"Script path: {script_path}")
                print(f"Results directory: {results_dir}")

            # Create progress file
            progress_file = f"{results_dir}/progress.txt"
            open(progress_file, 'w').close()  # Create empty file
            
            # Add progress file mount to Docker command
            cmd = [
                "docker", "run",
                "-v", self.media_mount,
                "-v", f"{script_path}:{self.base_path}/run_script.py",
                "-v", f"{results_dir}:/app/results",
                *(
                    ["-e", f"CHECKPOINT_DIRECTORY_MULTISRC={self.checkpoint_directory_multisrc}"]
                    if self.checkpoint_directory_multisrc
                    else []
                ),
                *(
                    ["-e", f"PYTORCH_CUDA_ALLOC_CONF={self.pytorch_cuda_alloc_conf}"]
                    if self.pytorch_cuda_alloc_conf
                    else []
                ),
                "--gpus", "all",
                "--shm-size=100GB",
                "--rm",
                "-w", self.base_path,
                self.docker_image,
                "python", "run_script.py"
            ]
            
            # Start Docker process
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Monitor progress file while Docker is running
            last_progress = ""
            while process.returncode is None:
                try:
                    with open(progress_file, 'r') as f:
                        progress = f.read().strip()
                        if progress and progress != last_progress:
                            yield {"status": progress}
                            last_progress = progress
                except Exception:
                    pass
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
            
            # Read results using the host path based on design type
            plots = {}
            if design_type == "deflector":
                farfield_path = f"{results_dir}/farfield_intensity.png"
                device_path = f"{results_dir}/full_pattern.png"
                
                # Check if files exist before trying to convert them
                if os.path.exists(farfield_path) and os.path.exists(device_path):
                    plots['farfield_plot'] = self._convert_plot_to_base64(farfield_path)
                    plots['device_plot'] = self._convert_plot_to_base64(device_path)
                else:
                    missing_files = []
                    if not os.path.exists(farfield_path):
                        missing_files.append("farfield_intensity.png")
                    if not os.path.exists(device_path):
                        missing_files.append("full_pattern.png")
                    yield {"success": False, "error": f"Docker process failed: output files not generated - missing {', '.join(missing_files)}"}
                    return
            elif design_type == "single_wavelength":
                farfield_path = f"{results_dir}/farfield_intensity_wavelength_1.png"
                device_path = f"{results_dir}/full_pattern.png"
                
                # Check if files exist before trying to convert them
                if os.path.exists(farfield_path) and os.path.exists(device_path):
                    plots['farfield_plot'] = self._convert_plot_to_base64(farfield_path)
                    plots['device_plot'] = self._convert_plot_to_base64(device_path)
                else:
                    missing_files = []
                    if not os.path.exists(farfield_path):
                        missing_files.append("farfield_intensity_wavelength_1.png")
                    if not os.path.exists(device_path):
                        missing_files.append("full_pattern.png")
                    yield {"success": False, "error": f"Docker process failed: output files not generated - missing {', '.join(missing_files)}"}
                    return
            elif design_type == "dual_wavelength":
                farfield_path_1 = f"{results_dir}/farfield_intensity_wavelength_1.png"
                farfield_path_2 = f"{results_dir}/farfield_intensity_wavelength_2.png"
                device_path = f"{results_dir}/full_pattern.png"
                
                # Check if files exist before trying to convert them
                if os.path.exists(farfield_path_1) and os.path.exists(farfield_path_2) and os.path.exists(device_path):
                    plots['farfield_plot_1'] = self._convert_plot_to_base64(farfield_path_1)
                    plots['farfield_plot_2'] = self._convert_plot_to_base64(farfield_path_2)
                    plots['device_plot'] = self._convert_plot_to_base64(device_path)
                else:
                    missing_files = []
                    if not os.path.exists(farfield_path_1):
                        missing_files.append("farfield_intensity_wavelength_1.png")
                    if not os.path.exists(farfield_path_2):
                        missing_files.append("farfield_intensity_wavelength_2.png")
                    if not os.path.exists(device_path):
                        missing_files.append("full_pattern.png")
                    yield {"success": False, "error": f"Docker process failed: output files not generated - missing {', '.join(missing_files)}"}
                    return
            
            yield {
                'success': True,
                'message': 'Design completed successfully. Generated farfield and device pattern plots.',
                'plots': plots
            }

            return
            
        except subprocess.CalledProcessError as e:
            yield {
                'success': False,
                'error': f"Docker execution failed: {str(e)}"
            }

            return
        except Exception as e:
            yield {
                'success': False,
                'error': str(e)
            }

            return
        finally:
            # Cleanup
            if os.path.exists(script_path):
                os.remove(script_path)
            # Optionally clean up results directory after processing
            # if os.path.exists(results_dir):
            #     import shutil
            #     shutil.rmtree(results_dir)

    def _convert_plot_to_base64(self, plot_path: str) -> str:
        """Convert a saved plot to base64 string"""
        with open(plot_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    def _convert_to_gds(self, pattern: np.ndarray, feature_size_meter: float, filename: str) -> str:
        """
        Convert numpy array pattern to GDS file
        
        Args:
            pattern: 2D numpy array (binary or continuous) representing the device
            feature_size_meter: Feature size in meters
            filename: Output filename (without extension)
            
        Returns:
            Path to the created GDS file
        """
        # Create a new GDSII library with nm precision
        lib = gdspy.GdsLibrary(precision=1e-9)
        
        # Create a new cell for the device
        cell = lib.new_cell(f'device_{uuid.uuid4().hex[:8]}')
        
        # Convert the pattern to binary if it's not already (using threshold)
        # Assuming air = 1 and material > 1
        binary_pattern = pattern > 1.5  # Threshold between air and material
        
        # The pattern is typically transposed from the simulation output
        binary_pattern = binary_pattern.T
        
        # Feature size in nm
        feature_size_nm = feature_size_meter * 1e9
        
        # Create polygons from the binary pattern using marching squares or similar algorithm
        polygons = []
        
        # Simple rectangle-based approach
        for i in range(binary_pattern.shape[0]):
            for j in range(binary_pattern.shape[1]):
                if binary_pattern[i, j]:
                    # Create a rectangle at this pixel location
                    rectangle = gdspy.Rectangle(
                        (j * feature_size_nm, i * feature_size_nm),
                        ((j + 1) * feature_size_nm, (i + 1) * feature_size_nm),
                        layer=0
                    )
                    cell.add(rectangle)
        
        # For efficiency with large patterns, we could use a polygon merging approach:
        # merged_polygons = gdspy.boolean(polygons, None, 'or')
        # cell.add(merged_polygons)
        
        # Output path
        gds_path = f"{filename}.gds"
        
        # Write the GDSII file
        lib.write_gds(gds_path)
        
        return gds_path
    
    async def _process_design_results(self, results_dir: str, pattern: np.ndarray) -> Dict[str, Any]:
        """Process design results and create GDS file"""
        # Generate a unique filename with a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        filename = f"device_{timestamp}_{unique_id}"
        gds_filename = os.path.join(results_dir, filename)
        
        gds_path = self._convert_to_gds(pattern, self.feature_size_meter, gds_filename)
        
        # Create a relative path for the download URL
        # Assuming results are stored in a location accessible via /downloads in the web server
        relative_path = os.path.relpath(gds_path, start=self.base_path)
        download_url = f"/download/{relative_path}"
        
        # Return the path to the GDS file along with other results
        return {
            'success': True,
            'gds_file': gds_path,
            'download_url': download_url,
            'message': f"Design complete. GDS file created and ready for download."
        }
    
    async def design_metalens(self,
                            refractive_index: list[float],
                            lens_diameter: float,
                            focal_lengths: list[float],
                            focal_x_offsets: list[float],
                            thickness: float,
                            feature_size_meter: float,
                            operating_wavelengths: list[float],
                            draft: bool = False) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Design a metalens for arbitrary number of wavelengths
        Args:
            refractive_index: List of refractive indices for each wavelength
            lens_diameter: Diameter of the lens in meters
            focal_lengths: List of focal lengths for each wavelength in meters
            focal_x_offsets: List of focal x offsets for each wavelength in meters
            thickness: Thickness of the device in meters
            operating_wavelengths: List of operating wavelengths in meters
            draft: If True, use faster but lower quality optimization (default: False)
        """
        # Type checking
        type_errors = []
        
        if not isinstance(refractive_index, list) or not all(isinstance(n, (int, float)) for n in refractive_index):
            type_errors.append("refractive_index must be a list of floats")
        
        if not isinstance(lens_diameter, (int, float)):
            type_errors.append("lens_diameter must be a float")
            
        if not isinstance(focal_lengths, list) or not all(isinstance(f, (int, float)) for f in focal_lengths):
            type_errors.append("focal_lengths must be a list of floats")
            
        if not isinstance(focal_x_offsets, list) or not all(isinstance(o, (int, float)) for o in focal_x_offsets):
            type_errors.append("focal_x_offsets must be a list of floats")
            
        if not isinstance(thickness, (int, float)):
            type_errors.append("thickness must be a float")

        if not isinstance(feature_size_meter, (int, float)):
            type_errors.append("feature_size_meter must be a float")
            
        if not isinstance(operating_wavelengths, list) or not all(isinstance(w, (int, float)) for w in operating_wavelengths):
            type_errors.append("operating_wavelengths must be a list of floats")
            
        if type_errors:
            yield {
                'success': False,
                'error': "Type errors detected:\n- " + "\n- ".join(type_errors) + 
                         "\n\nPlease fix these errors and try again. For example: " +
                         "design_metalens(refractive_indices=[2.0, 2.5], lens_diameter=100e-6, ...)"
            }
            return
            
        # Check that all parameter lists have the same length
        if not (len(refractive_index) == len(focal_lengths) == 
                len(focal_x_offsets) == len(operating_wavelengths)):
            yield {
                'success': False,
                'error': "All parameter lists (refractive_indices, focal_lengths, focal_x_offsets, operating_wavelengths) must have the same length"
            }
            return
        if not refractive_index:
            yield {
                'success': False,
                'error': "refractive_indices must contain at least one value"
            }
            return

        # Define the maximum supported wavelengths
        max_supported_wavelengths = 2
        
        num_wavelengths = len(operating_wavelengths)
        if num_wavelengths > max_supported_wavelengths:
            yield {
                'success': False,
                'error': f"Currently only supporting a maximum of {max_supported_wavelengths} wavelengths, received {num_wavelengths}"
            }
            return

        # Set batch size based on draft parameter
        batch_size = 3 if draft else 60

        # Create unique results directory
        results_dir = f"{self.base_path}/results_{os.getpid()}"
        
        # Generate appropriate Python script based on number of wavelengths
        if num_wavelengths == 1:
            script_content = f"""
import torch
import gc
import numpy as np
from superpixel_optimization_gpu_pared import singleWavelengthMetalens

gc.collect()
torch.cuda.empty_cache()

full_pattern = singleWavelengthMetalens(
    physical_length_meter={lens_diameter},
    wvl1={operating_wavelengths[0]},
    focal_length_wavelength1_meter={focal_lengths[0]},
    focal_x_offset_wavelength1_meter={focal_x_offsets[0]},
    material_index={refractive_index},
    gpu_ids={self.gpu_ids},
    feature_size_meter={feature_size_meter},
    thickness={thickness},
    ifPlot=True,
    ifDebug=False,
    plotDirectory="/app/results",
    max_retries=3,
    batch_size={batch_size}
)

# Save the pattern as numpy array for GDS conversion
np.save('/app/results/full_pattern.npy', full_pattern)
"""
            design_type = "single_wavelength"
        elif num_wavelengths == 2:
            script_content = f"""
import torch
import gc
import numpy as np
from superpixel_optimization_gpu_pared import dualWavelengthsMetalens

gc.collect()
torch.cuda.empty_cache()

full_pattern = dualWavelengthsMetalens(
    physical_length_meter={lens_diameter},
    wvl1={operating_wavelengths[0]},
    wvl2={operating_wavelengths[1]},
    focal_length_wavelength1_meter={focal_lengths[0]},
    focal_length_wavelength2_meter={focal_lengths[1]},
    focal_x_offset_wavelength1_meter={focal_x_offsets[0]},
    focal_x_offset_wavelength2_meter={focal_x_offsets[1]},
    material_index={refractive_index},
    gpu_ids={self.gpu_ids},
    feature_size_meter={feature_size_meter},
    thickness={thickness},
    ifPlot=True,
    ifDebug=False,
    plotDirectory="/app/results",
    max_retries=3,
    batch_size={batch_size}
)

# Save the pattern as numpy array for GDS conversion
np.save('/app/results/full_pattern.npy', full_pattern)
"""
            design_type = "dual_wavelength"
        
        # Execute the script in Docker
        async for result in self._run_in_docker(script_content, results_dir, design_type=design_type):
            if result.get('success', False):
                # Load the saved pattern and create GDS
                try:
                    pattern_file = os.path.join(results_dir, 'full_pattern.npy')
                    if os.path.exists(pattern_file):
                        pattern = np.load(pattern_file)
                        gds_result = await self._process_design_results(results_dir, pattern)
                        result.update(gds_result)
                        
                        # Save the design to the database
                        associated_files = []
                        # Add plot files to associated_files
                        if design_type == "single_wavelength":
                            if os.path.exists(os.path.join(results_dir, 'farfield_intensity_wavelength_1.png')):
                                associated_files.append({
                                    "file_type": "plot",
                                    "file_path": os.path.join(results_dir, 'farfield_intensity_wavelength_1.png'),
                                    "description": "Farfield intensity plot (wavelength 1)"
                                })
                        elif design_type == "dual_wavelength":
                            if os.path.exists(os.path.join(results_dir, 'farfield_intensity_wavelength_1.png')):
                                associated_files.append({
                                    "file_type": "plot",
                                    "file_path": os.path.join(results_dir, 'farfield_intensity_wavelength_1.png'),
                                    "description": "Farfield intensity plot (wavelength 1)"
                                })
                            if os.path.exists(os.path.join(results_dir, 'farfield_intensity_wavelength_2.png')):
                                associated_files.append({
                                    "file_type": "plot",
                                    "file_path": os.path.join(results_dir, 'farfield_intensity_wavelength_2.png'),
                                    "description": "Farfield intensity plot (wavelength 2)"
                                })
                        
                        if os.path.exists(os.path.join(results_dir, 'full_pattern.png')):
                            associated_files.append({
                                "file_type": "plot",
                                "file_path": os.path.join(results_dir, 'full_pattern.png'),
                                "description": "Device pattern plot"
                            })
                        
                        # Store in database
                        design_id = self.db.save_design(
                            design_type="metalens",
                            parameters={
                                "refractive_indices": refractive_index,
                                "lens_diameter": lens_diameter,
                                "focal_lengths": focal_lengths,
                                "focal_x_offsets": focal_x_offsets,
                                "thickness": thickness,
                                "operating_wavelengths": operating_wavelengths,
                                "num_wavelengths": num_wavelengths
                            },
                            gds_file_path=result.get('gds_file'),
                            success=True,
                            description=f"Metalens with {num_wavelengths} wavelength(s)",
                            associated_files=associated_files
                        )
                        
                        # Add design ID to the result
                        result['design_id'] = design_id
                        
                except Exception as e:
                    result['gds_error'] = str(e)
            yield result

    async def design_deflector(self,
                            refractive_index: list[float],
                            length: float,
                            incident_angles: list[float],
                            deflection_angles: list[float],
                            thickness: float,
                            feature_size_meter: float,
                            operating_wavelength: float,
                            draft: bool = False) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Design a deflector
        Args:
            refractive_index: List of refractive indices
            length: Length of the deflector in meters
            incident_angles: List of incident angles in degrees
            deflection_angles: List of deflection angles in degrees
            thickness: Thickness of the device in meters
            operating_wavelength: Operating wavelength in meters
            draft: If True, use faster but lower quality optimization (default: False)
        """
        # Type checking
        type_errors = []
        
        if not isinstance(refractive_index, list) or not all(isinstance(n, (int, float)) for n in refractive_index):
            type_errors.append("refractive_index must be a list of floats")
        
        if not isinstance(length, (int, float)):
            type_errors.append("length must be a float")
            
        if not isinstance(incident_angles, list) or not all(isinstance(a, (int, float)) for a in incident_angles):
            type_errors.append("incident_angles must be a list of floats")
            
        if not isinstance(deflection_angles, list) or not all(isinstance(a, (int, float)) for a in deflection_angles):
            type_errors.append("deflection_angles must be a list of floats")
            
        if not isinstance(thickness, (int, float)):
            type_errors.append("thickness must be a float")

        if not isinstance(feature_size_meter, (int, float)):
            type_errors.append("feature_size_meter must be a float")
            
        if not isinstance(operating_wavelength, list) or not all(isinstance(w, (int, float)) for w in operating_wavelength):
            type_errors.append("operating_wavelength must be a list of floats")
            
        if type_errors:
            yield {
                'success': False,
                'error': "Type errors detected:\n- " + "\n- ".join(type_errors) + 
                         "\n\nPlease fix these errors and try again. For example: " +
                         "design_deflector(refractive_indices=[2.0], length=100e-6, incident_angles=[0.0], ...)"
            }
            return
            
        # Check that all parameter lists have the same length
        if not (len(refractive_index) == len(incident_angles) == len(deflection_angles)):
            yield {
                'success': False,
                'error': "All parameter lists (refractive_indices, incident_angles, deflection_angles) must have the same length"
            }
            return
        if not refractive_index:
            yield {
                'success': False,
                'error': "refractive_indices must contain at least one value"
            }
            return

        # Set batch size based on draft parameter
        batch_size = 3 if draft else 60

        # Currently only supporting single wavelength deflectors
        # Use the first elements from each list
        incident_angle = incident_angles[0]
        deflection_angle = deflection_angles[0]
        material_index = refractive_index
        
        # Create unique results directory
        results_dir = f"{self.base_path}/results_{os.getpid()}"
        
        # Generate Python script content
        script_content = f"""
import torch
import gc
import numpy as np
from superpixel_optimization_gpu_pared import singleWavelengthDeflector

gc.collect()
torch.cuda.empty_cache()

full_pattern = singleWavelengthDeflector(
    physical_length_meter={length},
    wvl1={operating_wavelength[0]},
    incidence_angle_deg={incident_angle},
    deflection_angle_deg_wavelength1={deflection_angle},
    material_index={material_index},
    gpu_ids={self.gpu_ids},
    feature_size_meter={feature_size_meter},
    thickness={thickness},
    ifPlot=True,
    ifDebug=False,
    plotDirectory="/app/results",
    max_retries=3,
    batch_size={batch_size}
)

# Save the pattern as numpy array for GDS conversion
np.save('/app/results/full_pattern.npy', full_pattern)
"""
        # Execute the script in Docker
        async for result in self._run_in_docker(script_content, results_dir, design_type="deflector"):
            if result.get('success', False):
                # Load the saved pattern and create GDS
                try:
                    pattern_file = os.path.join(results_dir, 'full_pattern.npy')
                    if os.path.exists(pattern_file):
                        pattern = np.load(pattern_file)
                        gds_result = await self._process_design_results(results_dir, pattern)
                        result.update(gds_result)
                        
                        # Save the design to the database
                        associated_files = []
                        
                        if os.path.exists(os.path.join(results_dir, 'farfield_intensity.png')):
                            associated_files.append({
                                "file_type": "plot",
                                "file_path": os.path.join(results_dir, 'farfield_intensity.png'),
                                "description": "Farfield intensity plot"
                            })
                        
                        if os.path.exists(os.path.join(results_dir, 'full_pattern.png')):
                            associated_files.append({
                                "file_type": "plot",
                                "file_path": os.path.join(results_dir, 'full_pattern.png'),
                                "description": "Device pattern plot"
                            })
                        
                        # Store in database
                        design_id = self.db.save_design(
                            design_type="deflector",
                            parameters={
                                "refractive_indices": refractive_indices,
                                "length": length,
                                "incident_angles": incident_angles,
                                "deflection_angles": deflection_angles,
                                "thickness": thickness,
                                "operating_wavelength": operating_wavelength
                            },
                            gds_file_path=result.get('gds_file'),
                            success=True,
                            description=f"Deflector with incident angle {incident_angles[0]}° and deflection angle {deflection_angles[0]}°",
                            associated_files=associated_files
                        )
                        
                        # Add design ID to the result
                        result['design_id'] = design_id
                        
                except Exception as e:
                    result['gds_error'] = str(e)
            yield result

    async def execute(self, code_block: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute the neural design code block and return results
        Args:
            code_block: String containing the function call, e.g.
                'design_metalens(refractive_indices=[2.7, 2.8], lens_diameter=100e-6, ...)'
        """
        try:
            # Parse the function call to extract function name and arguments
            func_name = code_block.split('(')[0].strip()
            args_str = code_block[code_block.find('(')+1:code_block.rfind(')')]

            # Create a fake function call to leverage Python's AST parsing
            fake_call = f"f({args_str})"
            node = ast.parse(fake_call, mode='eval')
            call_node = node.body  # This is an ast.Call object

            # Convert the keyword arguments to a dictionary
            args = {}
            for keyword in call_node.keywords:
                args[keyword.arg] = ast.literal_eval(keyword.value)

            # Call appropriate design function and yield results
            if func_name == 'design_metalens':
                # Set default value for draft if not provided
                draft = args.get('draft', False)
                feature_size_meter = args.get('feature_size_meter', self.feature_size_meter)
                async for result in self.design_metalens(
                    refractive_index=args.get('refractive_index', args.get('refractive_indices')),
                    lens_diameter=args['lens_diameter'],
                    focal_lengths=args['focal_lengths'],
                    focal_x_offsets=args['focal_x_offsets'],
                    thickness=args['thickness'],
                    feature_size_meter=feature_size_meter,
                    operating_wavelengths=args['operating_wavelengths'],
                    draft=draft
                ):
                    yield result
                
            elif func_name == 'design_deflector':
                # Set default value for draft if not provided
                draft = args.get('draft', False)
                feature_size_meter = args.get('feature_size_meter', self.feature_size_meter)
                
                async for result in self.design_deflector(
                    refractive_index=args.get('refractive_index', args.get('refractive_indices')),
                    length=args['length'],
                    incident_angles=args['incident_angles'],
                    deflection_angles=args['deflection_angles'],
                    thickness=args['thickness'],
                    feature_size_meter=feature_size_meter,
                    operating_wavelength=args['operating_wavelength'],
                    draft=draft
                ):
                    yield result
            else:
                raise ValueError(f"Unknown function: {func_name}")
                
        except Exception as e:
            yield {
                'success': False,
                'error': f"Failed to execute neural design: {str(e)}"
            }
