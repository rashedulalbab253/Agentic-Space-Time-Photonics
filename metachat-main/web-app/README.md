# MetaChat Web App

This MetaChat web app example code demonstrates how to run an AIM design and materials agent on a backend that is interfaced via a simple frontend chat window. Several tools and APIs are included, including FiLM WaveY-Net, and NeuroShaper-enabled optimization of deflectors and single and dual-wavelength metalenses. These examples can be extended to different devices by following these steps:
1) Adding more APIs in `backend/tools/design/superpixel_optimization_gpu_pared.py`
2) Exposing them to the AIM agent in `backend/tools/design/api.py`
3) Adding them to the AIM agent prompt in `backend/agent/cot_iterative_tools_materials.py`

The web app and backend can be run from the same machine. The code is also set up to support running the backend on a separate GPU server and accessing the frontend locally via port forwarding.

## Prerequisites

- A machine with Docker installed, at least one NVIDIA GPU (more is better), and matching NVIDIA CUDA drivers.
- Python 3.10 or higher
- Poetry (Python package manager)
- A modern web browser
- An OpenAI API key

## Installation

1. **Install Poetry**
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
   Verify the installation:
   ```bash
   poetry --version
   ```

2. **Clone the repository**
   ```bash
   git clone https://github.com/jonfanlab/metachat.git
   cd metachat/web-app
   ```

3. **Install dependencies**
   ```bash
   poetry install
   ```

4. **Pull the WaveY-Net Docker image**
   The backend runs GPU design jobs inside a Docker container. Pull the image referenced by `backend/tools/design/api.py`:
   ```bash
   docker pull rclupoiu/waveynet:metachat
   ```
   To pull the exact immutable image digest:
   ```bash
   docker pull rclupoiu/waveynet@sha256:cf2ac6bfc0121a47aaf49f39561ca55a2edfe68d0ccf266e68ee256f313d4c17
   ```

5. **Get an OpenAI API key**
   - Go to [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
   - Sign in to your OpenAI account (or create one if you don't have one)
   - Click "Create new secret key"
   - Give your key a name (e.g., "MetaChat Web App")
   - Copy the API key immediately (you won't be able to see it again)
   - **Important:** Keep your API key secure and never share it publicly

6. **Set up environment variables**
   
   Copy `.env.example` to create your `.env` file:
   ```bash
   cp .env.example .env
   ```
   Then edit the `.env` file and replace the placeholder values with your actual configuration.

   
   **Configuration Details:**
   
   - **OPENAI_API_KEY**: Replace `your-api-key-here` with the API key you copied in step 5.

   - **OPENAI_MODEL_NAME**: The OpenAI chat model/version to use (e.g., `gpt-5.2-2025-12-11`). This is read by both the API server and the materials CLI.

   - **DESIGN_BASE_PATH**: Absolute path to the design workspace (used for Docker working dir, results, and downloads). Default: `/media/tmp2/metachat-app/backend/tools/design`

   - **CHECKPOINT_DIRECTORY_MULTISRC**: Absolute path to the WaveY-Net checkpoint directory (must contain `best_model.pt`, `scaling_factors.yaml`, and `source_code/`). Default: `/media/tmp1/metachat/metachat_code/waveynet`
     - Download and extract the WaveY-Net artifacts from Zenodo:
       ```bash
       mkdir -p /media/tmp1/metachat/metachat_code
       cd /media/tmp1/metachat/metachat_code
       curl -L -o metachat_code_data.zip "https://zenodo.org/records/15802727/files/metachat_code_data.zip?download=1"
       unzip -q metachat_code_data.zip
       ```
       Then set `CHECKPOINT_DIRECTORY_MULTISRC=/media/tmp1/metachat/metachat_code/waveynet`.

   - **MATERIAL_DB_PATH**: Absolute path to the materials database file. Default: `/media/tmp2/metachat-app/backend/tools/material_db/materials.db`
     - You can download the `materials.db` file from [Zenodo](https://zenodo.org/records/15802727).

   - **MEDIA_MOUNT**: Docker volume mount mapping for host media. Default: `/media:/media`

   - **PYTORCH_CUDA_ALLOC_CONF**: PyTorch CUDA allocator settings (useful for tuning memory usage). Default: `max_split_size_mb:49000`. Recommended to set this to the maximum memory of a single GPU on your system. e.g., an RTX 5090 has 32GB of memory, which amounts to 32768MiB. so:
     - Example: `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32768`

   - **BACKEND_BASE_URL**: Address/port where the backend is reachable. Default: `http://localhost:8000`
    
   - **GPU_IDS**: Specifies which GPUs to use for neural design optimization tasks. This is limited by the number of GPUs available on your system. The more GPUs you use, the more parallelization the optimization algorithm can take advantage of, yielding faster optimization results.
     - Format: Comma-separated list of GPU IDs (e.g., `"0,1,2,3"` or `"1,2,3,4,5,6,7"`)
     - If not specified, the application defaults to using GPU 0
     - To use multiple GPUs, specify your GPU IDs, e.g., `GPU_IDS=1,2,3,4,5,6,7`
     - To use a single GPU (e.g., GPU 2): `GPU_IDS=2`

  **Frontend IP Address**: If you are using a separate GPU server and using port forwarding to access the frontend, in `frontend/config.js`, set the IP address of the GPU server (e.g., `http://192.168.1.100:8000`).

## Running the Application

The application consists of two parts that need to be run simultaneously: the backend server and the frontend server.

### Option 1: Running directly in terminal windows

1. **Start the Backend Server**
   Open a terminal and run:
   ```bash
   poetry run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Start the Frontend Server**
   Open another terminal and navigate to the frontend directory:
   ```bash
   cd frontend
   python3 -m http.server 8080
   ```

### Option 2: Running with tmux (recommended)

Using tmux, or a similar session manager that allows sessions to be persisted across network interruptions, is recommended especially if using a separate GPU server.

1. **Start a new tmux session**
   ```bash
   tmux new -s metachat
   ```

2. **Create a window for the backend**
   ```bash
   # Press Ctrl+b then c to create a new window
   poetry run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Create a window for the frontend**
   ```bash
   # Press Ctrl+b then c again to create another window
   cd frontend
   python3 -m http.server 8080
   ```

   To switch between windows:
   - Press `Ctrl+b` then `n` for next window
   - Press `Ctrl+b` then `p` for previous window
   - Press `Ctrl+b` then window number (0, 1, etc.)

### Accessing the Application

1. Open your web browser
2. Navigate to: `http://localhost:8080`

You should see the chat interface where you can interact with the AI.

## Project Structure

The project consists of several key components:

- Frontend: A simple HTML/CSS/JavaScript interface
- Backend: A FastAPI server that handles chat requests

## Troubleshooting

1. **Port Already in Use**
   If you see an error about ports being in use, you can either:
   - Kill the process using the port:
     ```bash
     sudo lsof -i :8000  # For backend port
     sudo lsof -i :8080  # For frontend port
     kill -9 <PID>
     ```
   - Or use different ports:
     ```bash
     # For backend
     poetry run uvicorn main:app --host 0.0.0.0 --port 8001 --reload
     
     # For frontend
     python3 -m http.server 8081
     ```

2. **Poetry Installation Issues**
   If you encounter issues installing Poetry, try:
   ```bash
   pip install poetry
   ```

3. **Dependencies Installation Issues**
   If you encounter issues with Poetry installing dependencies:
   ```bash
   poetry env remove python
   poetry install --no-cache
   ```
