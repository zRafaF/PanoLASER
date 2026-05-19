<div align="center">
<h1>PanoLASER: Training-Free Streaming 3D Reconstruction from Panoramic Video</h1>
</div>

**PanoLASER** adapts the training-free sliding-window architecture of [LASER](https://github.com/neu-vi/LASER) to process 360° equirectangular panoramic video streams. By utilizing [PanoVGGT](https://github.com/YijingGuo-June/PanoVGGT) as the underlying geometry engine, this pipeline addresses monocular scale drift through concentric spherical shell alignment, enabling globally consistent multi-view 3D reconstruction in real-time.

This project uses a locked fork of PanoVGGT as a submodule (commit `1857537`).

## ⚙️ Installation & Setup

We use [`uv`](https://docs.astral.sh/uv/) for deterministic, lightning-fast project and dependency management.

### 1. Clone the Repository
Because this project relies on a submodule, ensure you clone recursively:

```bash
git clone --recurse-submodules https://github.com/zRafaF/PanoLASER
cd PanoLASER
```

### 2. Sync the Environment

Run the following command. `uv` will automatically pull Python 3.11 (the required version for maximum framework stability), create an isolated internal environment, and resolve/install all required dependencies (including the specific PyTorch CUDA 12.4 wheels) exactly as defined in the configuration:

```bash
uv sync
```

**NOTE**
To run in instances that already have cuda torch and some other dependencies you can use the following command.

```bash
uv pip install --system -e .
```

### 3. Compile Cython Modules

PanoLASER relies on compiled C-extensions for fast point cloud registration and graph processing. Compile them using `uv run` to guarantee that the compilation script runs safely inside your locked workspace environment:

```bash
uv run python setup.py build_ext --inplace
```

### 4. Download Pre-trained Weights

Download the PanoVGGT backbone weights into the local `checkpoints` folder:

```bash
mkdir -p checkpoints
wget https://huggingface.co/YijingGuo/PanoVGGT/resolve/main/model.pt -O checkpoints/model.pt
```

## 🚀 Usage

Whenever you execute scripts or run tests in this repository, prepend your command with `uv run` to automatically trigger execution inside the locked dependencies network without manually activating anything:

```bash
# Example for future streaming evaluation execution
uv run python demo.py --window_size 20 --overlap 5
```