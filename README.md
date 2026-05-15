<div align="center">
<h1>PanoLASER: Training-Free Streaming 3D Reconstruction from Panoramic Video</h1>
</div>

**PanoLASER** adapts the training-free sliding-window architecture of [LASER](https://github.com/neu-vi/LASER) to process 360° equirectangular panoramic video streams. By utilizing [PanoVGGT](https://github.com/YijingGuo-June/PanoVGGT) as the underlying geometry engine, this pipeline addresses monocular scale drift through concentric spherical shell alignment, enabling globally consistent multi-view 3D reconstruction in real-time.

This project uses a locked fork of PanoVGGT as a submodule (commit `1857537`).

## ⚙️ Installation & Setup

We recommend using Python 3.11 and [`uv`](https://docs.astral.sh/uv/) for lightning-fast dependency management.

### 1. Clone the Repository
Because this project relies on a submodule, ensure you clone recursively:
```bash
git clone --recurse-submodules https://github.com/zRafaF/PanoLASER
cd PanoLASER
```

### 2. Create the Environment

Use `uv` to pull Python 3.10 (if you don't have it) and create a standard virtual environment:

```bash
uv venv --python 3.10 venv
source venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

### 3. Install Dependencies

Install all requirements using `uv`. We pass the PyTorch index URL here to ensure we grab the correct CUDA 12.4 wheels. *(Note: We strictly use `numpy==1.26.4` to maintain compatibility with Cython/Numba extensions).*

```bash
uv pip install -r requirements.txt --extra-index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)
```

### 4. Compile Cython Modules

PanoLASER relies on compiled C-extensions for fast point cloud registration and graph processing. Compile them using:

```bash
python setup.py build_ext --inplace
```

### 5. Download Pre-trained Weights

Download the PanoVGGT backbone weights into the `checkpoints` directory:

```bash
mkdir -p checkpoints
wget [https://huggingface.co/YijingGuo/PanoVGGT/resolve/main/model.pt](https://huggingface.co/YijingGuo/PanoVGGT/resolve/main/model.pt) -O checkpoints/model.pt
```

## Usage

*(Usage instructions for `demo.py` will be added here as the streaming pipeline is finalized.)*
