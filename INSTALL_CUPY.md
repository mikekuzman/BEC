# CuPy Installation Guide for 4D BEC Simulator

## Quick Reference

**For CUDA 11.x:** `pip install cupy-cuda11x`
**For CUDA 12.x:** `pip install cupy-cuda12x`

---

## Step-by-Step Installation

### 1. Check Your CUDA Version

Run this command on your machine with the NVIDIA GPU:

```bash
nvidia-smi
```

Look for the line that says **"CUDA Version: X.X"** in the top-right corner.

**Example output:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0   |
+-----------------------------------------------------------------------------+
```

### 2. Install Matching CuPy Package

Based on your CUDA version, install the corresponding CuPy:

| Your CUDA Version | Install Command |
|-------------------|-----------------|
| **CUDA 12.x** (12.0, 12.1, 12.2, etc.) | `pip install cupy-cuda12x` |
| **CUDA 11.x** (11.0-11.8) | `pip install cupy-cuda11x` |
| **CUDA 10.2** | `pip install cupy-cuda102` |

**Recommended (most common):**
```bash
# For CUDA 12.x (newest)
pip install cupy-cuda12x

# OR for CUDA 11.x (common)
pip install cupy-cuda11x
```

### 3. Install Other Dependencies

```bash
pip install numpy scipy
```

### 4. Verify Installation

```bash
python -c "import cupy as cp; print(f'CuPy version: {cp.__version__}'); print(f'CUDA available: {cp.cuda.is_available()}')"
```

**Expected output:**
```
CuPy version: 12.3.0
CUDA available: True
```

---

## Complete Installation (Fresh Environment)

### Option A: Using pip

```bash
# Create new environment
python -m venv bec_env
source bec_env/bin/activate  # On Windows: bec_env\Scripts\activate

# Install for CUDA 12.x
pip install cupy-cuda12x numpy scipy

# OR install for CUDA 11.x
pip install cupy-cuda11x numpy scipy
```

### Option B: Using conda

```bash
# Create new environment with conda
conda create -n bec_env python=3.10
conda activate bec_env

# Install CuPy via conda (automatically detects CUDA)
conda install -c conda-forge cupy

# Install other dependencies
conda install numpy scipy
```

---

## Troubleshooting

### Issue: "ImportError: libcuda.so.1: cannot open shared object file"

**Cause:** CUDA runtime not found in library path

**Fix:**
```bash
# Add CUDA to library path (adjust version as needed)
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH

# Make permanent by adding to ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```

### Issue: "CuPy version mismatch with CUDA"

**Cause:** Installed wrong CuPy package for your CUDA version

**Fix:**
```bash
# Uninstall wrong version
pip uninstall cupy cupy-cuda11x cupy-cuda12x

# Check your CUDA version
nvidia-smi

# Install correct version
pip install cupy-cuda12x  # or cupy-cuda11x
```

### Issue: "RuntimeError: CUDA environment is not correctly set up"

**Cause:** CUDA toolkit not installed or not in PATH

**Fix:**
1. Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads
2. Add to PATH:
   ```bash
   export PATH=/usr/local/cuda-12.0/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH
   ```

### Issue: "No CUDA-capable device detected"

**Cause:**
- No NVIDIA GPU in system
- GPU driver not installed
- GPU not recognized

**Fix:**
1. Check GPU is detected: `lspci | grep -i nvidia`
2. Install NVIDIA drivers: `sudo ubuntu-drivers autoinstall` (Ubuntu)
3. Reboot system

---

## Version Compatibility Matrix

| CUDA Version | CuPy Package | Python Versions | Status |
|--------------|--------------|-----------------|--------|
| 12.x | cupy-cuda12x | 3.8-3.12 | ✅ Recommended |
| 11.x | cupy-cuda11x | 3.8-3.12 | ✅ Stable |
| 11.8 | cupy-cuda118 | 3.8-3.11 | ✅ Stable |
| 11.7 | cupy-cuda117 | 3.8-3.11 | ⚠️ Older |
| 11.2 | cupy-cuda112 | 3.7-3.10 | ⚠️ Older |
| 10.2 | cupy-cuda102 | 3.7-3.10 | ⚠️ Legacy |

---

## Specific Version Installation

If you need a specific CuPy version:

```bash
# List available versions
pip install cupy-cuda12x==

# Install specific version (example)
pip install cupy-cuda12x==12.3.0
```

---

## Testing Your Installation

### Quick Test

```bash
python -c "import cupy as cp; a = cp.array([1, 2, 3]); print('CuPy working:', a)"
```

### Full Test (from repo)

```bash
python test_vtu_standalone.py
```

### GPU Performance Test

```python
import cupy as cp
import time

# Allocate large array on GPU
n = 10000
a = cp.random.rand(n, n)
b = cp.random.rand(n, n)

# Benchmark matrix multiplication
start = time.time()
c = cp.dot(a, b)
cp.cuda.Stream.null.synchronize()
elapsed = time.time() - start

print(f"GPU Matrix multiply ({n}x{n}): {elapsed:.3f}s")
print(f"Performance: {2*n**3/elapsed/1e9:.1f} GFLOPS")
```

**Expected:** >100 GFLOPS on modern GPU

---

## Alternative: CPU-Only Version (No GPU)

If you don't have an NVIDIA GPU, you can modify the code to use NumPy instead of CuPy:

```bash
# Install only NumPy
pip install numpy scipy

# The simulation will be MUCH slower, but will work
```

**Note:** For CPU-only, reduce grid size significantly:
```python
params = SimulationParams(
    N=32,  # Instead of 128
    # ... other params
)
```

---

## Recommended Setup (Production)

```bash
# 1. Create clean environment
conda create -n bec_prod python=3.10
conda activate bec_prod

# 2. Install CuPy (auto-detects CUDA)
conda install -c conda-forge cupy

# 3. Install other dependencies
conda install numpy scipy

# 4. Verify
python -c "import cupy as cp; print('GPU available:', cp.cuda.is_available())"

# 5. Clone repo and run
git clone <repo_url>
cd BEC
python sim_vrt_standalone.py
```

---

## Getting Help

1. **Check CUDA version:** `nvidia-smi`
2. **Check CuPy installation:** `python -c "import cupy; print(cupy.__version__)"`
3. **Check GPU visibility:** `python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"`

**CuPy Documentation:** https://docs.cupy.dev/en/stable/install.html

**CUDA Download:** https://developer.nvidia.com/cuda-downloads

---

## Summary

**Most users should run:**

```bash
# Check CUDA version
nvidia-smi

# For CUDA 12.x
pip install cupy-cuda12x numpy scipy

# For CUDA 11.x
pip install cupy-cuda11x numpy scipy

# Verify
python -c "import cupy as cp; print('CuPy OK:', cp.cuda.is_available())"
```

Then you're ready to run the simulation! 🚀
