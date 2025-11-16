# 4D Bose-Einstein Condensate Simulator with ParaView Export

Complete 4D BEC simulation on hypersphere with VTU/VRT export for ParaView 6.0.1 visualization.

## Features

- ✅ **4D Hypersphere BEC Simulation** - Rotating superfluid on S³
- ✅ **Vortex Detection** - Topological winding number verification
- ✅ **VTU Export** - Native ParaView format with time series
- ✅ **Standalone** - No external dependencies (besides standard libraries)
- ✅ **GPU Accelerated** - CuPy for fast computation
- ✅ **Reproducible** - PKL initial state files with fixed seeds

## Quick Start

### 1. Run Test Export (No GPU Required)

```bash
python test_vtu_standalone.py
```

This creates synthetic 4D data and exports to `test_paraview_output/test_series.pvd`.

### 2. Run Full Simulation (Requires CUDA GPU)

```bash
python sim_vrt_standalone.py
```

**Outputs:**
- `initial_state_N128_seed42.pkl` - Reusable initial conditions
- `paraview_output/snapshot_*.vtu` - Individual VTU files
- `paraview_output/snapshot_series.pvd` - **Open this in ParaView!**
- `paraview_output/README_ParaView.txt` - Detailed usage guide

### 3. Load Previous Initial State

```bash
python sim_vrt_standalone.py --load initial_state_N128_seed42.pkl
```

Skips expensive initialization (shell scanning, neighbor tree construction).

## Files

| File | Description |
|------|-------------|
| `sim_vrt_standalone.py` | **Recommended** - Complete standalone simulation + VTU export |
| `sim_v006_vrt.py` | Alternative version (depends on sim_v005.py) |
| `test_vtu_standalone.py` | Test script with synthetic data |
| `README_VTU_EXPORT.md` | Comprehensive ParaView documentation |

## ParaView Workflow

1. **Open ParaView 6.0.1**
2. **File → Open** → `paraview_output/snapshot_series.pvd`
3. Click **Apply**
4. Change representation to **Point Gaussian**
5. Color by **density**
6. Press ▶️ to animate

### Advanced Visualizations

- **Isolate vortices**: Threshold filter on `vortex_marker > 0.5`
- **Density isosurfaces**: Contour filter on `density`
- **Velocity vectors**: Glyph filter with `velocity_4D`
- **4D axis swap**: Calculator filter to swap w ↔ z coordinates

See `README_VTU_EXPORT.md` for complete documentation.

## Data Fields

Each VTU file contains:

### Coordinates
- **x, y, z**: 3D spatial position
- **w_coordinate**: 4th dimension (scalar field)

### Scalar Fields
- **density**: Condensate density |ψ|²
- **phase**: Phase angle [0, 1]
- **vortex_marker**: 1 = vortex core, 0 = superfluid
- **radial_distance**: Distance from origin
- **velocity_3D_magnitude**: Speed in x,y,z subspace

### Vector Fields
- **velocity_4D**: Full 4D velocity (v_w, v_x, v_y, v_z)

## Requirements

```
numpy
cupy-cuda11x  # or cupy-cuda12x for CUDA 12
scipy
```

Optional:
```
msgpack  # For compact binary export (not needed for VTU)
```

## Performance

Typical performance on NVIDIA GPU:

| Grid Size | Points | Init Time | Steps/sec | Memory |
|-----------|--------|-----------|-----------|--------|
| N=64 | ~50K | 10s | 20 | 2 GB |
| N=128 | ~500K | 60s | 5 | 8 GB |
| N=192 | ~1.5M | 300s | 1 | 16 GB |

Adjust `downsample` parameter to reduce file size:
- `downsample=1` → Full resolution (largest files)
- `downsample=4` → Recommended for exploration
- `downsample=10` → Preview quality (smallest files)

## Physics

### Simulation Model
- **Geometry**: 4D hypersphere S³ (3-sphere embedded in R⁴)
- **System**: Rotating Bose-Einstein condensate
- **Equation**: Gross-Pitaevskii equation with rotation
- **Vortices**: Quantized circulation (topological defects)
- **Excitations**: Phonons (P₀) and rotons (R₀)

### Key Parameters
- **R**: Hypersphere radius (healing lengths)
- **δ**: Shell thickness
- **g**: Interaction strength (controls sound speed)
- **Ω**: Rotation rate (drives vortex nucleation)

## Examples

### Vortex Lattice Formation

```python
params = SimulationParams(
    R=1000.0,
    omega=0.05,  # Faster rotation → more vortices
    N=128,
    random_seed=42
)
```

### Ground State (No Vortices)

```python
params = SimulationParams(
    R=1000.0,
    omega=0.0,  # No rotation
    initial_condition_type="imaginary_time"
)
```

## Troubleshooting

### ParaView shows empty scene
- Check "Apply" was clicked
- Change representation to "Points" or "Point Gaussian"
- Verify visibility (eye icon)

### Points appear tiny
- Change to **Point Gaussian** representation
- Increase Gaussian Radius (5-20)
- Or use **Glyph** filter with sphere glyphs

### File size too large
- Increase `downsample` in export (4-10 recommended)
- Reduce `save_every` in simulation
- Use fewer snapshots

### Out of GPU memory
- Reduce grid size `N` (128 → 96 or 64)
- Close other GPU applications
- Monitor with `nvidia-smi`

## Citation

If you use this code in research, please cite:

```bibtex
@software{bec4d_vrt,
  title = {4D BEC Simulator with ParaView Export},
  author = {[Your Name]},
  year = {2025},
  url = {https://github.com/yourusername/BEC}
}
```

## License

MIT License - See LICENSE file for details

## References

- Gross-Pitaevskii Equation: [Pitaevskii & Stringari, 2003]
- VTK File Formats: https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf
- ParaView Guide: https://www.paraview.org/paraview-guide/

---

**Questions?** Open an issue or consult the detailed documentation in `README_VTU_EXPORT.md`.
