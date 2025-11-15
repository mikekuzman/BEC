# VTU Export for ParaView Visualization

## Overview

The `sim_v006_vrt.py` module adds **VTK Unstructured Grid (.vtu)** export functionality to the 4D Bose-Einstein Condensate simulator, making it compatible with **ParaView 6.0.1**.

### Key Features

- ✅ **VTU format**: Native ParaView support for time series animation
- ✅ **PVD collection files**: Single file to load entire simulation sequence
- ✅ **4D coordinate support**: Full 4D hypersphere data (w, x, y, z)
- ✅ **Multiple fields**: Density, phase, velocity, vortex markers
- ✅ **Preserves initial state**: PKL files still generated for reproducibility
- ✅ **Downsampling support**: Adjustable resolution vs file size trade-off

---

## Files

| File | Description |
|------|-------------|
| `sim_v006_vrt.py` | Main simulation with VTU export |
| `sim_v005.py` | Base simulation (unchanged, required import) |
| `test_vtu_export.py` | Test script with synthetic data |

---

## Quick Start

### 1. Run Test Export (No GPU Required)

```bash
python test_vtu_export.py
```

This creates synthetic 4D data and exports to `test_paraview_output/test_series.pvd`.

**Open in ParaView:**
- File → Open → `test_paraview_output/test_series.pvd`
- Click **Apply**
- Change representation to **Point Gaussian**
- Color by **density**

### 2. Run Full Simulation (Requires CUDA GPU)

```bash
python sim_v006_vrt.py
```

**Outputs:**
- `initial_state_N128_seed42.pkl` - Initial conditions (reusable)
- `paraview_output/snapshot_*.vtu` - Individual snapshots
- `paraview_output/snapshot_series.pvd` - **Open this in ParaView!**
- `paraview_output/README_ParaView.txt` - Detailed usage guide

### 3. Load Previous Initial State

```bash
python sim_v006_vrt.py --load initial_state_N128_seed42.pkl
```

Skips expensive initial state calculation (shell scanning, neighbor tree).

---

## Data Structure

### VTU File Contents

Each `.vtu` file contains:

#### Point Coordinates (3D + 1 scalar)
- **x, y, z**: Spatial coordinates (3D projection of 4D data)
- **w_coordinate**: 4th dimension (stored as scalar field)

*Rationale:* ParaView natively handles 3D. The w-coordinate can be swapped using Calculator filter.

#### Scalar Fields
| Field | Description | Range |
|-------|-------------|-------|
| `density` | Condensate density \|ψ\|² | [0, ~1] |
| `phase` | Phase angle arg(ψ) | [0, 1] (normalized) |
| `vortex_marker` | Vortex core indicator | 0 or 1 |
| `radial_distance` | Distance from origin | [R-δ/2, R+δ/2] |
| `velocity_3D_magnitude` | Speed in x,y,z subspace | [0, ~0.5] |

#### Vector Fields
| Field | Components | Description |
|-------|------------|-------------|
| `velocity_4D` | (v_w, v_x, v_y, v_z) | Full 4D velocity field |

---

## ParaView Workflow

### Basic Visualization

1. **Load data:**
   ```
   File → Open → paraview_output/snapshot_series.pvd
   Properties → Apply
   ```

2. **Adjust representation:**
   - Representation: **Point Gaussian**
   - Gaussian Radius: ~5.0
   - Opacity: ~0.3 (for dense clouds)

3. **Color mapping:**
   - Color by: **density**
   - Edit color map: Use "Viridis" or "Plasma"
   - Rescale to data range

### Animation

1. **Play timeline:**
   - Click ▶️ Play button
   - Adjust speed in Animation View

2. **Save animation:**
   ```
   File → Save Animation
   Format: AVI, MP4, or PNG sequence
   ```

### Advanced Techniques

#### Isolate Vortex Cores

```
Filters → Threshold
Scalars: vortex_marker
Minimum: 0.5
Apply
```

Color by **phase** to see circulation direction.

#### Density Isosurfaces

```
Filters → Contour
Contour By: density
Isosurfaces: 0.1, 0.5, 0.9
Apply
```

#### Velocity Glyphs

```
Filters → Glyph
Glyph Type: Arrow
Vectors: velocity_4D (first 3 components shown)
Scale Mode: vector
Apply
```

#### 4D Axis Swapping (w ↔ z)

```
Filters → Calculator
Result Array Name: coords_wxyz
Expression: w_coordinate*kHat + coordsX*iHat + coordsY*jHat
Apply → Color by new field
```

This visualizes the w-x-y subspace instead of x-y-z.

---

## Performance Tuning

### File Size vs Resolution

Adjust downsampling in export:

```python
export_snapshots_to_vtu_series(
    snapshots,
    params,
    downsample=4,  # ← Adjust this: 1=full, 10=1/10th points
    output_dir='paraview_output'
)
```

**Example:** N=128, 500k points/snapshot, 10 snapshots

| Downsample | Points/snap | File size | Total size |
|------------|-------------|-----------|------------|
| 1 | 500,000 | 50 MB | 500 MB |
| 2 | 250,000 | 25 MB | 250 MB |
| 5 | 100,000 | 10 MB | 100 MB |
| 10 | 50,000 | 5 MB | 50 MB |

**Recommendation:** Start with `downsample=4` for exploration, then reduce to 1-2 for final publication figures.

---

## Data Fields Explained

### Density Field
- Physical meaning: Condensate density \|ψ(r)\|²
- Vortex cores: density → 0
- Typical range: [0.0, 1.0] after normalization

### Phase Field
- Physical meaning: arg(ψ) ∈ [-π, π], normalized to [0, 1]
- Color mapping: Use "HSV" colormap for periodic boundary
- Vortex detection: Phase singularity (discontinuous phase)

### Vortex Marker
- Algorithm: Topological winding number detection
- Value: 1 = verified vortex core, 0 = normal fluid
- Clustering: Multiple points per vortex line

### Velocity Field (4D)
- Physical meaning: ∇φ where φ = arg(ψ)
- Units: healing length / time unit
- Interpretation: Superfluid flow velocity

---

## Comparison: VTU vs JSON/MessagePack

| Feature | VTU (.vtu + .pvd) | JSON | MessagePack |
|---------|-------------------|------|-------------|
| **ParaView support** | ✅ Native | ❌ Requires converter | ❌ Requires converter |
| **Time series** | ✅ PVD collection | Manual | Manual |
| **File size** | ~Medium | Large | Small |
| **Human readable** | ✅ XML | ✅ Yes | ❌ Binary |
| **Web browser viz** | ❌ No | ✅ Yes | ✅ Yes |
| **Python analysis** | Via PyVista | ✅ Direct | ✅ Direct |

**Recommendation:** Use VTU for ParaView visualization, keep JSON/MessagePack for web-based or custom Python analysis.

---

## Troubleshooting

### ParaView shows empty scene
- Check "Apply" was clicked after loading
- Verify representation is not "Outline" (change to "Points" or "Point Gaussian")
- Check eye icon visibility is enabled

### Points appear as tiny dots
- Change representation to **Point Gaussian**
- Increase Gaussian Radius (5-20)
- Or use **Glyph** filter with sphere glyphs

### Animation doesn't play
- Verify PVD file was loaded (not individual .vtu)
- Check Animation View → time range matches data
- Click ▶️ Play button (not just step buttons)

### File size too large
- Increase `downsample` parameter in export
- Reduce number of snapshots (`save_every` in simulation)
- Use ParaView's "Extract Subset" filter

### "w_coordinate" field not visible
- Check Point Data arrays in Information panel
- Verify you loaded .pvd file (not just .vtu)
- Try reloading file

---

## Integration with Existing Code

The new VTU export **supplements** existing formats:

```python
# Run simulation (unchanged)
sim = HypersphereBEC(params)
sim.save_initial_state()  # ← Still creates .pkl file
snapshots = sim.run(n_steps=5000, save_every=500)

# Export to multiple formats
export_snapshots_to_vtu_series(snapshots, sim.p, ...)      # ParaView
export_snapshots_to_json(snapshots, sim.p, ...)            # Web viz
export_snapshots_to_msgpack(snapshots, sim.p, ...)         # Compact
```

**All formats share the same snapshot data** - no conflicts or redundancy.

---

## Technical Details

### VTU Format Specification

- **Type:** UnstructuredGrid (arbitrary point cloud)
- **Cell type:** VTK_VERTEX (type code = 1)
- **Encoding:** ASCII (human-readable XML)
- **Version:** VTK XML 1.0
- **Compatibility:** ParaView 5.0+, VTK 8.0+

### PVD Format

- **Purpose:** Collection file for time series
- **Structure:** Lists all .vtu files with timesteps
- **Advantage:** Single file to load entire simulation

Example PVD:
```xml
<VTKFile type="Collection" version="1.0">
  <Collection>
    <DataSet timestep="0" file="snapshot_000000.vtu"/>
    <DataSet timestep="500" file="snapshot_000500.vtu"/>
    ...
  </Collection>
</VTKFile>
```

### Memory Usage

VTU export uses **streaming** - processes one snapshot at a time.

**Estimate:**
```
Memory per snapshot ≈ n_points × 60 bytes
                    = (N⁴ / downsample) × 60 bytes

Example: N=128, downsample=4
  → ~134M points → ~8 GB peak memory
```

For large simulations (N > 128), increase `downsample` or export in batches.

---

## Future Enhancements

Potential additions:
- [ ] Binary VTU encoding (smaller files, faster loading)
- [ ] Parallel VTU export (.pvtu) for distributed data
- [ ] Vortex line extraction as polylines
- [ ] Automatic 3D slicing (4D → 3D hypersurface)
- [ ] Integration with PyVista for programmatic visualization

---

## References

- **ParaView Guide:** https://www.paraview.org/paraview-guide/
- **VTK File Formats:** https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf
- **Python VTK Tools:** https://docs.pyvista.org/

---

## License & Citation

Part of the 4D BEC simulation suite. If you use this in research, please cite:

```bibtex
@software{bec4d_simulator,
  title = {4D Bose-Einstein Condensate Simulator with ParaView Export},
  author = {[Your Name]},
  year = {2025},
  url = {https://github.com/yourusername/BEC}
}
```

---

**Questions?** Open an issue or consult ParaView documentation.
