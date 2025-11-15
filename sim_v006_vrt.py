"""
sim_v006_vrt.py - 4D Bose-Einstein Condensate Simulator with VTK Export

V006 PARAVIEW EXPORT FORMAT:
===========================
- VTU (VTK Unstructured Grid) format for ParaView 6.0.1
- One .vtu file per snapshot
- PVD collection file for time series animation
- Full 4D coordinate support (w, x, y, z as point coordinates + scalars)
- Scalar fields: density, phase, vortex_marker
- Vector field: 4D velocity
- Per-snapshot statistics preserved
- Compatible with ParaView filters and animations

USAGE:
    python sim_v006_vrt.py              # Run new simulation
    python sim_v006_vrt.py --load <file.pkl>  # Load previous initial state

OUTPUT FILES:
    - initial_state_N{N}_seed{seed}.pkl  # Initial state (reusable)
    - snapshot_{step:06d}.vtu            # Individual snapshot files
    - snapshot_series.pvd                # ParaView collection (open this in ParaView!)

PARAVIEW WORKFLOW:
    1. Open snapshot_series.pvd in ParaView 6.0.1
    2. Apply "Glyph" filter for 3D projection (use w,x,y or x,y,z)
    3. Color by density, phase, or velocity magnitude
    4. Use Animation panel for time evolution
    5. Apply "Threshold" filter to isolate vortices (vortex_marker > 0.5)
"""

# Import all functionality from sim_v005.py
from sim_v005 import (
    np, cp, pickle, os, json, time,
    SimulationParams, HypersphereBEC,
    export_snapshots_to_json, export_snapshots_to_msgpack
)
import xml.etree.ElementTree as ET
from xml.dom import minidom
import base64
import struct


def export_snapshot_to_vtu(snapshot, output_file, downsample=1):
    """
    Export a single snapshot to VTU (VTK Unstructured Grid) format

    VTU format supports:
    - Arbitrary point clouds (perfect for sparse 4D shell)
    - Multiple scalar/vector fields
    - ParaView 6.0.1 native format

    Args:
        snapshot: snapshot dict with coords, density, phase, velocity, vortices
        output_file: path to output .vtu file
        downsample: reduce points by this factor (default=1, no downsampling)

    Returns:
        dict with file info and statistics
    """

    print(f"  Exporting {output_file}...")

    # Downsample data
    coords = snapshot['coords'][::downsample]
    density = snapshot['density'][::downsample]
    phase = snapshot['phase'][::downsample]
    velocity = snapshot['velocity'][::downsample]
    vortices = snapshot['vortices'][::downsample]

    n_points = len(coords)

    # Create VTK XML structure
    vtk_file = ET.Element('VTKFile')
    vtk_file.set('type', 'UnstructuredGrid')
    vtk_file.set('version', '1.0')
    vtk_file.set('byte_order', 'LittleEndian')
    vtk_file.set('header_type', 'UInt64')

    unstructured_grid = ET.SubElement(vtk_file, 'UnstructuredGrid')
    piece = ET.SubElement(unstructured_grid, 'Piece')
    piece.set('NumberOfPoints', str(n_points))
    piece.set('NumberOfCells', str(n_points))  # Each point is a vertex cell

    # ========================================================================
    # POINTS (4D coordinates stored as x,y,z with w as scalar field)
    # ========================================================================
    # ParaView natively handles 3D (x,y,z), so we store (x,y,z) as coordinates
    # and w as a scalar field. User can swap axes in ParaView filters.

    points_elem = ET.SubElement(piece, 'Points')
    points_data = ET.SubElement(points_elem, 'DataArray')
    points_data.set('type', 'Float32')
    points_data.set('Name', 'Points')
    points_data.set('NumberOfComponents', '3')
    points_data.set('format', 'ascii')

    # Store x, y, z as point coordinates (w will be a scalar field)
    points_text = []
    for coord in coords:
        # coord = [w, x, y, z]
        x, y, z = coord[1], coord[2], coord[3]  # Use x,y,z as spatial coords
        points_text.append(f"{x} {y} {z}")

    points_data.text = "\n" + "\n".join(points_text) + "\n"

    # ========================================================================
    # CELLS (Vertex cells - one per point)
    # ========================================================================
    cells_elem = ET.SubElement(piece, 'Cells')

    # Connectivity (each cell is a single vertex)
    connectivity = ET.SubElement(cells_elem, 'DataArray')
    connectivity.set('type', 'Int32')
    connectivity.set('Name', 'connectivity')
    connectivity.set('format', 'ascii')
    connectivity.text = "\n" + " ".join(str(i) for i in range(n_points)) + "\n"

    # Offsets (cumulative count)
    offsets = ET.SubElement(cells_elem, 'DataArray')
    offsets.set('type', 'Int32')
    offsets.set('Name', 'offsets')
    offsets.set('format', 'ascii')
    offsets.text = "\n" + " ".join(str(i+1) for i in range(n_points)) + "\n"

    # Cell types (1 = VTK_VERTEX)
    types = ET.SubElement(cells_elem, 'DataArray')
    types.set('type', 'UInt8')
    types.set('Name', 'types')
    types.set('format', 'ascii')
    types.text = "\n" + " ".join(["1"] * n_points) + "\n"

    # ========================================================================
    # POINT DATA (Scalar and vector fields)
    # ========================================================================
    point_data = ET.SubElement(piece, 'PointData')
    point_data.set('Scalars', 'density')
    point_data.set('Vectors', 'velocity')

    # --- W coordinate (4th dimension) ---
    w_data = ET.SubElement(point_data, 'DataArray')
    w_data.set('type', 'Float32')
    w_data.set('Name', 'w_coordinate')
    w_data.set('format', 'ascii')
    w_data.text = "\n" + " ".join(f"{coord[0]}" for coord in coords) + "\n"

    # --- Density field ---
    density_data = ET.SubElement(point_data, 'DataArray')
    density_data.set('type', 'Float32')
    density_data.set('Name', 'density')
    density_data.set('format', 'ascii')
    density_data.text = "\n" + " ".join(f"{d}" for d in density) + "\n"

    # --- Phase field (normalized to [0, 1]) ---
    phase_norm = (phase + np.pi) / (2 * np.pi)
    phase_data = ET.SubElement(point_data, 'DataArray')
    phase_data.set('type', 'Float32')
    phase_data.set('Name', 'phase')
    phase_data.set('format', 'ascii')
    phase_data.text = "\n" + " ".join(f"{p}" for p in phase_norm) + "\n"

    # --- Vortex marker (0 or 1) ---
    vortex_data = ET.SubElement(point_data, 'DataArray')
    vortex_data.set('type', 'Float32')
    vortex_data.set('Name', 'vortex_marker')
    vortex_data.set('format', 'ascii')
    vortex_data.text = "\n" + " ".join(f"{float(v)}" for v in vortices) + "\n"

    # --- 4D Velocity field (stored as 4-component vector) ---
    # ParaView handles multi-component vectors
    velocity_data = ET.SubElement(point_data, 'DataArray')
    velocity_data.set('type', 'Float32')
    velocity_data.set('Name', 'velocity_4D')
    velocity_data.set('NumberOfComponents', '4')
    velocity_data.set('format', 'ascii')
    velocity_text = []
    for vel in velocity:
        velocity_text.append(f"{vel[0]} {vel[1]} {vel[2]} {vel[3]}")
    velocity_data.text = "\n" + "\n".join(velocity_text) + "\n"

    # --- 3D Velocity magnitude (for easier visualization) ---
    velocity_3d_mag = np.sqrt(velocity[:, 1]**2 + velocity[:, 2]**2 + velocity[:, 3]**2)
    velocity_mag_data = ET.SubElement(point_data, 'DataArray')
    velocity_mag_data.set('type', 'Float32')
    velocity_mag_data.set('Name', 'velocity_3D_magnitude')
    velocity_mag_data.set('format', 'ascii')
    velocity_mag_data.text = "\n" + " ".join(f"{v}" for v in velocity_3d_mag) + "\n"

    # --- Radial distance from origin (useful for filtering) ---
    radial_dist = np.sqrt(np.sum(coords**2, axis=1))
    radial_data = ET.SubElement(point_data, 'DataArray')
    radial_data.set('type', 'Float32')
    radial_data.set('Name', 'radial_distance')
    radial_data.set('format', 'ascii')
    radial_data.text = "\n" + " ".join(f"{r}" for r in radial_dist) + "\n"

    # ========================================================================
    # Write XML to file with pretty formatting
    # ========================================================================
    xml_string = ET.tostring(vtk_file, encoding='unicode')
    dom = minidom.parseString(xml_string)
    pretty_xml = dom.toprettyxml(indent="  ")

    with open(output_file, 'w') as f:
        f.write(pretty_xml)

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)

    return {
        'filename': output_file,
        'n_points': n_points,
        'file_size_mb': file_size_mb,
        'step': snapshot['step']
    }


def export_snapshots_to_vtu_series(snapshots, params, output_dir='paraview_output',
                                    downsample=1, pvd_filename='snapshot_series.pvd'):
    """
    Export multiple snapshots to VTU format with PVD collection file

    Creates:
    - One .vtu file per snapshot
    - One .pvd collection file for time series in ParaView

    Args:
        snapshots: list of snapshot dicts
        params: SimulationParams object
        output_dir: directory to store output files
        downsample: reduce points by this factor
        pvd_filename: name of PVD collection file

    Returns:
        dict with export summary
    """

    print(f"\nExporting {len(snapshots)} snapshots to VTU format for ParaView...")
    print(f"  Output directory: {output_dir}")
    print(f"  Downsample factor: {downsample}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Export each snapshot to VTU
    vtu_files = []
    total_size_mb = 0

    for i, snapshot in enumerate(snapshots):
        step = snapshot['step']
        vtu_filename = os.path.join(output_dir, f'snapshot_{step:06d}.vtu')

        file_info = export_snapshot_to_vtu(snapshot, vtu_filename, downsample=downsample)
        vtu_files.append(file_info)
        total_size_mb += file_info['file_size_mb']

        print(f"    [{i+1}/{len(snapshots)}] Step {step}: {file_info['n_points']:,} points, "
              f"{file_info['file_size_mb']:.2f} MB")

    # ========================================================================
    # Create PVD collection file (ParaView Data format)
    # ========================================================================
    print(f"\n  Creating PVD collection file: {pvd_filename}")

    pvd_path = os.path.join(output_dir, pvd_filename)

    pvd_root = ET.Element('VTKFile')
    pvd_root.set('type', 'Collection')
    pvd_root.set('version', '1.0')
    pvd_root.set('byte_order', 'LittleEndian')

    collection = ET.SubElement(pvd_root, 'Collection')

    for file_info in vtu_files:
        dataset = ET.SubElement(collection, 'DataSet')
        dataset.set('timestep', str(file_info['step']))
        dataset.set('part', '0')
        # Use relative path
        rel_path = os.path.basename(file_info['filename'])
        dataset.set('file', rel_path)

    # Write PVD file
    pvd_string = ET.tostring(pvd_root, encoding='unicode')
    pvd_dom = minidom.parseString(pvd_string)
    pretty_pvd = pvd_dom.toprettyxml(indent="  ")

    with open(pvd_path, 'w') as f:
        f.write(pretty_pvd)

    # ========================================================================
    # Create README for ParaView usage
    # ========================================================================
    readme_path = os.path.join(output_dir, 'README_ParaView.txt')

    readme_content = f"""4D BEC Simulation - ParaView Visualization Guide
{'='*70}

SIMULATION PARAMETERS:
  - Hypersphere radius: R = {params.R} ξ
  - Shell thickness: δ = {params.delta} ξ
  - Interaction strength: g = {params.g}
  - Rotation rate: Ω = {params.omega}
  - Grid resolution: N = {params.N}
  - Random seed: {params.random_seed}

OUTPUT FILES:
  - {pvd_filename} ← OPEN THIS FILE IN PARAVIEW!
  - snapshot_*.vtu (individual snapshots)
  - Total: {len(snapshots)} snapshots, {total_size_mb:.1f} MB

DATA FIELDS:
  Coordinates:
    - x, y, z: Spatial coordinates (3D projection)
    - w_coordinate: 4th dimension (scalar field)

  Scalar Fields:
    - density: Condensate density |ψ|²
    - phase: Phase angle (normalized to [0,1])
    - vortex_marker: 1 = vortex core, 0 = normal fluid
    - radial_distance: Distance from origin
    - velocity_3D_magnitude: Speed in x,y,z subspace

  Vector Fields:
    - velocity_4D: Full 4D velocity (v_w, v_x, v_y, v_z)

PARAVIEW WORKFLOW:
  1. Open ParaView 6.0.1

  2. File → Open → Select "{pvd_filename}"
     Click "Apply" in Properties panel

  3. Basic visualization:
     - Change representation to "Point Gaussian" or "Points"
     - Color by: "density" (shows condensate density)
     - Adjust point size as needed

  4. Animation:
     - Use "Play" button to animate through time steps
     - View → Animation View for timeline control

  5. Vortex visualization:
     - Add "Threshold" filter
     - Threshold by "vortex_marker" > 0.5
     - Color by "phase" to see vortex circulation

  6. 3D slicing (4D → 3D projection):
     - Current view: (x, y, z) coordinates
     - To use w-coordinate: Calculator filter to swap axes
     - Example: Create new coords = w*iHat + x*jHat + y*kHat

  7. Advanced filters:
     - Glyph: Visualize velocity vectors
     - Contour: Iso-density surfaces
     - Slice: Cut through the hypersphere
     - Calculator: Compute derived quantities

TIPS:
  - For dense point clouds: Use "Point Gaussian" with opacity ~0.3
  - Vortex cores appear as low-density regions (threshold density < 0.1)
  - Phase singularities mark vortex centers
  - Use "Extract Surface" → "Clean" to reduce point count if slow

TROUBLESHOOTING:
  - If points look sparse: Reduce downsample factor in export
  - If loading is slow: Try loading single .vtu file first
  - Memory issues: Increase downsample factor or reduce N

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Random seed: {params.random_seed} (use for reproducibility)
"""

    with open(readme_path, 'w') as f:
        f.write(readme_content)

    # ========================================================================
    # Summary
    # ========================================================================
    summary = {
        'n_snapshots': len(snapshots),
        'output_dir': output_dir,
        'pvd_file': pvd_path,
        'vtu_files': [f['filename'] for f in vtu_files],
        'total_size_mb': total_size_mb,
        'downsample_factor': downsample,
        'total_points': sum(f['n_points'] for f in vtu_files)
    }

    print(f"\n{'='*70}")
    print(f"VTU Export Complete!")
    print(f"{'='*70}")
    print(f"  Output directory: {output_dir}")
    print(f"  Snapshots: {len(snapshots)}")
    print(f"  Total size: {total_size_mb:.1f} MB")
    print(f"  Total points: {summary['total_points']:,}")
    print(f"\n  → OPEN IN PARAVIEW: {pvd_path}")
    print(f"  → Read instructions: {readme_path}")
    print(f"{'='*70}\n")

    return summary


# Example usage
if __name__ == "__main__":
    import sys

    # Check if loading previous state
    if len(sys.argv) > 1 and sys.argv[1] == '--load':
        if len(sys.argv) > 2:
            filename = sys.argv[2]
        else:
            filename = "initial_state.pkl"
        sim = HypersphereBEC.load_initial_state(filename)
    else:
        params = SimulationParams(
            R=1000.0,
            delta=25.0,
            g=0.05,
            omega=0.03,
            N=128,
            dt=0.001,
            n_neighbors=6,
            random_seed=42,  # Fixed seed for reproducibility
            initial_condition_type="imaginary_time",
            imag_time_steps=1000
        )

        sim = HypersphereBEC(params)
        sim.save_initial_state()

    # Run simulation
    snapshots = sim.run(n_steps=5000, save_every=500)

    if len(snapshots) > 0:
        print(f"\nSimulation complete! Captured {len(snapshots)} snapshots.")

        # Check if stable
        final_density = snapshots[-1]['density']
        if not np.isnan(final_density.mean()) and final_density.max() < 1e6:
            print("Simulation stable! Exporting to VTU format...")

            # Export to VTU format for ParaView
            export_summary = export_snapshots_to_vtu_series(
                snapshots,
                sim.p,
                output_dir='paraview_output',
                downsample=4,  # Adjust based on desired resolution vs file size
                pvd_filename='snapshot_series.pvd'
            )

            # Also keep the original PKL export for Python analysis
            pkl_filename = f'initial_state_N{sim.p.N}_seed{sim.p.random_seed}.pkl'

            print(f"\nFiles created:")
            print(f"  Initial state (PKL): {pkl_filename}")
            print(f"  ParaView series: {export_summary['pvd_file']}")
            print(f"  Individual snapshots: {len(export_summary['vtu_files'])} .vtu files")
            print(f"\nNext steps:")
            print(f"  1. Open ParaView 6.0.1")
            print(f"  2. File → Open → {export_summary['pvd_file']}")
            print(f"  3. Click 'Apply' and explore the 4D BEC simulation!")
            print(f"\nTo reuse initial conditions:")
            print(f"  python sim_v006_vrt.py --load {pkl_filename}")
        else:
            print("\nSimulation became unstable.")
            print(f"Random seed {sim.p.random_seed} produced unstable initial conditions.")
            print("Try running again with a different seed.")
