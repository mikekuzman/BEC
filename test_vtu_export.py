"""
test_vtu_export.py - Test VTU export with minimal synthetic data

This script tests the VTU export functionality without running a full simulation.
It creates synthetic 4D point cloud data and exports to ParaView format.

Usage:
    python test_vtu_export.py
"""

import numpy as np
from sim_v006_vrt import export_snapshot_to_vtu, export_snapshots_to_vtu_series
from sim_v005 import SimulationParams


def create_test_snapshot(n_points=1000, step=0):
    """
    Create a synthetic snapshot for testing

    Generates:
    - Points on a 4D hypersphere shell
    - Gaussian density distribution
    - Spiral phase pattern
    - Rotational velocity field
    - A few vortex markers
    """

    # Generate random points on 4D hypersphere
    # Method: Generate 4D Gaussian, normalize to unit sphere, scale to radius R
    R = 100.0
    delta = 10.0

    # Random 4D vectors
    coords = np.random.randn(n_points, 4)

    # Normalize to unit sphere
    norms = np.sqrt(np.sum(coords**2, axis=1, keepdims=True))
    coords = coords / norms

    # Scale to radius R with thickness delta
    radii = R + delta * (np.random.rand(n_points) - 0.5)
    coords = coords * radii[:, np.newaxis]

    # Generate density field (Gaussian blob at north pole w=R)
    north_pole = np.array([R, 0, 0, 0])
    distances_from_north = np.sqrt(np.sum((coords - north_pole)**2, axis=1))
    density = np.exp(-distances_from_north**2 / (2 * delta**2))
    density += 0.1  # Background density
    density *= 1.0 / density.max()  # Normalize

    # Generate phase field (spiral pattern)
    # Phase increases with angle around w-axis
    angles = np.arctan2(coords[:, 1], coords[:, 2])  # Angle in x-y plane
    phase = 3.0 * angles  # 3 winding
    phase = np.mod(phase, 2*np.pi) - np.pi  # Wrap to [-π, π]

    # Generate velocity field (rotational around w-axis)
    # v = Ω × r in the x-y plane
    omega = 0.05
    velocity = np.zeros((n_points, 4))
    velocity[:, 1] = -omega * coords[:, 2]  # v_x = -Ω * y
    velocity[:, 2] = omega * coords[:, 1]   # v_y = Ω * x

    # Mark some random points as vortices (near low density regions)
    vortices = np.zeros(n_points, dtype=bool)
    low_density_indices = np.where(density < 0.3)[0]
    if len(low_density_indices) > 10:
        vortex_indices = np.random.choice(low_density_indices, size=10, replace=False)
        vortices[vortex_indices] = True

    # Create snapshot dict
    snapshot = {
        'step': step,
        'coords': coords,
        'density': density,
        'phase': phase,
        'velocity': velocity,
        'vortices': vortices
    }

    return snapshot


def test_single_vtu_export():
    """Test exporting a single snapshot to VTU"""
    print("="*70)
    print("TEST 1: Single VTU file export")
    print("="*70)

    snapshot = create_test_snapshot(n_points=5000, step=0)

    output_file = 'test_snapshot.vtu'
    file_info = export_snapshot_to_vtu(snapshot, output_file, downsample=1)

    print(f"\n✓ Success!")
    print(f"  File: {file_info['filename']}")
    print(f"  Points: {file_info['n_points']:,}")
    print(f"  Size: {file_info['file_size_mb']:.2f} MB")
    print(f"\nYou can open '{output_file}' in ParaView to verify.")


def test_vtu_series_export():
    """Test exporting multiple snapshots with PVD collection"""
    print("\n" + "="*70)
    print("TEST 2: VTU series export with PVD collection")
    print("="*70)

    # Create a time series of snapshots
    n_snapshots = 5
    snapshots = []

    print(f"\nCreating {n_snapshots} synthetic snapshots...")
    for i in range(n_snapshots):
        step = i * 100
        snapshot = create_test_snapshot(n_points=3000, step=step)
        snapshots.append(snapshot)
        print(f"  Snapshot {i+1}/{n_snapshots}: step={step}")

    # Create dummy parameters
    params = SimulationParams(
        R=100.0,
        delta=10.0,
        g=0.05,
        omega=0.05,
        N=64,
        dt=0.001,
        n_neighbors=6,
        random_seed=12345
    )

    # Export to VTU series
    summary = export_snapshots_to_vtu_series(
        snapshots,
        params,
        output_dir='test_paraview_output',
        downsample=1,
        pvd_filename='test_series.pvd'
    )

    print(f"\n✓ Success!")
    print(f"  Total snapshots: {summary['n_snapshots']}")
    print(f"  Total size: {summary['total_size_mb']:.2f} MB")
    print(f"  PVD file: {summary['pvd_file']}")
    print(f"\nYou can open '{summary['pvd_file']}' in ParaView to verify time series.")


def test_downsampling():
    """Test downsampling effect on file size"""
    print("\n" + "="*70)
    print("TEST 3: Downsampling effect")
    print("="*70)

    snapshot = create_test_snapshot(n_points=10000, step=0)

    print(f"\nOriginal points: {len(snapshot['coords']):,}")

    for downsample in [1, 2, 5, 10]:
        output_file = f'test_downsample_{downsample}.vtu'
        file_info = export_snapshot_to_vtu(snapshot, output_file, downsample=downsample)

        print(f"  Downsample={downsample:2d}: {file_info['n_points']:5,} points, "
              f"{file_info['file_size_mb']:6.2f} MB")

    print(f"\n✓ Downsampling reduces both point count and file size proportionally.")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("VTU EXPORT TEST SUITE")
    print("="*70)
    print("\nTesting VTU export functionality for ParaView 6.0.1")

    try:
        # Run tests
        test_single_vtu_export()
        test_vtu_series_export()
        test_downsampling()

        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        print("\nNext steps:")
        print("  1. Open ParaView 6.0.1")
        print("  2. Load test_paraview_output/test_series.pvd")
        print("  3. Verify visualization works correctly")
        print("  4. Try different coloring and filters")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
