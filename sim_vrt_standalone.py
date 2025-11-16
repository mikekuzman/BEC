"""
sim_vrt_standalone.py - 4D BEC Simulator with VTU Export (Standalone Version)

STANDALONE VTU EXPORT FOR PARAVIEW 6.0.1
========================================
This is a self-contained version with NO external dependencies on other sim files.
Includes full 4D BEC simulation + VTU/VRT export for ParaView.

FEATURES:
- Complete 4D hypersphere BEC simulation
- VTU (VTK Unstructured Grid) export for ParaView 6.0.1
- PVD collection files for time series animation
- PKL initial state files for reproducibility
- Vortex detection with topological winding numbers
- Phonon and roton analysis

OUTPUT FILES:
- initial_state_N{N}_seed{seed}.pkl  # Reusable initial conditions
- paraview_output/snapshot_*.vtu     # Individual VTU files
- paraview_output/snapshot_series.pvd # ParaView collection (OPEN THIS!)
- paraview_output/README_ParaView.txt # Usage instructions

USAGE:
    python sim_vrt_standalone.py              # Run new simulation
    python sim_vrt_standalone.py --load <pkl> # Load previous initial state

PARAVIEW WORKFLOW:
    1. Open snapshot_series.pvd in ParaView 6.0.1
    2. Apply, change to "Point Gaussian" representation
    3. Color by "density" or "phase"
    4. Press Play for time animation
"""

import numpy as np
import cupy as cp
import pickle
import os
import json
import time
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Tuple, Optional
from dataclasses import dataclass
from scipy.spatial import cKDTree


@dataclass
class SimulationParams:
    """Parameters for 4D hypersphere BEC simulation"""
    # Physical parameters (dimensionless units: hbar=m=xi=1)
    R: float = 1000.0           # Hypersphere radius in healing lengths
    delta: float = 25.0         # Shell thickness in healing lengths
    g: float = 0.05             # Interaction strength (weak)
    omega: float = 0.03         # Rotation rate (moderate)

    # Computational parameters
    N: int = 128                 # Grid points per dimension
    dt: float = 0.001           # Time step
    n_neighbors: int = 6        # Number of neighbors for gradient calculation

    # Rotation plane (4D has 6 possible planes, using w-x plane)
    rotation_plane: Tuple[int, int] = (0, 1)  # (w, x) indices

    # Random seed for reproducibility
    random_seed: Optional[int] = None  # If None, generates random seed

    # Initial condition type
    initial_condition_type: str = "imaginary_time"
    imag_time_steps: int = 1000  # Number of imaginary time steps

    def __post_init__(self):
        self.box_size = self.R * 1.2  # Slightly larger than R
        self.dx = 2 * self.box_size / self.N  # Lattice spacing


class HypersphereBEC:
    """4D Hypersphere Quantum Superfluid Simulator"""

    def __init__(self, params: SimulationParams):
        self.p = params

        # Set random seed if provided
        if self.p.random_seed is not None:
            np.random.seed(self.p.random_seed)
            cp.random.seed(self.p.random_seed)
            print(f"Using random seed: {self.p.random_seed}")
        else:
            # Generate and store a random seed for reproducibility
            self.p.random_seed = np.random.randint(0, 2**31)
            np.random.seed(self.p.random_seed)
            cp.random.seed(self.p.random_seed)
            print(f"Generated random seed: {self.p.random_seed}")

        print(f"Initializing 4D BEC simulation:")
        print(f"  R = {self.p.R} ξ")
        print(f"  δ = {self.p.delta} ξ (R/δ = {self.p.R/self.p.delta:.1f})")
        print(f"  Grid: {self.p.N}^4, dx = {self.p.dx:.3f} ξ")
        print(f"  g = {self.p.g}, Ω = {self.p.omega}")

        # Find shell points efficiently
        self._find_shell_points()

        # Build neighbor lookup structure
        self._build_neighbor_tree()

        # Initialize wavefunction (sparse storage)
        self._initialize_wavefunction()

        print(f"  Active shell points: {self.n_active:,}")
        print(f"  Memory estimate: ~{self.n_active * 100 / 1e6:.1f} MB")

    def _find_shell_points(self):
        """Identify shell points without allocating full 4D grid (vectorized)"""
        print("  Scanning for shell points (vectorized)...")
        t_start = time.time()

        x = np.linspace(-self.p.box_size, self.p.box_size, self.p.N)
        r_inner = self.p.R - self.p.delta / 2
        r_outer = self.p.R + self.p.delta / 2

        # Chunked vectorization to avoid memory explosion
        chunk_size = 8  # Process 8 w-values at a time
        shell_points = []

        # Pre-compute x,y,z grids (reused for each w chunk)
        xv, yv, zv = np.meshgrid(x, x, x, indexing='ij')

        for chunk_start in range(0, self.p.N, chunk_size):
            chunk_end = min(chunk_start + chunk_size, self.p.N)
            w_chunk = x[chunk_start:chunk_end]

            # Vectorized computation for this w-chunk
            for wi in w_chunk:
                # Broadcast w across the 3D grid
                r = np.sqrt(wi**2 + xv**2 + yv**2 + zv**2)

                # Find all points in the shell
                mask = (r >= r_inner) & (r <= r_outer)

                # Extract coordinates where mask is True
                w_vals = np.full(mask.sum(), wi)
                x_vals = xv[mask]
                y_vals = yv[mask]
                z_vals = zv[mask]

                # Stack into (n_points, 4) array and append
                chunk_points = np.column_stack([w_vals, x_vals, y_vals, z_vals])
                shell_points.append(chunk_points)

            if chunk_end % 20 == 0 or chunk_end == self.p.N:
                print(f"    Progress: {chunk_end}/{self.p.N}")

        # Concatenate all chunks
        self.coords = np.vstack(shell_points) if shell_points else np.empty((0, 4))
        self.n_active = len(self.coords)

        t_elapsed = time.time() - t_start
        print(f"  Found {self.n_active:,} shell points in {t_elapsed:.2f}s")

        # Store on GPU
        self.coords_gpu = cp.asarray(self.coords)

    def _build_neighbor_tree(self):
        """Build KD-tree for fast neighbor lookup"""
        print("  Building neighbor tree...")
        t_start = time.time()

        self.tree = cKDTree(self.coords)

        # Find nearest neighbors for each point
        distances, indices = self.tree.query(self.coords, k=self.p.n_neighbors + 1)

        # Remove self (first neighbor is always self)
        self.neighbor_indices = indices[:, 1:].astype(np.int32)
        self.neighbor_distances = distances[:, 1:].astype(np.float64)

        # Transfer to GPU
        self.neighbor_indices_gpu = cp.asarray(self.neighbor_indices)
        self.neighbor_distances_gpu = cp.asarray(self.neighbor_distances)

        t_elapsed = time.time() - t_start
        avg_dist = np.mean(self.neighbor_distances)
        print(f"  Average neighbor distance: {avg_dist:.3f} ξ ({t_elapsed:.2f}s)")

    def _initialize_wavefunction(self):
        """Initialize the BEC order parameter ψ"""
        print(f"  Initializing wavefunction: {self.p.initial_condition_type}")

        if self.p.initial_condition_type == "uniform_noise":
            self._initialize_uniform_noise()
        elif self.p.initial_condition_type == "gaussian":
            self._initialize_gaussian()
        elif self.p.initial_condition_type == "imaginary_time":
            self._initialize_imaginary_time()
        else:
            raise ValueError(f"Unknown initial condition type: {self.p.initial_condition_type}")

    def _initialize_uniform_noise(self):
        """Initialize with uniform amplitude + small random perturbations"""
        self.psi = cp.ones(self.n_active, dtype=cp.complex128)

        # Add small random perturbations
        noise_amplitude = 0.01
        noise = noise_amplitude * (cp.random.randn(self.n_active) +
                                   1j * cp.random.randn(self.n_active))
        self.psi += noise
        print(f"    Uniform noise: amplitude=1.0, noise={noise_amplitude}")

    def _initialize_gaussian(self):
        """Initialize with a Gaussian wavepacket centered on the north pole"""
        coords_centered = self.coords_gpu.copy()
        coords_centered[:, 0] -= self.p.R  # Shift so north pole is at origin

        r_from_north = cp.sqrt(cp.sum(coords_centered**2, axis=1))
        sigma = self.p.delta * 2  # Gaussian width ~ 2*shell thickness

        # Gaussian envelope
        amplitude = cp.exp(-r_from_north**2 / (2 * sigma**2))

        # Add small random phase
        phase = 0.01 * cp.random.randn(self.n_active)
        self.psi = amplitude * cp.exp(1j * phase)

        # Normalize
        norm = cp.sqrt(cp.sum(cp.abs(self.psi)**2))
        self.psi /= norm
        self.psi *= cp.sqrt(self.n_active)  # Restore total density

        print(f"    Gaussian blob: centered at north pole, σ={sigma:.1f} ξ")

    def _initialize_imaginary_time(self):
        """Find ground state using imaginary time evolution"""
        print(f"    Running imaginary time evolution for {self.p.imag_time_steps} steps...")

        # Start with uniform + noise
        self.psi = cp.ones(self.n_active, dtype=cp.complex128)
        noise_amplitude = 0.01
        noise = noise_amplitude * (cp.random.randn(self.n_active) +
                                   1j * cp.random.randn(self.n_active))
        self.psi += noise

        # Imaginary time evolution: ψ(t+dt) = ψ(t) - dt*H*ψ(t), then normalize
        dt_imag = 0.01

        for step in range(self.p.imag_time_steps):
            # Compute energy terms (same as real time, but no rotation)
            laplacian = self.compute_laplacian()
            kinetic_term = -0.5 * laplacian

            density = cp.abs(self.psi)**2
            interaction_term = self.p.g * density * self.psi

            # Imaginary time step (gradient descent in energy)
            self.psi -= dt_imag * (kinetic_term + interaction_term)

            # Renormalize
            norm = cp.sqrt(cp.sum(cp.abs(self.psi)**2))
            self.psi /= norm
            self.psi *= cp.sqrt(self.n_active)

            if step % 200 == 0:
                energy = cp.sum(cp.abs(kinetic_term + interaction_term)**2)
                print(f"      Step {step}/{self.p.imag_time_steps}, Energy: {float(energy):.6e}")

        print(f"    Ground state found!")

    def compute_gradient(self, field, axis):
        """Compute gradient along specified axis using neighbor interpolation"""
        gradient = cp.zeros(self.n_active, dtype=cp.complex128)

        # Vector from point to each neighbor
        coord_diff = self.coords_gpu[:, axis:axis+1] - \
                     self.coords_gpu[self.neighbor_indices_gpu, axis]

        # Field difference to each neighbor
        field_diff = field[self.neighbor_indices_gpu] - field[:, cp.newaxis]

        # Weighted least squares gradient estimate
        weights = 1.0 / (self.neighbor_distances_gpu + 1e-10)
        numerator = cp.sum(field_diff * coord_diff * weights, axis=1)
        denominator = cp.sum(coord_diff**2 * weights, axis=1)

        gradient = numerator / (denominator + 1e-10)

        return gradient

    def compute_laplacian(self):
        """Compute 4D Laplacian using neighbor interpolation (vectorized)"""
        neighbor_vals = self.psi[self.neighbor_indices_gpu]
        dist_sq = self.neighbor_distances_gpu**2

        laplacian = cp.sum(
            2.0 * (neighbor_vals - self.psi[:, cp.newaxis]) / (dist_sq + 1e-10),
            axis=1
        )
        laplacian /= self.p.n_neighbors

        return laplacian

    def compute_rotation_term(self):
        """Compute rotation term: -Ω*L_z*ψ"""
        w = self.coords_gpu[:, 0]
        x = self.coords_gpu[:, 1]

        dpsi_dx = self.compute_gradient(self.psi, axis=1)
        dpsi_dw = self.compute_gradient(self.psi, axis=0)

        Lz_psi = w * dpsi_dx - x * dpsi_dw

        return -1j * self.p.omega * Lz_psi

    def evolve_step(self):
        """Single time step using split-step method"""
        # 1. Kinetic energy step (half)
        laplacian = self.compute_laplacian()
        self.psi *= cp.exp(-0.5j * self.p.dt * (-0.5 * laplacian))

        # 2. Interaction + rotation step (full)
        density = cp.abs(self.psi)**2
        rotation_term = self.compute_rotation_term()

        potential = self.p.g * density + rotation_term
        self.psi *= cp.exp(-1j * self.p.dt * potential)

        # 3. Kinetic energy step (half)
        laplacian = self.compute_laplacian()
        self.psi *= cp.exp(-0.5j * self.p.dt * (-0.5 * laplacian))

    def get_density(self):
        """Return density field |ψ|²"""
        return cp.abs(self.psi)**2

    def get_phase(self):
        """Return phase field arg(ψ)"""
        return cp.angle(self.psi)

    def get_velocity(self):
        """Return velocity field v = ∇φ (in units where ℏ/m = 1)"""
        phase_gpu = self.get_phase()

        # Compute velocity components in each direction
        v_w = cp.real(self.compute_gradient(cp.exp(1j * phase_gpu), axis=0) * (-1j))
        v_x = cp.real(self.compute_gradient(cp.exp(1j * phase_gpu), axis=1) * (-1j))
        v_y = cp.real(self.compute_gradient(cp.exp(1j * phase_gpu), axis=2) * (-1j))
        v_z = cp.real(self.compute_gradient(cp.exp(1j * phase_gpu), axis=3) * (-1j))

        # Stack into velocity field array
        velocity = cp.stack([v_w, v_x, v_y, v_z], axis=1)

        return velocity

    def detect_vortices(self):
        """Detect vortex cores using topological winding number verification"""
        density = self.get_density()
        phase = self.get_phase()

        # Step 1: Find candidate vortex points (low density)
        density_threshold = 0.1
        low_density_mask = density < density_threshold
        candidate_indices = cp.where(low_density_mask)[0]

        if len(candidate_indices) == 0:
            return cp.zeros(self.n_active, dtype=bool)

        # Step 2: Verify each candidate has non-zero winding number
        verified_vortices = cp.zeros(self.n_active, dtype=bool)

        for idx in candidate_indices:
            winding = self._compute_winding_number(int(idx), phase)
            if abs(winding) >= 0.5:
                verified_vortices[idx] = True

        return verified_vortices

    def _compute_winding_number(self, center_idx, phase):
        """Compute topological winding number around a point"""
        center_pos = self.coords_gpu[center_idx]

        loop_radius = 2.5 * self.p.dx
        tolerance = 0.8 * self.p.dx

        displacements = self.coords_gpu - center_pos
        distances = cp.sqrt(cp.sum(displacements**2, axis=1))

        annulus_mask = (distances > loop_radius - tolerance) & (distances < loop_radius + tolerance)
        loop_indices = cp.where(annulus_mask)[0]

        if len(loop_indices) < 6:
            return 0.0

        loop_displacements = displacements[loop_indices]
        angles = cp.arctan2(loop_displacements[:, 1], loop_displacements[:, 0])

        sorted_order = cp.argsort(angles)
        ordered_indices = loop_indices[sorted_order]

        loop_phases = phase[ordered_indices]

        total_winding = 0.0
        n_loop_points = len(ordered_indices)

        for i in range(n_loop_points):
            phase_curr = loop_phases[i]
            phase_next = loop_phases[(i + 1) % n_loop_points]

            phase_diff = cp.angle(cp.exp(1j * (phase_next - phase_curr)))
            total_winding += phase_diff

        winding_number = total_winding / (2 * cp.pi)

        return float(winding_number)

    def cluster_vortex_cores(self, vortex_mask):
        """Cluster nearby vortex core points into individual vortex lines"""
        vortex_indices = cp.asnumpy(cp.where(vortex_mask)[0])

        if len(vortex_indices) == 0:
            return []

        clustering_radius = 3.0 * self.p.dx

        vortex_coords = self.coords[vortex_indices]
        visited = np.zeros(len(vortex_indices), dtype=bool)
        clusters = []

        for i in range(len(vortex_indices)):
            if visited[i]:
                continue

            cluster_indices = [i]
            visited[i] = True
            stack = [i]

            while stack:
                current_idx = stack.pop()
                current_pos = vortex_coords[current_idx]

                for j in range(len(vortex_indices)):
                    if visited[j]:
                        continue

                    distance = np.linalg.norm(vortex_coords[j] - current_pos)
                    if distance < clustering_radius:
                        cluster_indices.append(j)
                        visited[j] = True
                        stack.append(j)

            cluster_core_indices = vortex_indices[cluster_indices]
            cluster_positions = vortex_coords[cluster_indices]

            center_of_mass = np.mean(cluster_positions, axis=0)

            distances_from_com = np.linalg.norm(cluster_positions - center_of_mass, axis=1)
            representative_idx = cluster_core_indices[np.argmin(distances_from_com)]

            phase_gpu = self.get_phase()
            quantum_number = self._compute_winding_number(int(representative_idx), phase_gpu)
            quantum_number_int = int(np.round(quantum_number))

            clusters.append({
                'core_indices': cluster_core_indices,
                'center_of_mass': center_of_mass,
                'representative_idx': representative_idx,
                'quantum_number': quantum_number_int,
                'n_points': len(cluster_core_indices)
            })

        return clusters

    def analyze_phonons(self):
        """Analyze P₀ phonon excitations"""
        density = self.get_density()
        velocity = self.get_velocity()

        n0 = cp.mean(density)
        delta_n = density - n0

        velocity_mag = cp.sqrt(cp.sum(velocity**2, axis=1))

        sound_speed = float(cp.sqrt(self.p.g * n0))
        healing_length = 1.0 / float(cp.sqrt(self.p.g * n0))

        typical_k = 2 * np.pi / self.p.dx
        phonon_energy_scale = sound_speed * typical_k

        delta_n_rms = float(cp.sqrt(cp.mean(delta_n**2)))
        velocity_rms = float(cp.sqrt(cp.mean(velocity_mag**2)))

        phonon_data = {
            'sound_speed': sound_speed,
            'healing_length': healing_length,
            'mean_density': float(n0),
            'density_fluctuation_rms': delta_n_rms,
            'velocity_rms': velocity_rms,
            'phonon_energy_scale': phonon_energy_scale,
            'typical_k': typical_k
        }

        return phonon_data

    def detect_rotons(self, density_threshold_factor=0.7):
        """Detect R₀ roton excitations"""
        density = self.get_density()
        velocity = self.get_velocity()

        n0 = cp.mean(density)

        density_threshold = density_threshold_factor * n0
        min_density_for_roton = 0.3 * n0

        candidate_mask = (density < density_threshold) & (density > min_density_for_roton)

        velocity_mag = cp.sqrt(cp.sum(velocity**2, axis=1))
        velocity_threshold = 0.05

        roton_mask = candidate_mask & (velocity_mag > velocity_threshold)

        roton_indices = cp.asnumpy(cp.where(roton_mask)[0])

        if len(roton_indices) == 0:
            return {
                'roton_indices': np.array([]),
                'roton_positions': np.array([]).reshape(0, 4),
                'roton_count': 0,
                'characteristic_wavelength': 0.0,
                'roton_density_mean': 0.0
            }

        roton_positions = self.coords[roton_indices]

        if len(roton_positions) > 1:
            from scipy.spatial.distance import pdist
            distances = pdist(roton_positions)
            characteristic_wavelength = float(np.median(distances))
        else:
            characteristic_wavelength = self.p.dx * 5

        roton_density_mean = float(cp.mean(density[roton_mask]))

        roton_data = {
            'roton_indices': roton_indices,
            'roton_positions': roton_positions,
            'roton_count': len(roton_indices),
            'characteristic_wavelength': characteristic_wavelength,
            'roton_density_mean': roton_density_mean
        }

        return roton_data

    def run(self, n_steps: int, save_every: int = 100):
        """Run simulation for n_steps"""
        print(f"\nRunning simulation for {n_steps} steps...")
        print(f"  Expected vortex nucleation timescale: ~{1.0/self.p.omega:.1f} steps")

        snapshots = []
        t_sim_start = time.time()
        t_last_report = t_sim_start
        steps_since_report = 0

        for step in range(n_steps):
            t_step_start = time.time()
            self.evolve_step()
            cp.cuda.Stream.null.synchronize()
            t_step_elapsed = time.time() - t_step_start

            steps_since_report += 1

            if step % save_every == 0:
                t_save_start = time.time()

                density = cp.asnumpy(self.get_density())
                phase = cp.asnumpy(self.get_phase())
                velocity = cp.asnumpy(self.get_velocity())
                coords = self.coords

                vortex_mask = self.detect_vortices()
                vortices = cp.asnumpy(vortex_mask)

                vortex_clusters = self.cluster_vortex_cores(vortex_mask)
                quantum_numbers = np.array([cluster['quantum_number'] for cluster in vortex_clusters])

                phonon_data = self.analyze_phonons()
                roton_data = self.detect_rotons()

                snapshots.append({
                    'step': step,
                    'coords': coords,
                    'density': density,
                    'phase': phase,
                    'velocity': velocity,
                    'vortices': vortices,
                    'vortex_quantum_numbers': quantum_numbers,
                    'vortex_clusters': vortex_clusters,
                    'phonon_data': phonon_data,
                    'roton_data': roton_data
                })

                n_vortex_lines = len(vortex_clusters)
                avg_density = float(cp.mean(density))

                quantum_counts = {}
                for qn in quantum_numbers:
                    quantum_counts[qn] = quantum_counts.get(qn, 0) + 1

                qn_summary = ', '.join([f"{qn:+d}×{count}" for qn, count in sorted(quantum_counts.items())])
                if not qn_summary:
                    qn_summary = "none"

                n_rotons = roton_data['roton_count']
                c_s = phonon_data['sound_speed']
                xi_heal = phonon_data['healing_length']

                t_save_elapsed = time.time() - t_save_start
                t_since_last = time.time() - t_last_report
                steps_per_sec = steps_since_report / t_since_last if t_since_last > 0 else 0

                print(f"  Step {step:5d}: <ρ>={avg_density:.3f}, c_s={c_s:.3f}, ξ={xi_heal:.2f}, "
                      f"vortices={n_vortex_lines} ({qn_summary}), rotons={n_rotons}")
                print(f"    Timing: {steps_per_sec:.1f} steps/s (snapshot: {t_save_elapsed:.2f}s, step: {t_step_elapsed*1000:.1f}ms)")

                t_last_report = time.time()
                steps_since_report = 0

                if np.isnan(avg_density) or avg_density > 1e6:
                    print(f"\n  WARNING: Numerical instability at step {step}!")
                    break

        t_sim_elapsed = time.time() - t_sim_start
        avg_steps_per_sec = n_steps / t_sim_elapsed if t_sim_elapsed > 0 else 0
        print(f"\n  Simulation complete: {n_steps} steps in {t_sim_elapsed:.1f}s ({avg_steps_per_sec:.1f} steps/s)")

        return snapshots

    def save_initial_state(self, filename=None):
        """Save initial state for reproducibility"""
        if filename is None:
            filename = f"initial_state_N{self.p.N}_seed{self.p.random_seed}.pkl"

        state = {
            'params': self.p,
            'random_seed': self.p.random_seed,
            'coords': self.coords,
            'psi_initial': cp.asnumpy(self.psi),
            'neighbor_indices': self.neighbor_indices,
            'neighbor_distances': self.neighbor_distances
        }

        with open(filename, 'wb') as f:
            pickle.dump(state, f)
        print(f"Initial state saved to {filename}")
        return filename

    @classmethod
    def load_initial_state(cls, filename):
        """Load initial state from file (fast - uses cached data)"""
        print(f"Loading initial state from {filename}...")
        with open(filename, 'rb') as f:
            state = pickle.load(f)

        sim = cls.__new__(cls)
        sim.p = state['params']

        sim.coords = state['coords']
        sim.n_active = len(sim.coords)
        sim.neighbor_indices = state['neighbor_indices']
        sim.neighbor_distances = state['neighbor_distances']

        sim.coords_gpu = cp.asarray(sim.coords)
        sim.neighbor_indices_gpu = cp.asarray(sim.neighbor_indices)
        sim.neighbor_distances_gpu = cp.asarray(sim.neighbor_distances)
        sim.psi = cp.asarray(state['psi_initial'])

        print(f"Initial state loaded successfully!")
        print(f"  Active shell points: {sim.n_active:,}")
        print(f"  Skipped shell scanning and neighbor tree (loaded from cache)")
        return sim


# ============================================================================
# VTU EXPORT FUNCTIONS FOR PARAVIEW 6.0.1
# ============================================================================

def export_snapshot_to_vtu(snapshot, output_file, downsample=1):
    """
    Export a single snapshot to VTU (VTK Unstructured Grid) format

    Args:
        snapshot: snapshot dict with coords, density, phase, velocity, vortices
        output_file: path to output .vtu file
        downsample: reduce points by this factor

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
    piece.set('NumberOfCells', str(n_points))

    # POINTS (x, y, z coordinates; w stored as scalar field)
    points_elem = ET.SubElement(piece, 'Points')
    points_data = ET.SubElement(points_elem, 'DataArray')
    points_data.set('type', 'Float32')
    points_data.set('Name', 'Points')
    points_data.set('NumberOfComponents', '3')
    points_data.set('format', 'ascii')

    points_text = []
    for coord in coords:
        x, y, z = coord[1], coord[2], coord[3]  # Use x,y,z as spatial coords
        points_text.append(f"{x} {y} {z}")

    points_data.text = "\n" + "\n".join(points_text) + "\n"

    # CELLS (Vertex cells - one per point)
    cells_elem = ET.SubElement(piece, 'Cells')

    connectivity = ET.SubElement(cells_elem, 'DataArray')
    connectivity.set('type', 'Int32')
    connectivity.set('Name', 'connectivity')
    connectivity.set('format', 'ascii')
    connectivity.text = "\n" + " ".join(str(i) for i in range(n_points)) + "\n"

    offsets = ET.SubElement(cells_elem, 'DataArray')
    offsets.set('type', 'Int32')
    offsets.set('Name', 'offsets')
    offsets.set('format', 'ascii')
    offsets.text = "\n" + " ".join(str(i+1) for i in range(n_points)) + "\n"

    types = ET.SubElement(cells_elem, 'DataArray')
    types.set('type', 'UInt8')
    types.set('Name', 'types')
    types.set('format', 'ascii')
    types.text = "\n" + " ".join(["1"] * n_points) + "\n"

    # POINT DATA (Scalar and vector fields)
    point_data = ET.SubElement(piece, 'PointData')
    point_data.set('Scalars', 'density')
    point_data.set('Vectors', 'velocity')

    # W coordinate (4th dimension)
    w_data = ET.SubElement(point_data, 'DataArray')
    w_data.set('type', 'Float32')
    w_data.set('Name', 'w_coordinate')
    w_data.set('format', 'ascii')
    w_data.text = "\n" + " ".join(f"{coord[0]}" for coord in coords) + "\n"

    # Density field
    density_data = ET.SubElement(point_data, 'DataArray')
    density_data.set('type', 'Float32')
    density_data.set('Name', 'density')
    density_data.set('format', 'ascii')
    density_data.text = "\n" + " ".join(f"{d}" for d in density) + "\n"

    # Phase field (normalized to [0, 1])
    phase_norm = (phase + np.pi) / (2 * np.pi)
    phase_data = ET.SubElement(point_data, 'DataArray')
    phase_data.set('type', 'Float32')
    phase_data.set('Name', 'phase')
    phase_data.set('format', 'ascii')
    phase_data.text = "\n" + " ".join(f"{p}" for p in phase_norm) + "\n"

    # Vortex marker (0 or 1)
    vortex_data = ET.SubElement(point_data, 'DataArray')
    vortex_data.set('type', 'Float32')
    vortex_data.set('Name', 'vortex_marker')
    vortex_data.set('format', 'ascii')
    vortex_data.text = "\n" + " ".join(f"{float(v)}" for v in vortices) + "\n"

    # 4D Velocity field
    velocity_data = ET.SubElement(point_data, 'DataArray')
    velocity_data.set('type', 'Float32')
    velocity_data.set('Name', 'velocity_4D')
    velocity_data.set('NumberOfComponents', '4')
    velocity_data.set('format', 'ascii')
    velocity_text = []
    for vel in velocity:
        velocity_text.append(f"{vel[0]} {vel[1]} {vel[2]} {vel[3]}")
    velocity_data.text = "\n" + "\n".join(velocity_text) + "\n"

    # 3D Velocity magnitude
    velocity_3d_mag = np.sqrt(velocity[:, 1]**2 + velocity[:, 2]**2 + velocity[:, 3]**2)
    velocity_mag_data = ET.SubElement(point_data, 'DataArray')
    velocity_mag_data.set('type', 'Float32')
    velocity_mag_data.set('Name', 'velocity_3D_magnitude')
    velocity_mag_data.set('format', 'ascii')
    velocity_mag_data.text = "\n" + " ".join(f"{v}" for v in velocity_3d_mag) + "\n"

    # Radial distance from origin
    radial_dist = np.sqrt(np.sum(coords**2, axis=1))
    radial_data = ET.SubElement(point_data, 'DataArray')
    radial_data.set('type', 'Float32')
    radial_data.set('Name', 'radial_distance')
    radial_data.set('format', 'ascii')
    radial_data.text = "\n" + " ".join(f"{r}" for r in radial_dist) + "\n"

    # Write XML to file with pretty formatting
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

    # Create PVD collection file
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
        rel_path = os.path.basename(file_info['filename'])
        dataset.set('file', rel_path)

    # Write PVD file
    pvd_string = ET.tostring(pvd_root, encoding='unicode')
    pvd_dom = minidom.parseString(pvd_string)
    pretty_pvd = pvd_dom.toprettyxml(indent="  ")

    with open(pvd_path, 'w') as f:
        f.write(pretty_pvd)

    # Create README for ParaView usage
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
     - Change representation to "Point Gaussian"
     - Color by: "density" (shows condensate density)
     - Adjust Gaussian Radius: ~5-10

  4. Animation:
     - Use "Play" button to animate through time steps
     - View → Animation View for timeline control

  5. Vortex visualization:
     - Add "Threshold" filter
     - Threshold by "vortex_marker" > 0.5
     - Color by "phase" to see vortex circulation

  6. Advanced:
     - Glyph filter for velocity vectors
     - Contour filter for density isosurfaces
     - Calculator filter to swap axes (w ↔ z)

TIPS:
  - Dense clouds: Use "Point Gaussian" with opacity ~0.3
  - Vortex cores: Threshold density < 0.1
  - Use colormap "Viridis" or "Plasma" for density
  - Save animations: File → Save Animation

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Random seed: {params.random_seed}
"""

    with open(readme_path, 'w') as f:
        f.write(readme_content)

    # Summary
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


# ============================================================================
# MAIN PROGRAM
# ============================================================================

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
                downsample=4,  # Adjust based on desired resolution
                pvd_filename='snapshot_series.pvd'
            )

            # PKL filename for reference
            pkl_filename = f'initial_state_N{sim.p.N}_seed{sim.p.random_seed}.pkl'

            print(f"\nFiles created:")
            print(f"  Initial state (PKL): {pkl_filename}")
            print(f"  ParaView series: {export_summary['pvd_file']}")
            print(f"  Individual snapshots: {len(export_summary['vtu_files'])} .vtu files")
            print(f"\nNext steps:")
            print(f"  1. Open ParaView 6.0.1")
            print(f"  2. File → Open → {export_summary['pvd_file']}")
            print(f"  3. Click 'Apply' and explore!")
            print(f"\nTo reuse initial conditions:")
            print(f"  python sim_vrt_standalone.py --load {pkl_filename}")
        else:
            print("\nSimulation became unstable.")
            print(f"Random seed {sim.p.random_seed} produced unstable initial conditions.")
            print("Try running again with a different seed.")
