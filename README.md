# 3D Finite Volume Method (FVM) Pipe Flow Simulation

A Python implementation of a 3D pipe flow simulation using the Finite Volume Method (FVM) with the SIMPLE algorithm for pressure-velocity coupling.

## Features

- **3D Simulation**: Models fluid flow through a rectangular pipe
- **Finite Volume Method**: Discretizes the Navier-Stokes equations using FVM
- **SIMPLE Algorithm**: Handles pressure-velocity coupling
- **VTK Output**: Saves results in VTK format for visualization in ParaView or VisIt
- **Matplotlib Visualization**: Generates 3D plots of velocity and pressure distributions
- **Residual Tracking**: Monitors and plots convergence of velocity and pressure equations
- **Configurable Parameters**: Adjust fluid properties, grid resolution, and pipe dimensions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/3d-fvm-pipe-flow.git
cd 3d-fvm-pipe-flow
```

2. Install required dependencies:
```bash
pip install numpy matplotlib scipy vtk
```

## Usage

Run the simulation with default parameters:
```bash
python pipe_flow_simulation.py
```

Customize the simulation with command-line arguments:
```bash
python pipe_flow_simulation.py \
    --velocity 2.0 \
    --density 1000 \
    --viscosity 0.001 \
    --resolution 0.05 \
    --dimension "0.5,0.5,20" \
    --output_dir results \
    --vtk my_simulation.vtk \
    --png my_visualization.png
```

### Command-line Arguments

- `--velocity`: Inlet velocity [m/s] (default: 1.0)
- `--density`: Fluid density [kg/m³] (default: 1.0)
- `--viscosity`: Dynamic viscosity [Pa·s] (default: 0.001)
- `--resolution`: Grid resolution [m] (default: 0.1)
- `--dimension`: Pipe dimensions as "width,height,length" in meters (default: "1,1,10")
- `--inlet_mass_flow_rate`: Inlet mass flow rate [kg/s] (overrides velocity if specified)
- `--output_dir`: Output directory for results (default: "output")
- `--vtk`: VTK output filename (default: "pipe_flow.vtk")
- `--png`: PNG visualization filename (default: "pipe_flow.png")

## Output Files

The simulation generates:
1. **VTK file**: Contains 3D velocity and pressure data for visualization in ParaView or VisIt
2. **PNG file**: 2D visualizations including velocity magnitude, pressure distribution, outlet profile, and pressure drop
3. **Residual plot**: Convergence history of velocity and pressure equations

## Mathematical Formulation

The simulation solves the incompressible Navier-Stokes equations:

**Continuity equation**:
∇·u = 0

**Momentum equation**:
ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u

The SIMPLE (Semi-Implicit Method for Pressure-Linked Equations) algorithm is used for pressure-velocity coupling.

## Code Structure

- `FVM3DPipeFlow`: Main class containing the simulation implementation
  - `__init__`: Initializes fluid properties and grid
  - `apply_initial_conditions`: Sets up parabolic inlet velocity profile
  - `solve_momentum_z`: Solves the z-momentum equation
  - `solve_pressure_correction`: Solves pressure correction equation
  - `solve`: Main solution loop using SIMPLE algorithm
  - `save_vtk`: Exports results to VTK format
  - `visualize_matplotlib`: Creates 3D visualizations

## Visualization

Results can be visualized using:
1. **ParaView** or **VisIt**: Open the VTK file for interactive 3D visualization
2. **Matplotlib**: The generated PNG file shows various 2D cross-sections and plots

## Example Results

The simulation provides:
- Velocity magnitude distribution throughout the pipe
- Pressure distribution and pressure drop along the pipe length
- Developed velocity profile at the outlet
- Convergence history of the iterative solution

## Applications

This code can be used for:
- Educational purposes to understand FVM and the SIMPLE algorithm
- Preliminary analysis of pipe flow problems
- Testing different fluid properties and boundary conditions
- Benchmarking against analytical solutions for fully developed flow

## Limitations

- Currently implements only the z-momentum equation (axial flow)
- Uses a simplified treatment of boundary conditions
- Limited to rectangular pipe geometry
- Does not include turbulence modeling

## Future Enhancements

- Implementation of all three momentum equations
- Additional turbulence models (k-ε, k-ω, etc.)
- Support for curved pipes and complex geometries
- Parallel computation for larger grids
- Transient simulation capability

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

1. Patankar, S. V. (1980). Numerical Heat Transfer and Fluid Flow.
2. Ferziger, J. H., & Perić, M. (2002). Computational Methods for Fluid Dynamics.
3. Versteeg, H. K., & Malalasekera, W. (2007). An Introduction to Computational Fluid Dynamics.
