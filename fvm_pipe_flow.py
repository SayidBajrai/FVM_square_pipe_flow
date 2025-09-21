import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import diags, lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import vtk
from vtk.util import numpy_support
import argparse
import os

class FVM3DPipeFlow:
    def __init__(self, velocity=1.0, density=1.0, viscosity=0.001, resolution=0.1, 
                 dimensions=(1.0, 1.0, 10.0), max_iter=100, tolerance=1e-4, 
                 alpha_u=0.7, alpha_p=0.3, inlet_mass_flow_rate=None):
        # Fluid properties
        self.rho = density       # Fluid density [kg/m³]
        self.mu = viscosity      # Dynamic viscosity [Pa·s]
        self.nu = self.mu / self.rho  # Kinematic viscosity
        
        # Pipe dimensions (width, height, length)
        self.Lx, self.Ly, self.Lz = dimensions
        
        # Grid resolution
        self.dx = resolution
        self.dy = resolution
        self.dz = resolution
        
        # Grid dimensions
        self.nx = int(self.Lx / self.dx)
        self.ny = int(self.Ly / self.dy)
        self.nz = int(self.Lz / self.dz)
        
        # Create grid coordinates
        self.x = np.linspace(self.dx/2, self.Lx - self.dx/2, self.nx)
        self.y = np.linspace(self.dy/2, self.Ly - self.dy/2, self.ny)
        self.z = np.linspace(self.dz/2, self.Lz - self.dz/2, self.nz)
        
        # Initialize fields
        self.u = np.zeros((self.nx, self.ny, self.nz))  # x-velocity
        self.v = np.zeros((self.nx, self.ny, self.nz))  # y-velocity
        self.w = np.zeros((self.nx, self.ny, self.nz))  # z-velocity
        self.p = np.zeros((self.nx, self.ny, self.nz))  # pressure
        
        # Solver parameters
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.alpha_u = alpha_u  # Under-relaxation for velocity
        self.alpha_p = alpha_p  # Under-relaxation for pressure
        
        # Residual tracking
        self.w_residuals = []
        self.p_residuals = []
        
        # Calculate inlet velocity based on mass flow rate or provided velocity
        if inlet_mass_flow_rate is not None:
            # Calculate average velocity from mass flow rate
            area = self.Lx * self.Ly  # Cross-sectional area of square pipe
            self.V_avg = inlet_mass_flow_rate / (self.rho * area)  # Average velocity [m/s]
            self.V = self.V_avg  # For consistency with existing code
            print(f"Using mass flow rate: {inlet_mass_flow_rate} kg/s")
            print(f"Calculated average inlet velocity: {self.V_avg:.4f} m/s")
        else:
            self.V = velocity  # Use provided velocity as average velocity
            self.V_avg = velocity
            print(f"Using inlet velocity: {velocity} m/s")
        
        # Apply initial conditions
        self.apply_initial_conditions()
    
    def apply_initial_conditions(self):
        """Set initial velocity profile"""
        # Parabolic velocity profile at inlet (z=0)
        for i in range(self.nx):
            for j in range(self.ny):
                # Distance from pipe center
                y_dist = abs(self.y[j] - self.Ly/2)
                x_dist = abs(self.x[i] - self.Lx/2)
                r = np.sqrt(x_dist**2 + y_dist**2)
                R = min(self.Lx, self.Ly)/2  # Pipe radius
                
                # Parabolic profile: w = 2*V_avg * (1 - (r/R)^2)
                if r <= R:
                    self.w[i, j, 0] = 2 * self.V_avg * (1 - (r/R)**2)
                else:
                    self.w[i, j, 0] = 0  # No-slip at walls
    
    def solve_momentum_z(self):
        """Solve z-momentum equation"""
        n = self.nx * self.ny * self.nz
        A = lil_matrix((n, n))
        b = np.zeros(n)
        
        for i in range(1, self.nx-1):
            for j in range(1, self.ny-1):
                for k in range(1, self.nz-1):
                    idx = i * self.ny * self.nz + j * self.nz + k
                    
                    # Convection terms (upwind scheme)
                    F_e = self.rho * self.w[i+1, j, k] * self.dy * self.dz
                    F_w = self.rho * self.w[i, j, k] * self.dy * self.dz
                    F_n = self.rho * self.v[i, j+1, k] * self.dx * self.dz
                    F_s = self.rho * self.v[i, j, k] * self.dx * self.dz
                    F_t = self.rho * self.w[i, j, k+1] * self.dx * self.dy
                    F_b = self.rho * self.w[i, j, k] * self.dx * self.dy
                    
                    aE = max(-F_e, 0) + self.mu * self.dy * self.dz / self.dx
                    aW = max(F_w, 0) + self.mu * self.dy * self.dz / self.dx
                    aN = max(-F_n, 0) + self.mu * self.dx * self.dz / self.dy
                    aS = max(F_s, 0) + self.mu * self.dx * self.dz / self.dy
                    aT = max(-F_t, 0) + self.mu * self.dx * self.dy / self.dz
                    aB = max(F_b, 0) + self.mu * self.dx * self.dy / self.dz
                    
                    # Pressure gradient term
                    dp_dz = (self.p[i, j, k-1] - self.p[i, j, k]) / self.dz
                    source = dp_dz * self.dx * self.dy
                    
                    # Central coefficient
                    aP = aE + aW + aN + aS + aT + aB + (F_e - F_w + F_n - F_s + F_t - F_b)
                    
                    # Under-relaxation
                    aP /= self.alpha_u
                    
                    # Set coefficients
                    A[idx, idx] = aP
                    if i < self.nx-2:
                        A[idx, idx + self.ny*self.nz] = -aE
                    if i > 0:
                        A[idx, idx - self.ny*self.nz] = -aW
                    if j < self.ny-2:
                        A[idx, idx + self.nz] = -aN
                    if j > 0:
                        A[idx, idx - self.nz] = -aS
                    if k < self.nz-2:
                        A[idx, idx + 1] = -aT
                    if k > 0:
                        A[idx, idx - 1] = -aB
                    
                    b[idx] = source
        
        # Apply boundary conditions
        # Inlet (k=0): Fixed velocity
        for i in range(self.nx):
            for j in range(self.ny):
                idx = i * self.ny * self.nz + j * self.nz
                A[idx, :] = 0
                A[idx, idx] = 1
                b[idx] = self.w[i, j, 0]
        
        # Outlet (k=nz-1): Zero gradient
        for i in range(self.nx):
            for j in range(self.ny):
                idx = i * self.ny * self.nz + j * self.nz + (self.nz-1)
                A[idx, :] = 0
                A[idx, idx] = 1
                A[idx, idx-1] = -1
                b[idx] = 0
        
        # Walls: No-slip
        # x=0 and x=nx-1 walls
        for j in range(self.ny):
            for k in range(self.nz):
                idx = 0 * self.ny * self.nz + j * self.nz + k
                A[idx, :] = 0
                A[idx, idx] = 1
                b[idx] = 0
                
                idx = (self.nx-1) * self.ny * self.nz + j * self.nz + k
                A[idx, :] = 0
                A[idx, idx] = 1
                b[idx] = 0
        
        # y=0 and y=ny-1 walls
        for i in range(self.nx):
            for k in range(self.nz):
                idx = i * self.ny * self.nz + 0 * self.nz + k
                A[idx, :] = 0
                A[idx, idx] = 1
                b[idx] = 0
                
                idx = i * self.ny * self.nz + (self.ny-1) * self.nz + k
                A[idx, :] = 0
                A[idx, idx] = 1
                b[idx] = 0
        
        # Solve system
        A = A.tocsr()
        w_flat = spsolve(A, b)
        self.w = w_flat.reshape((self.nx, self.ny, self.nz))
    
    def solve_pressure_correction(self):
        """Solve pressure correction equation"""
        n = self.nx * self.ny * self.nz
        A = lil_matrix((n, n))
        b = np.zeros(n)
        
        for i in range(1, self.nx-1):
            for j in range(1, self.ny-1):
                for k in range(1, self.nz-1):
                    idx = i * self.ny * self.nz + j * self.nz + k
                    
                    # Coefficients from momentum equations
                    aE = self.rho * self.dy * self.dz / (self.mu * self.dy * self.dz / self.dx)
                    aW = self.rho * self.dy * self.dz / (self.mu * self.dy * self.dz / self.dx)
                    aN = self.rho * self.dx * self.dz / (self.mu * self.dx * self.dz / self.dy)
                    aS = self.rho * self.dx * self.dz / (self.mu * self.dx * self.dz / self.dy)
                    aT = self.rho * self.dx * self.dy / (self.mu * self.dx * self.dy / self.dz)
                    aB = self.rho * self.dx * self.dy / (self.mu * self.dx * self.dy / self.dz)
                    
                    # Central coefficient
                    aP = aE + aW + aN + aS + aT + aB
                    
                    # Mass imbalance (source term)
                    div = ((self.w[i, j, k+1] - self.w[i, j, k]) / self.dz)
                    b[idx] = -div * self.dx * self.dy * self.dz
                    
                    # Set coefficients
                    A[idx, idx] = aP
                    if i < self.nx-2:
                        A[idx, idx + self.ny*self.nz] = -aE
                    if i > 0:
                        A[idx, idx - self.ny*self.nz] = -aW
                    if j < self.ny-2:
                        A[idx, idx + self.nz] = -aN
                    if j > 0:
                        A[idx, idx - self.nz] = -aS
                    if k < self.nz-2:
                        A[idx, idx + 1] = -aT
                    if k > 0:
                        A[idx, idx - 1] = -aB
        
        # Boundary conditions for pressure correction
        # Inlet: Zero gradient
        for i in range(self.nx):
            for j in range(self.ny):
                idx = i * self.ny * self.nz + j * self.nz
                A[idx, :] = 0
                A[idx, idx] = 1
                A[idx, idx+1] = -1
                b[idx] = 0
        
        # Outlet: Fixed pressure (p' = 0)
        for i in range(self.nx):
            for j in range(self.ny):
                idx = i * self.ny * self.nz + j * self.nz + (self.nz-1)
                A[idx, :] = 0
                A[idx, idx] = 1
                b[idx] = 0
        
        # Walls: Zero gradient
        # x=0 and x=nx-1 walls
        for j in range(self.ny):
            for k in range(self.nz):
                idx = 0 * self.ny * self.nz + j * self.nz + k
                A[idx, :] = 0
                A[idx, idx] = 1
                A[idx, idx + self.ny*self.nz] = -1
                b[idx] = 0
                
                idx = (self.nx-1) * self.ny * self.nz + j * self.nz + k
                A[idx, :] = 0
                A[idx, idx] = 1
                A[idx, idx - self.ny*self.nz] = -1
                b[idx] = 0
        
        # y=0 and y=ny-1 walls
        for i in range(self.nx):
            for k in range(self.nz):
                idx = i * self.ny * self.nz + 0 * self.nz + k
                A[idx, :] = 0
                A[idx, idx] = 1
                A[idx, idx + self.nz] = -1
                b[idx] = 0
                
                idx = i * self.ny * self.nz + (self.ny-1) * self.nz + k
                A[idx, :] = 0
                A[idx, idx] = 1
                A[idx, idx - self.nz] = -1
                b[idx] = 0
        
        # Solve system
        A = A.tocsr()
        p_prime = spsolve(A, b)
        p_prime = p_prime.reshape((self.nx, self.ny, self.nz))
        
        # Correct pressure and velocities
        self.p += self.alpha_p * p_prime
        
        # Correct w-velocity
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(1, self.nz):
                    if k < self.nz-1:
                        dp_dz = (p_prime[i, j, k] - p_prime[i, j, k-1]) / self.dz
                        self.w[i, j, k] -= self.dx * self.dy * dp_dz / (self.mu * self.dx * self.dy / self.dz)
    
    def plot_residuals(self, iteration, output_dir):
        """Plot and save residuals for current iteration"""
        plt.figure(figsize=(15, 12))
        
        # Plot velocity residuals
        plt.subplot(2, 1, 1)
        plt.semilogy(range(1, iteration+2), self.w_residuals, 'b-o', label='Velocity')
        plt.xlabel('Iteration')
        plt.ylabel('Residual')
        plt.title('Velocity Residuals')
        plt.grid(True)
        plt.legend()
        
        # Plot pressure residuals
        plt.subplot(2, 1, 2)
        plt.semilogy(range(1, iteration+2), self.p_residuals, 'r-s', label='Pressure')
        plt.xlabel('Iteration')
        plt.ylabel('Residual')
        plt.title('Pressure Residuals')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/residuals.png", dpi=100)
        plt.close()
    
    def solve(self, args):
        """Main solution loop using SIMPLE algorithm"""
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        for iteration in range(self.max_iter):
            # Store previous values for convergence check
            w_old = self.w.copy()
            p_old = self.p.copy()
            
            # Solve momentum equations
            self.solve_momentum_z()
            
            # Solve pressure correction
            self.solve_pressure_correction()
            
            # Check convergence
            w_res = np.linalg.norm(self.w - w_old) / np.linalg.norm(w_old)
            p_res = np.linalg.norm(self.p - p_old) / np.linalg.norm(p_old)
            
            # Store residuals
            self.w_residuals.append(w_res)
            self.p_residuals.append(p_res)
            
            print(f"Iteration {iteration+1}: w_res = {w_res:.2e}, p_res = {p_res:.2e}")
            
            # Plot and save residuals
            self.plot_residuals(iteration, args.output_dir)
    
            # Save results
            vtk_path = os.path.join(args.output_dir, args.vtk)
            png_path = os.path.join(args.output_dir, args.png)
            self.save_vtk(vtk_path)
            self.visualize_matplotlib(png_path)
            
            if w_res < self.tolerance and p_res < self.tolerance:
                print(f"Converged after {iteration+1} iterations")
                break
        
        # Save final residual plot
        self.plot_residuals(iteration, args.output_dir)
    
        # Save results
        vtk_path = os.path.join(args.output_dir, args.vtk)
        png_path = os.path.join(args.output_dir, args.png)
        self.save_vtk(vtk_path)
        self.visualize_matplotlib(png_path) 
    
    def save_vtk(self, filename="pipe_flow.vtk"):
        """Save results to VTK file for visualization"""
        # Create VTK grid
        grid = vtk.vtkStructuredGrid()
        grid.SetDimensions(self.nx, self.ny, self.nz)
        
        # Create points
        points = vtk.vtkPoints()
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    points.InsertNextPoint(self.x[i], self.y[j], self.z[k])
        grid.SetPoints(points)
        
        # Add velocity data
        velocity = numpy_support.numpy_to_vtk(
            np.column_stack([
                self.u.flatten(),
                self.v.flatten(),
                self.w.flatten()
            ])
        )
        velocity.SetName("Velocity")
        grid.GetPointData().SetVectors(velocity)
        
        # Add pressure data
        pressure = numpy_support.numpy_to_vtk(self.p.flatten())
        pressure.SetName("Pressure")
        grid.GetPointData().SetScalars(pressure)
        
        # Write to file
        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetFileName(filename)
        writer.SetInputData(grid)
        writer.Write()
        print(f"Results saved to {filename}")
    
    def visualize_matplotlib(self, filename="pipe_flow.png"):
        """Create 3D visualization with matplotlib"""
        fig = plt.figure(figsize=(15, 10))
        
        # Create 3D grid
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        
        # Plot velocity magnitude at mid-plane
        ax1 = fig.add_subplot(221, projection='3d')
        mid_y = self.ny // 2
        vel_mag = np.sqrt(self.u[:, :, :]**2 + self.v[:, :, :]**2 + self.w[:, :, :]**2)
        ax1.scatter(X[:, :, :], Y[:, :, :], Z[:, :, :], c=vel_mag, cmap='viridis')
        ax1.set_title('Velocity Magnitude')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        
        # Plot pressure distribution
        ax2 = fig.add_subplot(222, projection='3d')
        ax2.scatter(X[:, :, :], Y[:, :, :], Z[:, :, :], c=self.p[:, :, :], cmap='coolwarm')
        ax2.set_title('Pressure Distribution')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Z (m)')
        
        # Plot velocity profile at outlet
        ax3 = fig.add_subplot(223)
        outlet_vel = self.w[:, :, -1]
        X_out, Y_out = np.meshgrid(self.x, self.y, indexing='ij')
        contour = ax3.contourf(X_out, Y_out, outlet_vel, levels=20, cmap='viridis')
        ax3.set_title('Outlet Velocity Profile (W-component)')
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        plt.colorbar(contour, ax=ax3)
        
        # Plot pressure drop along centerline
        ax4 = fig.add_subplot(224)
        centerline_p = self.p[self.nx//2, self.ny//2, :]
        ax4.plot(self.z, centerline_p, 'b-', linewidth=2)
        ax4.set_title('Pressure Drop Along Centerline')
        ax4.set_xlabel('Z (m)')
        ax4.set_ylabel('Pressure (Pa)')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"Visualization saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description='3D FVM Pipe Flow Simulation')
    parser.add_argument('--velocity', type=float, default=1.0, help='Inlet velocity [m/s]')
    parser.add_argument('--density', type=float, default=1.0, help='Fluid density [kg/m³]')
    parser.add_argument('--viscosity', type=float, default=0.001, help='Dynamic viscosity [Pa·s]')
    parser.add_argument('--resolution', type=float, default=0.1, help='Grid resolution [m]')
    parser.add_argument('--dimension', type=str, default='1,1,10', 
                        help='Pipe dimensions (width,height,length) in meters, comma-separated')
    parser.add_argument('--inlet_mass_flow_rate', type=float, default=None, 
                        help='Inlet mass flow rate [kg/s] (overrides velocity)')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--vtk', type=str, default='pipe_flow.vtk', help='VTK output filename')
    parser.add_argument('--png', type=str, default='pipe_flow.png', help='PNG output filename')
    
    args = parser.parse_args()
    
    # Parse dimensions
    dimensions = tuple(map(float, args.dimension.split(',')))
    if len(dimensions) != 3:
        raise ValueError("Dimensions must be three comma-separated values (width,height,length)")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create and run simulation
    print(f"Starting 3D Pipe Flow Simulation:")
    print(f"  Density: {args.density} kg/m³")
    print(f"  Viscosity: {args.viscosity} Pa·s")
    print(f"  Resolution: {args.resolution} m")
    print(f"  Dimensions: {dimensions[0]}x{dimensions[1]}x{dimensions[2]} m")
    print(f"  Grid: {int(dimensions[0]/args.resolution)}x{int(dimensions[1]/args.resolution)}x{int(dimensions[2]/args.resolution)} cells")
    print(f"  Output directory: {args.output_dir}")
    
    solver = FVM3DPipeFlow(
        velocity=args.velocity,
        density=args.density,
        viscosity=args.viscosity,
        resolution=args.resolution,
        dimensions=dimensions,
        inlet_mass_flow_rate=args.inlet_mass_flow_rate
    )
    
    # Run simulation
    solver.solve(args)
    
    print("\nSimulation completed successfully!")
    vtk_path = os.path.join(args.output_dir, args.vtk)
    png_path = os.path.join(args.output_dir, args.png)
    print(f"  VTK file: {vtk_path} (open with ParaView or VisIt)")
    print(f"  PNG file: {png_path}")
    print(f"  Residual plots: {args.output_dir}/residuals.png")

if __name__ == "__main__":
    main()