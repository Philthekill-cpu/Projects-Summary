"""
===============================================================================
FERMI-PASTA-ULAM-TSINGOU (FPUT) SIMULATION
===============================================================================

CAPABILITIES:
- Solves the FPUT nonlinear lattice problem using 4th-order Runge-Kutta
- Supports quadratic nonlinearity (alpha parameter)
- Multiple initial condition profiles: sine, half-sine, parabola
- Modal (Fourier) analysis of energy distribution across modes
- Visualization of string displacement and mode energy evolution
- Tolerance modeling for Task E (suppression of recurrence)

LIMITATIONS:
- Fixed boundary conditions only (x_0 = x_{N-1} = 0)
- Quadratic nonlinearity only (no cubic beta term)
- No adaptive time-stepping (fixed dt for simplicity and speed)

USAGE:
1. Edit the INPUT PARAMETERS section below
2. Run: python fput_simulation.py

Author: Ping Fan Teng
Date: November 2022
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional

# =============================================================================
# INPUT PARAMETERS - USER INTERFACE (Task C)
# =============================================================================
# Modify these parameters to configure your simulation

# Physical parameters
L = 1.0              # Length of the string
N = 65               # Number of oscillators (including boundaries)
kappa = 0.1          # Young's modulus
rho = 400.0          # Mass per unit length
alpha = 0.25         # Nonlinear coefficient (0 for linear case)

# Simulation parameters
T_final = 12000.0    # Total simulation time
dt = 0.1             # Time step (adjust for accuracy vs speed)

# Initial condition type:
# 1: Single sine wave (mode 1)
# 2: Half sine wave
# 3: Parabola
# 4: Custom mode (specify mode_number below)
init_type = 1
mode_number = 1      # Used if init_type = 4
amplitude = 1.0      # Amplitude of initial displacement
init_velocity = 0.0  # Initial velocity amplitude (0 for stationary start)

# Output settings
output_times = [0, 1000, 3000, 6000, 9000, 12000]  # Times to plot displacement
n_modes_plot = 4     # Number of Fourier modes to track

# Task E: Tolerance settings
use_tolerances = False    # Set True to enable tolerance modeling
tolerance_percent = 1.0   # Tolerance percentage (e.g., 1.0 for ±1%)

# =============================================================================
# DERIVED QUANTITIES
# =============================================================================

def compute_derived_params(L: float, N: int, kappa: float, rho: float) -> Tuple[float, float, float]:
    """
    Compute derived physical parameters from input.
    
    Parameters:
        L: String length
        N: Number of oscillators
        kappa: Young's modulus
        rho: Mass per unit length
    
    Returns:
        h: Lattice spacing
        m: Mass of each oscillator
        k: Spring constant
    """
    h = L / (N - 1)      # Lattice spacing
    m = rho * h          # Mass per oscillator
    k = kappa / h        # Spring constant
    return h, m, k

# =============================================================================
# INITIAL CONDITIONS (Task C)
# =============================================================================

def initialize_displacement(N: int, init_type: int, amplitude: float, 
                           mode_number: int = 1) -> np.ndarray:
    """
    Generate initial displacement profile for the string.
    
    Parameters:
        N: Number of oscillators
        init_type: Type of initial condition (1-4)
        amplitude: Maximum displacement amplitude
        mode_number: Mode number for type 4
    
    Returns:
        x: Array of initial displacements (length N)
    """
    x = np.zeros(N)
    # Interior points only (boundaries fixed at 0)
    j = np.arange(1, N - 1)
    
    if init_type == 1:
        # Single sine wave (first mode)
        x[1:N-1] = amplitude * np.sin(np.pi * j / (N - 1))
    elif init_type == 2:
        # Half sine wave (sharper peak)
        x[1:N-1] = amplitude * np.sin(np.pi * j / (2 * (N - 1)))
    elif init_type == 3:
        # Parabola
        x[1:N-1] = amplitude * 4 * j * (N - 1 - j) / (N - 1)**2
    elif init_type == 4:
        # Custom mode
        x[1:N-1] = amplitude * np.sin(mode_number * np.pi * j / (N - 1))
    else:
        raise ValueError(f"Unknown init_type: {init_type}")
    
    return x

def initialize_velocity(N: int, init_velocity: float) -> np.ndarray:
    """
    Generate initial velocity profile.
    
    Parameters:
        N: Number of oscillators
        init_velocity: Velocity amplitude
    
    Returns:
        v: Array of initial velocities (length N)
    """
    v = np.zeros(N)
    if init_velocity != 0:
        j = np.arange(1, N - 1)
        v[1:N-1] = init_velocity * np.sin(np.pi * j / (N - 1))
    return v

# =============================================================================
# EQUATIONS OF MOTION (Task A)
# =============================================================================

def compute_acceleration(x: np.ndarray, k: float, m: float, alpha: float,
                        tolerances: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute acceleration for each oscillator based on FPUT equations.
    
    The governing equation is:
    m * d²xᵢ/dt² = k(xᵢ₊₁ + xᵢ₋₁ - 2xᵢ)[1 + α(xᵢ₊₁ - xᵢ₋₁)]
    
    Parameters:
        x: Current displacements (length N)
        k: Spring constant
        m: Oscillator mass
        alpha: Nonlinearity coefficient
        tolerances: Optional tolerance factors for each oscillator (Task E)
    
    Returns:
        a: Accelerations for interior points (length N)
    """
    N = len(x)
    a = np.zeros(N)
    
    # Interior points (boundaries have zero acceleration)
    for i in range(1, N - 1):
        # Hooke's law term
        linear_term = x[i+1] + x[i-1] - 2*x[i]
        # Nonlinear correction
        nonlinear_factor = 1 + alpha * (x[i+1] - x[i-1])
        
        if tolerances is not None:
            # Task E: Include tolerance factors
            # Using the form from the Nelson et al. paper
            t_ip1 = tolerances[i+1] if i+1 < N else 1.0
            t_i = tolerances[i]
            t_im1 = tolerances[i-1] if i-1 >= 0 else 1.0
            
            linear_term = t_ip1*x[i+1] + t_im1*x[i-1] - 2*t_i*x[i]
            nonlinear_correction = alpha * ((t_ip1*x[i+1] - t_i*x[i])**2 
                                           - (t_i*x[i] - t_im1*x[i-1])**2)
            a[i] = (k / m) * (linear_term + nonlinear_correction)
        else:
            a[i] = (k / m) * linear_term * nonlinear_factor
    
    return a

def fput_derivatives(state: np.ndarray, k: float, m: float, alpha: float,
                    tolerances: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute time derivatives for the FPUT system.
    
    Converts second-order ODE to first-order system:
    dx/dt = v
    dv/dt = a(x)
    
    Parameters:
        state: Combined state vector [x, v] of length 2N
        k, m, alpha: Physical parameters
        tolerances: Optional tolerance factors
    
    Returns:
        derivatives: [dx/dt, dv/dt] of length 2N
    """
    N = len(state) // 2
    x = state[:N]
    v = state[N:]
    
    dxdt = v.copy()
    dvdt = compute_acceleration(x, k, m, alpha, tolerances)
    
    # Enforce boundary conditions
    dxdt[0] = 0
    dxdt[N-1] = 0
    dvdt[0] = 0
    dvdt[N-1] = 0
    
    return np.concatenate([dxdt, dvdt])

# =============================================================================
# NUMERICAL INTEGRATION: 4TH-ORDER RUNGE-KUTTA (Task A)
# =============================================================================

def rk4_step(state: np.ndarray, dt: float, k: float, m: float, alpha: float,
            tolerances: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Perform one step of 4th-order Runge-Kutta integration.
    
    Parameters:
        state: Current state [x, v]
        dt: Time step
        k, m, alpha: Physical parameters
        tolerances: Optional tolerance factors
    
    Returns:
        new_state: State after time dt
    """
    k1 = fput_derivatives(state, k, m, alpha, tolerances)
    k2 = fput_derivatives(state + 0.5*dt*k1, k, m, alpha, tolerances)
    k3 = fput_derivatives(state + 0.5*dt*k2, k, m, alpha, tolerances)
    k4 = fput_derivatives(state + dt*k3, k, m, alpha, tolerances)
    
    return state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

def integrate_fput(x0: np.ndarray, v0: np.ndarray, T_final: float, dt: float,
                  k: float, m: float, alpha: float,
                  output_times: List[float],
                  tolerances: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Integrate the FPUT system over time.
    
    Parameters:
        x0, v0: Initial displacement and velocity
        T_final: Total simulation time
        dt: Time step
        k, m, alpha: Physical parameters
        output_times: Times at which to save full displacement profile
        tolerances: Optional tolerance factors
    
    Returns:
        t_history: Array of all time points
        x_history: Displacement at each time step (for Fourier analysis)
        x_snapshots: Displacements at requested output_times
    """
    N = len(x0)
    n_steps = int(T_final / dt)
    
    # Storage for full history (for Fourier analysis)
    save_interval = max(1, n_steps // 10000)  # Limit memory usage
    t_history = []
    x_history = []
    
    # Storage for snapshots at specific times
    output_times_set = set(output_times)
    x_snapshots = {}
    
    # Initial state
    state = np.concatenate([x0, v0])
    
    print(f"Starting integration: {n_steps} steps, dt={dt}")
    
    for step in range(n_steps + 1):
        t = step * dt
        x = state[:N]
        
        # Save snapshot if at output time
        for t_out in output_times:
            if abs(t - t_out) < dt/2 and t_out not in x_snapshots:
                x_snapshots[t_out] = x.copy()
        
        # Save for history (subsampled)
        if step % save_interval == 0:
            t_history.append(t)
            x_history.append(x.copy())
        
        # Progress indicator
        if step % (n_steps // 10) == 0:
            print(f"  Progress: {100*step/n_steps:.0f}%")
        
        # Advance one time step
        if step < n_steps:
            state = rk4_step(state, dt, k, m, alpha, tolerances)
    
    return np.array(t_history), np.array(x_history), x_snapshots

# =============================================================================
# FOURIER / MODAL ANALYSIS (Task A)
# =============================================================================

def compute_mode_energies(x_history: np.ndarray, n_modes: int) -> np.ndarray:
    """
    Compute the energy in each Fourier mode over time.
    
    For a string with fixed ends, the natural modes are sine functions.
    We compute the projection onto each mode and track the amplitude squared
    (proportional to energy).
    
    Parameters:
        x_history: Array of shape (n_times, N) with displacement history
        n_modes: Number of modes to compute
    
    Returns:
        mode_energies: Array of shape (n_times, n_modes) with normalized energies
    """
    n_times, N = x_history.shape
    mode_energies = np.zeros((n_times, n_modes))
    
    # Interior points only
    j = np.arange(1, N - 1)
    
    for mode in range(n_modes):
        # Mode shape: sin((mode+1) * pi * j / (N-1))
        mode_shape = np.sin((mode + 1) * np.pi * j / (N - 1))
        mode_shape /= np.linalg.norm(mode_shape)  # Normalize
        
        for t_idx in range(n_times):
            # Project displacement onto mode
            projection = np.dot(x_history[t_idx, 1:N-1], mode_shape)
            mode_energies[t_idx, mode] = projection**2
    
    # Normalize each time step so energies sum to 1
    total_energy = np.sum(mode_energies, axis=1, keepdims=True)
    total_energy = np.where(total_energy > 0, total_energy, 1)  # Avoid division by zero
    mode_energies /= total_energy
    
    return mode_energies

def compute_fft_coefficients(x_history: np.ndarray, n_modes: int) -> np.ndarray:
    """
    Compute FFT-based mode amplitudes (alternative to direct projection).
    
    Parameters:
        x_history: Displacement history
        n_modes: Number of modes to track
    
    Returns:
        fft_amplitudes: Normalized FFT amplitudes for first n_modes
    """
    n_times, N = x_history.shape
    fft_amplitudes = np.zeros((n_times, n_modes))
    
    for t_idx in range(n_times):
        # FFT of interior points
        fft = np.fft.fft(x_history[t_idx, 1:N-1])
        amplitudes = np.abs(fft[:n_modes])
        
        # Normalize
        total = np.sum(np.abs(fft))
        if total > 0:
            fft_amplitudes[t_idx] = amplitudes / total
    
    return fft_amplitudes

# =============================================================================
# TOLERANCE GENERATION (Task E)
# =============================================================================

def generate_tolerances(N: int, tolerance_percent: float, seed: int = None) -> np.ndarray:
    """
    Generate random tolerance factors for each oscillator.
    
    Following the Nelson et al. paper, tolerances are drawn from a Gaussian
    distribution with mean 1 and standard deviation sigma = (1/3) * 0.01 * tau
    where tau is the tolerance percentage.
    
    Parameters:
        N: Number of oscillators
        tolerance_percent: Tolerance as percentage (e.g., 1.0 for ±1%)
        seed: Random seed for reproducibility
    
    Returns:
        tolerances: Array of tolerance factors
    """
    if seed is not None:
        np.random.seed(seed)
    
    sigma = (1/3) * 0.01 * tolerance_percent
    tolerances = np.random.normal(1.0, sigma, N)
    
    # Clip to ±3σ range
    lower = 1 - 0.01 * tolerance_percent
    upper = 1 + 0.01 * tolerance_percent
    tolerances = np.clip(tolerances, lower, upper)
    
    return tolerances

# =============================================================================
# VISUALIZATION (Tasks A, B, D)
# =============================================================================

def plot_displacement_snapshots(x_snapshots: dict, N: int, L: float, 
                                title_suffix: str = ""):
    """
    Plot string displacement at multiple time instants.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    positions = np.linspace(0, L, N)
    times = sorted(x_snapshots.keys())
    
    for idx, t in enumerate(times[:6]):
        ax = axes[idx]
        ax.plot(positions, x_snapshots[t], 'b-', linewidth=1.5)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Position along string')
        ax.set_ylabel('Displacement')
        ax.set_title(f't = {t:.0f}')
        ax.set_ylim([-1.5, 1.5])
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'String Displacement at Selected Times {title_suffix}', fontsize=14)
    plt.tight_layout()
    return fig

def plot_mode_energies(t_history: np.ndarray, mode_energies: np.ndarray,
                       n_modes: int, title_suffix: str = ""):
    """
    Plot time evolution of energy in first several modes.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    labels = ['1st mode', '2nd mode', '3rd mode', '4th mode', '5th mode', '6th mode']
    
    for mode in range(min(n_modes, len(colors))):
        ax.plot(t_history, mode_energies[:, mode], 
                color=colors[mode], label=labels[mode], linewidth=1)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Normalized Mode Energy')
    ax.set_title(f'Evolution of Fourier Mode Energies {title_suffix}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, t_history[-1]])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    return fig

def plot_recurrence_comparison(results_dict: dict):
    """
    Compare mode energy evolution for different parameter values (Task D).
    """
    fig, axes = plt.subplots(len(results_dict), 1, figsize=(12, 4*len(results_dict)))
    if len(results_dict) == 1:
        axes = [axes]
    
    for idx, (label, (t_hist, mode_en)) in enumerate(results_dict.items()):
        ax = axes[idx]
        colors = ['blue', 'red', 'green', 'purple']
        for mode in range(min(4, mode_en.shape[1])):
            ax.plot(t_hist, mode_en[:, mode], color=colors[mode], 
                   label=f'Mode {mode+1}', linewidth=1)
        ax.set_xlabel('Time')
        ax.set_ylabel('Normalized Energy')
        ax.set_title(label)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# =============================================================================
# MAIN SIMULATION FUNCTION
# =============================================================================

def run_simulation(L: float, N: int, kappa: float, rho: float, alpha: float,
                  T_final: float, dt: float, init_type: int, amplitude: float,
                  init_velocity: float, output_times: List[float], n_modes: int,
                  use_tolerances: bool = False, tolerance_percent: float = 0.0,
                  tolerance_seed: int = None, plot_results: bool = True,
                  title_suffix: str = "") -> Tuple:
    """
    Run a complete FPUT simulation with visualization.
    
    Returns:
        t_history, mode_energies, x_snapshots
    """
    # Compute derived parameters
    h, m, k = compute_derived_params(L, N, kappa, rho)
    print(f"Derived parameters: h={h:.6f}, m={m:.4f}, k={k:.4f}")
    
    # Generate initial conditions
    x0 = initialize_displacement(N, init_type, amplitude)
    v0 = initialize_velocity(N, init_velocity)
    
    # Generate tolerances if needed
    tolerances = None
    if use_tolerances:
        tolerances = generate_tolerances(N, tolerance_percent, tolerance_seed)
        print(f"Tolerances generated: mean={np.mean(tolerances):.4f}, std={np.std(tolerances):.4f}")
    
    # Run integration
    t_history, x_history, x_snapshots = integrate_fput(
        x0, v0, T_final, dt, k, m, alpha, output_times, tolerances
    )
    
    # Compute mode energies
    mode_energies = compute_mode_energies(x_history, n_modes)
    
    # Plot results
    if plot_results:
        fig1 = plot_displacement_snapshots(x_snapshots, N, L, title_suffix)
        fig2 = plot_mode_energies(t_history, mode_energies, n_modes, title_suffix)
        plt.show()
    
    return t_history, mode_energies, x_snapshots

# =============================================================================
# TASK EXECUTION
# =============================================================================

def task_B_validation():
    """
    Task B: Validation with specified parameters.
    Run both linear (α=0) and nonlinear (α=0.25) cases.
    """
    print("="*60)
    print("TASK B: VALIDATION")
    print("="*60)
    
    # Common parameters from assignment
    params = {
        'L': 1.0, 'N': 65, 'kappa': 0.1, 'rho': 400.0,
        'T_final': 12000.0, 'dt': 0.1, 'init_type': 1, 'amplitude': 1.0,
        'init_velocity': 0.0, 'output_times': [0, 1000, 3000, 6000, 9000, 12000],
        'n_modes': 4
    }
    
    # Linear case (α = 0)
    print("\n--- Linear case (α = 0) ---")
    t1, me1, xs1 = run_simulation(**params, alpha=0.0, title_suffix="(α = 0)")
    
    # Nonlinear case (α = 0.25)
    print("\n--- Nonlinear case (α = 0.25) ---")
    t2, me2, xs2 = run_simulation(**params, alpha=0.25, title_suffix="(α = 0.25)")
    
    return (t1, me1), (t2, me2)

def task_D_study():
    """
    Task D: Study the FPUT phenomenon by varying parameters.
    """
    print("="*60)
    print("TASK D: PARAMETER STUDY")
    print("="*60)
    
    base_params = {
        'L': 1.0, 'N': 65, 'kappa': 0.1, 'rho': 400.0,
        'T_final': 12000.0, 'dt': 0.1, 'init_type': 1,
        'init_velocity': 0.0, 'output_times': [0, 6000, 12000],
        'n_modes': 4, 'plot_results': False
    }
    
    results = {}
    
    # Study 1: Effect of initial amplitude
    print("\n--- Study 1: Effect of initial amplitude ---")
    for amp in [0.5, 1.0, 2.0]:
        label = f"Amplitude = {amp}"
        print(f"Running: {label}")
        t, me, _ = run_simulation(**base_params, alpha=0.25, amplitude=amp)
        results[label] = (t, me)
    
    fig1 = plot_recurrence_comparison(results)
    plt.suptitle("Effect of Initial Amplitude on Recurrence")
    plt.savefig("task_d_amplitude_study.png", dpi=150)
    plt.show()
    
    # Study 2: Effect of α
    results = {}
    print("\n--- Study 2: Effect of α ---")
    for alpha_val in [0.1, 0.25, 0.5]:
        label = f"α = {alpha_val}"
        print(f"Running: {label}")
        t, me, _ = run_simulation(**base_params, alpha=alpha_val, amplitude=1.0)
        results[label] = (t, me)
    
    fig2 = plot_recurrence_comparison(results)
    plt.suptitle("Effect of Nonlinearity Parameter α")
    plt.savefig("task_d_alpha_study.png", dpi=150)
    plt.show()

def task_E_tolerances():
    """
    Task E: Study effect of tolerances on recurrence.
    """
    print("="*60)
    print("TASK E: TOLERANCE STUDY")
    print("="*60)
    
    base_params = {
        'L': 1.0, 'N': 65, 'kappa': 0.1, 'rho': 400.0, 'alpha': 0.25,
        'T_final': 12000.0, 'dt': 0.1, 'init_type': 1, 'amplitude': 1.0,
        'init_velocity': 0.0, 'output_times': [0, 6000, 12000],
        'n_modes': 4, 'plot_results': False
    }
    
    results = {}
    
    # Ideal case (no tolerance)
    print("Running: Ideal (no tolerance)")
    t, me, _ = run_simulation(**base_params, use_tolerances=False)
    results["Ideal (no tolerance)"] = (t, me)
    
    # Various tolerance levels
    for tol in [0.1, 1.0, 5.0, 10.0]:
        label = f"Tolerance ±{tol}%"
        print(f"Running: {label}")
        t, me, _ = run_simulation(**base_params, use_tolerances=True, 
                                  tolerance_percent=tol, tolerance_seed=42)
        results[label] = (t, me)
    
    fig = plot_recurrence_comparison(results)
    plt.suptitle("Effect of Tolerances on FPUT Recurrence (Task E)")
    plt.savefig("task_e_tolerance_study.png", dpi=150)
    plt.show()

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("FERMI-PASTA-ULAM-TSINGOU SIMULATION")
    print("="*60)
    
    # Run validation (Task B)
    task_B_validation()
    
    # Uncomment below to run additional studies:
    # task_D_study()
    # task_E_tolerances()
    
    print("\nSimulation complete.")
