"""
Double Pendulum: Analytical Formulation and Numerical Solvers

Mathematical derivation of equations of motion using both Lagrangian
and Hamiltonian mechanics, plus multiple numerical integration schemes.
"""

import numpy as np
from scipy.integrate import solve_ivp


# =============================================================================
# Physical Constants & Parameters
# =============================================================================

class DoublePendulumParams:
    """Parameters for the double pendulum system."""
    def __init__(self, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81):
        self.m1 = m1  # mass of pendulum 1
        self.m2 = m2  # mass of pendulum 2
        self.l1 = l1  # length of rod 1
        self.l2 = l2  # length of rod 2
        self.g = g    # gravitational acceleration


# =============================================================================
# Lagrangian Formulation
# =============================================================================

def lagrangian(state, params):
    """
    Compute the Lagrangian L = T - V for the double pendulum.
    
    State: [theta1, theta2, omega1, omega2]
    where omega_i = d(theta_i)/dt
    
    Kinetic Energy:
        T = (1/2)*m1*l1^2*w1^2 
          + (1/2)*m2*(l1^2*w1^2 + l2^2*w2^2 + 2*l1*l2*w1*w2*cos(th1-th2))
    
    Potential Energy:
        V = -(m1+m2)*g*l1*cos(th1) - m2*g*l2*cos(th2)
    """
    th1, th2, w1, w2 = state
    p = params
    delta = th1 - th2
    
    T = (0.5 * p.m1 * p.l1**2 * w1**2 +
         0.5 * p.m2 * (p.l1**2 * w1**2 + p.l2**2 * w2**2 +
                       2 * p.l1 * p.l2 * w1 * w2 * np.cos(delta)))
    
    V = (-(p.m1 + p.m2) * p.g * p.l1 * np.cos(th1) -
         p.m2 * p.g * p.l2 * np.cos(th2))
    
    return T - V


def total_energy(state, params):
    """Compute H = T + V (conserved quantity)."""
    th1, th2, w1, w2 = state
    p = params
    delta = th1 - th2
    
    T = (0.5 * p.m1 * p.l1**2 * w1**2 +
         0.5 * p.m2 * (p.l1**2 * w1**2 + p.l2**2 * w2**2 +
                       2 * p.l1 * p.l2 * w1 * w2 * np.cos(delta)))
    
    V = (-(p.m1 + p.m2) * p.g * p.l1 * np.cos(th1) -
         p.m2 * p.g * p.l2 * np.cos(th2))
    
    return T + V


# =============================================================================
# Equations of Motion (Lagrangian form)
# =============================================================================

def equations_of_motion(t, state, params):
    """
    Derive accelerations from the Euler-Lagrange equations.
    
    The coupled equations are:
    
    (m1+m2)*l1*ddth1 + m2*l2*ddth2*cos(delta) + m2*l2*w2^2*sin(delta) 
        + (m1+m2)*g*sin(th1) = 0
        
    m2*l2*ddth2 + m2*l1*ddth1*cos(delta) - m2*l1*w1^2*sin(delta) 
        + m2*g*sin(th2) = 0
    
    Solving this 2x2 linear system for [ddth1, ddth2]:
    """
    th1, th2, w1, w2 = state
    p = params
    delta = th1 - th2
    
    # Mass matrix M * [ddth1, ddth2] = f
    M11 = (p.m1 + p.m2) * p.l1
    M12 = p.m2 * p.l2 * np.cos(delta)
    M21 = p.m2 * p.l1 * np.cos(delta)
    M22 = p.m2 * p.l2
    
    f1 = (-p.m2 * p.l2 * w2**2 * np.sin(delta) -
           (p.m1 + p.m2) * p.g * np.sin(th1))
    f2 = (p.m2 * p.l1 * w1**2 * np.sin(delta) -
           p.m2 * p.g * np.sin(th2))
    
    det = M11 * M22 - M12 * M21
    ddth1 = (M22 * f1 - M12 * f2) / det
    ddth2 = (M11 * f2 - M21 * f1) / det
    
    return [w1, w2, ddth1, ddth2]


# =============================================================================
# Hamiltonian Formulation
# =============================================================================

def state_to_hamiltonian(state, params):
    """
    Convert (theta1, theta2, omega1, omega2) to Hamiltonian coordinates
    (theta1, theta2, p1, p2) where p_i are conjugate momenta.
    
    p1 = (m1+m2)*l1^2*w1 + m2*l1*l2*w2*cos(delta)
    p2 = m2*l2^2*w2 + m2*l1*l2*w1*cos(delta)
    """
    th1, th2, w1, w2 = state
    p = params
    delta = th1 - th2
    
    p1 = (p.m1 + p.m2) * p.l1**2 * w1 + p.m2 * p.l1 * p.l2 * w2 * np.cos(delta)
    p2 = p.m2 * p.l2**2 * w2 + p.m2 * p.l1 * p.l2 * w1 * np.cos(delta)
    
    return [th1, th2, p1, p2]


def hamiltonian_eom(t, state_h, params):
    """
    Hamilton's equations: dq/dt = dH/dp, dp/dt = -dH/dq
    
    State: [theta1, theta2, p1, p2]
    """
    th1, th2, p1, p2 = state_h
    p = params
    delta = th1 - th2
    
    # Invert momentum to get velocities
    denom = (p.m1 + p.m2) * p.m2 * p.l1**2 * p.l2**2 - (p.m2 * p.l1 * p.l2 * np.cos(delta))**2
    
    w1 = (p.m2 * p.l2**2 * p1 - p.m2 * p.l1 * p.l2 * np.cos(delta) * p2) / denom
    w2 = ((p.m1 + p.m2) * p.l1**2 * p2 - p.m2 * p.l1 * p.l2 * np.cos(delta) * p1) / denom
    
    # Hamilton's equations
    A = p.m2 * p.l1 * p.l2 * w1 * w2 * np.sin(delta)
    
    dp1 = -A - (p.m1 + p.m2) * p.g * p.l1 * np.sin(th1)
    dp2 = A - p.m2 * p.g * p.l2 * np.sin(th2)
    
    return [w1, w2, dp1, dp2]


# =============================================================================
# Numerical Solvers
# =============================================================================

def solve_rk45(y0, t_span, dt, params):
    """Scipy's adaptive RK45 solver (reference solution)."""
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(equations_of_motion, t_span, y0, method='RK45',
                    t_eval=t_eval, args=(params,), rtol=1e-10, atol=1e-12)
    return sol.t, sol.y.T


def solve_rk4(y0, t_span, dt, params):
    """Classic 4th-order Runge-Kutta (fixed step)."""
    t = np.arange(t_span[0], t_span[1], dt)
    n = len(t)
    y = np.zeros((n, 4))
    y[0] = y0
    
    for i in range(n - 1):
        h = dt
        k1 = np.array(equations_of_motion(t[i], y[i], params))
        k2 = np.array(equations_of_motion(t[i] + h/2, y[i] + h/2*k1, params))
        k3 = np.array(equations_of_motion(t[i] + h/2, y[i] + h/2*k2, params))
        k4 = np.array(equations_of_motion(t[i] + h, y[i] + h*k3, params))
        y[i+1] = y[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return t, y


def solve_symplectic_euler(y0, t_span, dt, params):
    """
    Symplectic Euler method - preserves symplectic structure.
    Uses Hamiltonian formulation.
    """
    t = np.arange(t_span[0], t_span[1], dt)
    n = len(t)
    
    # Convert to Hamiltonian coords
    state_h = state_to_hamiltonian(y0, params)
    y_h = np.zeros((n, 4))
    y_h[0] = state_h
    
    for i in range(n - 1):
        th1, th2, p1, p2 = y_h[i]
        derivs = hamiltonian_eom(t[i], y_h[i], params)
        
        # Update momenta first (implicit in p)
        p1_new = p1 + dt * derivs[2]
        p2_new = p2 + dt * derivs[3]
        
        # Then update positions with new momenta
        derivs_new = hamiltonian_eom(t[i], [th1, th2, p1_new, p2_new], params)
        th1_new = th1 + dt * derivs_new[0]
        th2_new = th2 + dt * derivs_new[1]
        
        y_h[i+1] = [th1_new, th2_new, p1_new, p2_new]
    
    # Convert back to Lagrangian coords for comparison
    y = np.zeros((n, 4))
    for i in range(n):
        th1, th2, pp1, pp2 = y_h[i]
        delta = th1 - th2
        p = params
        denom = (p.m1 + p.m2) * p.m2 * p.l1**2 * p.l2**2 - (p.m2 * p.l1 * p.l2 * np.cos(delta))**2
        w1 = (p.m2 * p.l2**2 * pp1 - p.m2 * p.l1 * p.l2 * np.cos(delta) * pp2) / denom
        w2 = ((p.m1 + p.m2) * p.l1**2 * pp2 - p.m2 * p.l1 * p.l2 * np.cos(delta) * pp1) / denom
        y[i] = [th1, th2, w1, w2]
    
    return t, y


def solve_stormer_verlet(y0, t_span, dt, params):
    """
    Störmer-Verlet (leapfrog) integrator - 2nd order symplectic.
    Excellent long-term energy conservation.
    """
    t = np.arange(t_span[0], t_span[1], dt)
    n = len(t)
    y = np.zeros((n, 4))
    y[0] = y0
    
    for i in range(n - 1):
        th1, th2, w1, w2 = y[i]
        
        # Half-step velocity
        derivs = equations_of_motion(t[i], y[i], params)
        w1_half = w1 + 0.5 * dt * derivs[2]
        w2_half = w2 + 0.5 * dt * derivs[3]
        
        # Full-step position
        th1_new = th1 + dt * w1_half
        th2_new = th2 + dt * w2_half
        
        # Full-step velocity using new positions
        state_mid = [th1_new, th2_new, w1_half, w2_half]
        derivs_new = equations_of_motion(t[i] + dt, state_mid, params)
        w1_new = w1_half + 0.5 * dt * derivs_new[2]
        w2_new = w2_half + 0.5 * dt * derivs_new[3]
        
        y[i+1] = [th1_new, th2_new, w1_new, w2_new]
    
    return t, y


# =============================================================================
# Utility Functions
# =============================================================================

def pendulum_positions(state, params):
    """Compute (x,y) positions of both pendulum bobs."""
    th1, th2 = state[0], state[1]
    x1 = params.l1 * np.sin(th1)
    y1 = -params.l1 * np.cos(th1)
    x2 = x1 + params.l2 * np.sin(th2)
    y2 = y1 - params.l2 * np.cos(th2)
    return x1, y1, x2, y2


def compute_lyapunov(y0, params, t_span=(0, 50), dt=0.01, delta0=1e-9):
    """
    Estimate the maximal Lyapunov exponent by tracking divergence
    of nearby trajectories.
    """
    # Reference trajectory
    t, y_ref = solve_rk4(y0, t_span, dt, params)
    
    # Perturbed trajectory
    y0_pert = y0.copy()
    y0_pert[0] += delta0
    _, y_pert = solve_rk4(y0_pert, t_span, dt, params)
    
    # Compute divergence
    diff = y_ref - y_pert
    dist = np.sqrt(np.sum(diff**2, axis=1))
    dist[dist < 1e-15] = 1e-15  # avoid log(0)
    
    # Lyapunov exponent from exponential divergence
    lyap = np.log(dist / delta0) / t
    lyap[0] = 0  # avoid division by zero
    
    return t, lyap


if __name__ == "__main__":
    # Quick test
    params = DoublePendulumParams()
    y0 = [np.pi/2, np.pi/2, 0.0, 0.0]  # Both at 90 degrees
    
    t, y = solve_rk45(y0, (0, 10), 0.01, params)
    
    # Check energy conservation
    E = np.array([total_energy(y[i], params) for i in range(len(t))])
    print(f"Energy drift: {np.max(np.abs(E - E[0])):.2e}")
    print(f"Initial energy: {E[0]:.4f}")
    print(f"Final state: th1={y[-1,0]:.4f}, th2={y[-1,1]:.4f}")
