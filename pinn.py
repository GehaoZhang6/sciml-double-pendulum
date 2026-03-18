"""
Physics-Informed Neural Network (PINN) for the Double Pendulum.

Key idea: Train a neural network to satisfy both:
1. Data loss: Match observed trajectories
2. Physics loss: Satisfy the governing ODE at collocation points

The physics residual enforces the Euler-Lagrange equations without
requiring the network to explicitly learn the Lagrangian or Hamiltonian.

Reference: Raissi et al., "Physics-informed neural networks," JCP 2019
"""

import torch
import torch.nn as nn
import numpy as np


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for the double pendulum.
    
    Architecture: Maps time t -> state [theta1(t), theta2(t), omega1(t), omega2(t)]
    
    The physics residual enforces:
        d(omega_i)/dt = f_i(theta1, theta2, omega1, omega2)
    where f_i are the analytical accelerations from the EOM.
    """
    def __init__(self, hidden_dim=128, n_layers=4):
        super().__init__()
        
        # Network: t -> [theta1, theta2, omega1, omega2]
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 4))
        self.net = nn.Sequential(*layers)
        
        # Physical parameters (can be learned or fixed)
        self.m1 = 1.0
        self.m2 = 1.0
        self.l1 = 1.0
        self.l2 = 1.0
        self.g = 9.81
    
    def forward(self, t):
        """
        Predict state at time t.
        
        Args:
            t: (batch, 1) — time points
        Returns:
            state: (batch, 4) — [theta1, theta2, omega1, omega2]
        """
        return self.net(t)
    
    def physics_residual(self, t):
        """
        Compute the physics residual at collocation points.
        
        The residual measures how well the predicted trajectory satisfies
        the equations of motion. Zero residual = perfect physics.
        
        Args:
            t: (batch, 1) — collocation points (requires_grad=True)
        Returns:
            residual: (batch, 2) — [r1, r2] residuals for each EOM
        """
        t = t.requires_grad_(True)
        state = self.forward(t)
        
        th1 = state[:, 0:1]
        th2 = state[:, 1:2]
        w1 = state[:, 2:3]
        w2 = state[:, 3:4]
        
        # Compute time derivatives via autograd
        dth1_dt = torch.autograd.grad(th1.sum(), t, create_graph=True)[0]
        dth2_dt = torch.autograd.grad(th2.sum(), t, create_graph=True)[0]
        dw1_dt = torch.autograd.grad(w1.sum(), t, create_graph=True)[0]
        dw2_dt = torch.autograd.grad(w2.sum(), t, create_graph=True)[0]
        
        # Kinematic constraints: dtheta/dt = omega
        res_kin1 = dth1_dt - w1
        res_kin2 = dth2_dt - w2
        
        # Dynamic constraints: domega/dt = f(state)
        # From the analytical EOM of the double pendulum
        delta = th1 - th2
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g
        
        M11 = (m1 + m2) * l1
        M12 = m2 * l2 * torch.cos(delta)
        M21 = m2 * l1 * torch.cos(delta)
        M22 = m2 * l2
        
        f1 = -m2 * l2 * w2**2 * torch.sin(delta) - (m1 + m2) * g * torch.sin(th1)
        f2 = m2 * l1 * w1**2 * torch.sin(delta) - m2 * g * torch.sin(th2)
        
        det = M11 * M22 - M12 * M21
        ddth1_analytical = (M22 * f1 - M12 * f2) / det
        ddth2_analytical = (M11 * f2 - M21 * f1) / det
        
        res_dyn1 = dw1_dt - ddth1_analytical
        res_dyn2 = dw2_dt - ddth2_analytical
        
        return torch.cat([res_kin1, res_kin2, res_dyn1, res_dyn2], dim=-1)
    
    def compute_loss(self, t_data, y_data, t_colloc, 
                     lambda_data=1.0, lambda_physics=1.0, lambda_ic=10.0):
        """
        Combined PINN loss.
        
        L = lambda_data * L_data + lambda_physics * L_physics + lambda_ic * L_ic
        
        Args:
            t_data: (N_data, 1) — time points with observations
            y_data: (N_data, 4) — observed states
            t_colloc: (N_colloc, 1) — collocation points for physics
            lambda_*: loss weights
        Returns:
            total_loss, data_loss, physics_loss, ic_loss
        """
        # Data loss
        y_pred = self.forward(t_data)
        data_loss = torch.mean((y_pred - y_data)**2)
        
        # Physics loss (residual at collocation points)
        residual = self.physics_residual(t_colloc)
        physics_loss = torch.mean(residual**2)
        
        # Initial condition loss (stronger weight on t=0)
        ic_pred = self.forward(torch.zeros(1, 1, device=t_data.device))
        ic_loss = torch.mean((ic_pred - y_data[0:1])**2)
        
        total = lambda_data * data_loss + lambda_physics * physics_loss + lambda_ic * ic_loss
        
        return total, data_loss, physics_loss, ic_loss
    
    def predict_trajectory(self, x0, t_eval, dt=None):
        """
        Predict trajectory by evaluating the network at each time point.
        (PINN directly maps t -> state, no integration needed.)
        """
        t_tensor = torch.tensor(t_eval, dtype=torch.float32).unsqueeze(-1)
        with torch.no_grad():
            return self.forward(t_tensor)
