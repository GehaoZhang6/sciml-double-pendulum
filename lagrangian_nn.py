"""
Lagrangian Neural Network (LNN) for the Double Pendulum.

Key idea: Learn the Lagrangian L(q, qdot) and derive dynamics via
the Euler-Lagrange equations:
    d/dt (∂L/∂qdot) - ∂L/∂q = 0

This is equivalent to learning the kinetic and potential energy structure,
which guarantees energy conservation and physical consistency.

Reference: Cranmer et al., "Lagrangian Neural Networks," ICLR 2020 Workshop
"""

import torch
import torch.nn as nn


class LNN(nn.Module):
    """
    Lagrangian Neural Network.
    
    Learns L(q, qdot) and derives accelerations via Euler-Lagrange equations.
    
    For the double pendulum:
        q = [theta1, theta2]
        qdot = [omega1, omega2]
    
    The Euler-Lagrange equation gives:
        d²L/dqdot² * qddot = dL/dq - d²L/(dq dqdot) * qdot
    
    We solve for qddot (accelerations).
    """
    def __init__(self, q_dim=2, hidden_dim=200, n_layers=3):
        super().__init__()
        self.q_dim = q_dim
        input_dim = 2 * q_dim  # [q, qdot]
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.Softplus()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Softplus()])
        layers.append(nn.Linear(hidden_dim, 1))  # Scalar Lagrangian
        self.lagrangian_net = nn.Sequential(*layers)
    
    def lagrangian(self, q, qdot):
        """
        Compute L(q, qdot).
        
        Args:
            q: (batch, 2) — [theta1, theta2]
            qdot: (batch, 2) — [omega1, omega2]
        Returns:
            L: (batch, 1)
        """
        x = torch.cat([q, qdot], dim=-1)
        return self.lagrangian_net(x)
    
    def forward(self, x):
        """
        Compute time derivatives [qdot, qddot] from the Euler-Lagrange equation.
        
        The EL equation in matrix form:
            M(q, qdot) * qddot = f(q, qdot)
        
        where M = d²L/dqdot² (the mass matrix / Hessian of L w.r.t. qdot)
        and f = dL/dq - d²L/(dq dqdot) * qdot
        
        Args:
            x: (batch, 4) — [theta1, theta2, omega1, omega2]
        Returns:
            dx/dt: (batch, 4) — [omega1, omega2, alpha1, alpha2]
        """
        q = x[..., :self.q_dim].requires_grad_(True)
        qdot = x[..., self.q_dim:].requires_grad_(True)
        
        L = self.lagrangian(q, qdot)
        
        # dL/dqdot
        dL_dqdot = torch.autograd.grad(L.sum(), qdot, create_graph=True)[0]
        
        # dL/dq
        dL_dq = torch.autograd.grad(L.sum(), q, create_graph=True)[0]
        
        # Hessian d²L/dqdot² (mass matrix M)
        # M_ij = d²L / (dqdot_i dqdot_j)
        n = self.q_dim
        M = torch.zeros(*q.shape[:-1], n, n, device=q.device)
        for i in range(n):
            dL_dqdot_i = dL_dqdot[..., i]
            grad_i = torch.autograd.grad(dL_dqdot_i.sum(), qdot, create_graph=True)[0]
            M[..., i, :] = grad_i
        
        # Mixed Hessian d²L/(dq dqdot) * qdot
        # C_i = sum_j d²L/(dq_j dqdot_i) * qdot_j
        C = torch.zeros_like(qdot)
        for i in range(n):
            dL_dqdot_i = dL_dqdot[..., i]
            grad_q_i = torch.autograd.grad(dL_dqdot_i.sum(), q, create_graph=True)[0]
            C[..., i] = (grad_q_i * qdot).sum(dim=-1)
        
        # Solve M * qddot = dL/dq - C
        rhs = dL_dq - C
        
        # Add small regularization for numerical stability
        M_reg = M + 1e-6 * torch.eye(n, device=M.device)
        qddot = torch.linalg.solve(M_reg, rhs.unsqueeze(-1)).squeeze(-1)
        
        return torch.cat([qdot, qddot], dim=-1)
    
    def predict_trajectory(self, x0, t_eval, dt=0.01):
        """Roll out using RK4 integration."""
        traj = [x0]
        x = x0
        for i in range(len(t_eval) - 1):
            k1 = self.forward(x)
            k2 = self.forward(x + dt/2 * k1)
            k3 = self.forward(x + dt/2 * k2)
            k4 = self.forward(x + dt * k3)
            x = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            x = x.detach()
            traj.append(x)
        return torch.stack(traj)
