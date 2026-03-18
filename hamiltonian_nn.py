"""
Hamiltonian Neural Network (HNN) for the Double Pendulum.

Key idea: Instead of learning arbitrary dynamics, learn the Hamiltonian H(q,p)
and derive dynamics via Hamilton's equations:
    dq/dt =  ∂H/∂p
    dp/dt = -∂H/∂q

This automatically conserves the learned Hamiltonian, giving excellent
long-term energy conservation.

Reference: Greydanus et al., "Hamiltonian Neural Networks," NeurIPS 2019
"""

import torch
import torch.nn as nn


class HNN(nn.Module):
    """
    Hamiltonian Neural Network.
    
    Learns H(q, p) as a neural network, then computes dynamics via
    automatic differentiation of Hamilton's equations.
    
    For the double pendulum:
        q = [theta1, theta2]  (generalized coordinates)
        p = [p1, p2]          (conjugate momenta)
    
    The network input is the full state [q, p] = [theta1, theta2, p1, p2].
    """
    def __init__(self, state_dim=4, hidden_dim=200, n_layers=3):
        super().__init__()
        
        # Network to learn H(q, p)
        layers = [nn.Linear(state_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))  # Scalar output: H
        self.hamiltonian_net = nn.Sequential(*layers)
    
    def hamiltonian(self, x):
        """
        Compute the learned Hamiltonian H(q, p).
        
        Args:
            x: (batch, 4) — [theta1, theta2, p1, p2]
        Returns:
            H: (batch, 1) — scalar energy
        """
        return self.hamiltonian_net(x)
    
    def forward(self, x):
        """
        Compute time derivatives via Hamilton's equations.
        
        dq_i/dt =  ∂H/∂p_i
        dp_i/dt = -∂H/∂q_i
        
        Args:
            x: (batch, 4) — [theta1, theta2, p1, p2]
        Returns:
            dx/dt: (batch, 4) — [dtheta1/dt, dtheta2/dt, dp1/dt, dp2/dt]
        """
        x = x.requires_grad_(True)
        H = self.hamiltonian(x)
        
        # Compute gradients: dH/dx = [dH/dq1, dH/dq2, dH/dp1, dH/dp2]
        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        
        # Hamilton's equations
        # dq/dt = dH/dp, dp/dt = -dH/dq
        n = x.shape[-1] // 2  # n=2 for double pendulum
        
        dqdt = dH[..., n:]    # dH/dp -> dq/dt
        dpdt = -dH[..., :n]   # -dH/dq -> dp/dt
        
        return torch.cat([dqdt, dpdt], dim=-1)
    
    def predict_trajectory(self, x0, t_eval, dt=0.01):
        """
        Roll out trajectory using symplectic Euler integration.
        
        Using symplectic integration preserves the Hamiltonian structure
        even during inference, giving better long-term behavior.
        """
        traj = [x0]
        x = x0
        n = x.shape[-1] // 2
        
        for i in range(len(t_eval) - 1):
            x_grad = x.requires_grad_(True)
            H = self.hamiltonian(x_grad)
            dH = torch.autograd.grad(H.sum(), x_grad, create_graph=False)[0]
            
            # Symplectic Euler: update p first, then q
            p_new = x[..., n:] - dt * dH[..., :n]   # dp/dt = -dH/dq
            
            # Recompute dH/dp with new p
            x_new_p = torch.cat([x[..., :n], p_new], dim=-1).requires_grad_(True)
            H_new = self.hamiltonian(x_new_p)
            dH_new = torch.autograd.grad(H_new.sum(), x_new_p, create_graph=False)[0]
            
            q_new = x[..., :n] + dt * dH_new[..., n:]  # dq/dt = dH/dp
            
            x = torch.cat([q_new, p_new], dim=-1).detach()
            traj.append(x)
        
        return torch.stack(traj)
