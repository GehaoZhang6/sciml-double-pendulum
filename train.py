"""
Unified training script for all ML models on the double pendulum.

Usage:
    python train.py --model all --epochs 500 --lr 1e-3
    python train.py --model hnn --epochs 1000 --lr 5e-4
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import time
from tqdm import tqdm

from models.baseline_mlp import BaselineMLP
from models.resnet import ResNetODE
from models.neural_ode import NeuralODE
from models.hamiltonian_nn import HNN
from models.lagrangian_nn import LNN
from models.pinn import PINN


def load_data(data_path='data/double_pendulum_data.npz'):
    """Load preprocessed training data."""
    data = np.load(data_path)
    return data


def train_derivative_model(model, train_states, train_derivs, val_states, val_derivs,
                           epochs=500, lr=1e-3, batch_size=512, device='cpu'):
    """
    Train models that learn state -> derivative mapping.
    (MLP, ResNet, Neural ODE, HNN, LNN)
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Flatten trajectories to (N*T, 4) pairs
    X_train = torch.tensor(train_states.reshape(-1, 4), dtype=torch.float32, device=device)
    Y_train = torch.tensor(train_derivs.reshape(-1, 4), dtype=torch.float32, device=device)
    X_val = torch.tensor(val_states.reshape(-1, 4), dtype=torch.float32, device=device)
    Y_val = torch.tensor(val_derivs.reshape(-1, 4), dtype=torch.float32, device=device)
    
    n_train = X_train.shape[0]
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            x_batch = X_train[idx]
            y_batch = Y_train[idx]
            
            pred = model(x_batch)
            loss = nn.functional.mse_loss(pred, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = nn.functional.mse_loss(val_pred, Y_val).item()
        
        history['train_loss'].append(epoch_loss / n_batches)
        history['val_loss'].append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs} — Train: {epoch_loss/n_batches:.6f}, Val: {val_loss:.6f}")
    
    model.load_state_dict(best_state)
    return model, history


def train_pinn(model, t_data, y_data, t_span=(0, 10), epochs=2000, lr=1e-3,
               n_colloc=1000, device='cpu'):
    """
    Train PINN with combined data + physics loss.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    t_tensor = torch.tensor(t_data, dtype=torch.float32, device=device).unsqueeze(-1)
    y_tensor = torch.tensor(y_data, dtype=torch.float32, device=device)
    
    history = {'total': [], 'data': [], 'physics': [], 'ic': []}
    
    for epoch in range(epochs):
        model.train()
        
        # Random collocation points in [0, t_span]
        t_colloc = torch.rand(n_colloc, 1, device=device) * (t_span[1] - t_span[0]) + t_span[0]
        
        # Adaptive lambda: increase physics weight over training
        lambda_physics = min(1.0, 0.01 + epoch / epochs)
        
        total, data_loss, phys_loss, ic_loss = model.compute_loss(
            t_tensor, y_tensor, t_colloc,
            lambda_data=1.0, lambda_physics=lambda_physics, lambda_ic=10.0
        )
        
        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        history['total'].append(total.item())
        history['data'].append(data_loss.item())
        history['physics'].append(phys_loss.item())
        history['ic'].append(ic_loss.item())
        
        if (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1}/{epochs} — Total: {total.item():.6f}, "
                  f"Data: {data_loss.item():.6f}, Physics: {phys_loss.item():.6f}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='all',
                        choices=['mlp', 'resnet', 'node', 'hnn', 'lnn', 'pinn', 'all'])
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--data', type=str, default='data/double_pendulum_data.npz')
    parser.add_argument('--output', type=str, default='results/')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    data = load_data(args.data)
    
    models_to_train = {
        'mlp': BaselineMLP(state_dim=4, hidden_dim=128, n_layers=3),
        'resnet': ResNetODE(state_dim=4, hidden_dim=128, n_blocks=4),
        'node': NeuralODE(state_dim=4, hidden_dim=128),
        'hnn': HNN(state_dim=4, hidden_dim=200, n_layers=3),
        'lnn': LNN(q_dim=2, hidden_dim=200, n_layers=3),
    }
    
    if args.model != 'all':
        if args.model == 'pinn':
            models_to_train = {}
        else:
            models_to_train = {args.model: models_to_train[args.model]}
    
    # Train derivative-based models
    for name, model in models_to_train.items():
        print(f"\n{'='*60}")
        print(f"Training {name.upper()}...")
        print(f"{'='*60}")
        
        t_start = time.time()
        model, history = train_derivative_model(
            model, data['train_states'], data['train_derivatives'],
            data['val_states'], data['val_derivatives'],
            epochs=args.epochs, lr=args.lr, device=device
        )
        elapsed = time.time() - t_start
        
        # Save
        torch.save(model.state_dict(), os.path.join(args.output, f'{name}_weights.pt'))
        np.savez(os.path.join(args.output, f'{name}_history.npz'), **history)
        print(f"  Done in {elapsed:.1f}s. Best val loss: {min(history['val_loss']):.6f}")
    
    # Train PINN separately (different training loop)
    if args.model in ['pinn', 'all']:
        print(f"\n{'='*60}")
        print("Training PINN...")
        print(f"{'='*60}")
        
        pinn = PINN(hidden_dim=128, n_layers=4)
        # Use a single trajectory for PINN
        t_data = np.arange(0, 10, 0.01)
        y_data = data['train_states'][0]  # First trajectory
        
        t_start = time.time()
        pinn, history = train_pinn(pinn, t_data, y_data, epochs=args.epochs * 4,
                                    lr=args.lr, device=device)
        elapsed = time.time() - t_start
        
        torch.save(pinn.state_dict(), os.path.join(args.output, 'pinn_weights.pt'))
        np.savez(os.path.join(args.output, 'pinn_history.npz'), **history)
        print(f"  Done in {elapsed:.1f}s")
    
    print("\nAll models trained!")


if __name__ == "__main__":
    main()
