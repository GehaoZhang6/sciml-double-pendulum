# Scientific Machine Learning for the Double Pendulum: A Comparative Study

**Course**: Scientific Machine Learning (Northwestern University)  
**Author**: Gehao Zhang  
**Date**: March 2026

---

## Problem Overview

The **double pendulum** is a classical chaotic system that exhibits extreme sensitivity to initial conditions. Despite simple governing equations derived from Lagrangian mechanics, the system produces complex, unpredictable long-term dynamics — making it an ideal testbed for Scientific Machine Learning (SciML) methods.

This project provides a comprehensive analysis of the double pendulum, including:

1. **Mathematical Analysis**: Lagrangian/Hamiltonian formulation, equations of motion, conservation laws, Lyapunov exponent estimation, and phase space visualization.
2. **Numerical Methods**: Comparison of RK4, RK45, Symplectic Euler, Störmer-Verlet, and implicit midpoint integrators for long-term energy conservation and trajectory accuracy.
3. **Machine Learning Approaches**: Systematic comparison of:
   - Baseline Neural Network (MLP)
   - Residual Network (ResNet)
   - Neural ODE (via `torchdiffeq`)
   - Lagrangian Neural Network (LNN)
   - Hamiltonian Neural Network (HNN)
   - Physics-Informed Neural Network (PINN)

## Repository Structure

```
├── README.md
├── requirements.txt
├── src/
│   ├── double_pendulum.py          # Analytical formulation & numerical solvers
│   ├── numerical_analysis.py       # Solver comparison & stability analysis
│   ├── generate_data.py            # Training data generation
│   ├── models/
│   │   ├── baseline_mlp.py         # Vanilla MLP
│   │   ├── resnet.py               # Residual Network
│   │   ├── neural_ode.py           # Neural ODE
│   │   ├── lagrangian_nn.py        # Lagrangian Neural Network
│   │   ├── hamiltonian_nn.py       # Hamiltonian Neural Network
│   │   └── pinn.py                 # Physics-Informed Neural Network
│   ├── train.py                    # Unified training script
│   ├── evaluate.py                 # Evaluation & comparison
│   └── utils/
│       ├── plotting.py             # Visualization utilities
│       └── metrics.py              # MSE, energy error, Lyapunov exponent
├── figures/                        # Generated plots
├── results/                        # Saved metrics and model checkpoints
├── notebooks/
│   └── analysis.ipynb              # Interactive exploration notebook
└── report/
    └── report.tex                  # 2-page LaTeX report
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Generate Training Data

```bash
python src/generate_data.py --n_trajectories 200 --t_span 10 --dt 0.01
```

### Train All Models

```bash
python src/train.py --model all --epochs 500 --lr 1e-3
```

### Evaluate & Compare

```bash
python src/evaluate.py --output figures/
```

## Mathematical Background

### Lagrangian Formulation

For a double pendulum with masses $m_1, m_2$, rod lengths $l_1, l_2$, and angles $\theta_1, \theta_2$:

$$\mathcal{L} = T - V$$

where the kinetic energy is:

$$T = \frac{1}{2}m_1 l_1^2 \dot{\theta}_1^2 + \frac{1}{2}m_2 \left[ l_1^2 \dot{\theta}_1^2 + l_2^2 \dot{\theta}_2^2 + 2l_1 l_2 \dot{\theta}_1 \dot{\theta}_2 \cos(\theta_1 - \theta_2) \right]$$

and the potential energy is:

$$V = -(m_1 + m_2) g l_1 \cos\theta_1 - m_2 g l_2 \cos\theta_2$$

### Equations of Motion

Applying the Euler-Lagrange equations $\frac{d}{dt}\frac{\partial \mathcal{L}}{\partial \dot{\theta}_i} - \frac{\partial \mathcal{L}}{\partial \theta_i} = 0$ yields a system of coupled second-order nonlinear ODEs.

### Hamiltonian Formulation

The conjugate momenta are:

$$p_1 = \frac{\partial \mathcal{L}}{\partial \dot{\theta}_1}, \quad p_2 = \frac{\partial \mathcal{L}}{\partial \dot{\theta}_2}$$

and the Hamiltonian $H = T + V$ is conserved (total energy).

## Key Results

| Method | Trajectory MSE | Energy Drift (10s) | Training Time |
|--------|---------------|--------------------:|--------------|
| Baseline MLP | High | Large | Fast |
| ResNet | Medium | Medium | Fast |
| Neural ODE | Medium | Small | Moderate |
| LNN | Low | Very Small | Moderate |
| HNN | Low | Negligible | Moderate |
| PINN | Low | Small | Slow |

*(See the full report and `figures/` for detailed quantitative comparisons.)*

## Key Findings

1. **Structure-preserving networks** (HNN, LNN) significantly outperform black-box models in long-term energy conservation.
2. **Neural ODEs** provide continuous-time dynamics but don't inherently conserve energy without structural priors.
3. **PINNs** effectively encode the physics loss but require careful balancing of loss terms.
4. **Symplectic integrators** (Störmer-Verlet) provide the best energy conservation among numerical methods but are slower.

## References

- Greydanus et al., "Hamiltonian Neural Networks," NeurIPS 2019
- Cranmer et al., "Lagrangian Neural Networks," ICLR 2020 Workshop
- Chen et al., "Neural Ordinary Differential Equations," NeurIPS 2018
- Raissi et al., "Physics-Informed Neural Networks," JCP 2019
