# model.py
import torch
import torch.nn as nn


class TorchMinMaxScaler(nn.Module):
    """
    Differentiable drop-in for sklearn's MinMaxScaler.

    Constructed directly from a fitted sklearn MinMaxScaler so the same
    normalisation is used at preprocessing time and inside the loss function.
    All operations are native PyTorch — the computational graph is never broken.

    sklearn transform:  X_scaled = X * scale_ + min_
    sklearn inverse:    X        = (X_scaled - min_) / scale_
    """

    def __init__(self, sk_scaler):
        super().__init__()
        # Register as buffers so they move to the right device with .to(device)
        # and are included in state_dict (but have no gradient).
        self.register_buffer('scale', torch.tensor(sk_scaler.scale_, dtype=torch.float32))
        self.register_buffer('min_',  torch.tensor(sk_scaler.min_,   dtype=torch.float32))

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Map physical values → scaled space  (fully differentiable)."""
        return x * self.scale + self.min_

    def inverse_transform(self, x_scaled: torch.Tensor) -> torch.Tensor:
        """Map scaled space → physical values  (fully differentiable)."""
        return (x_scaled - self.min_) / self.scale


# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Pre-activation residual block with LayerNorm and Tanh.
    Helps gradients flow through deep networks without vanishing.
    """

    def __init__(self, dim: int, hidden: int, dropout: float = 0.05):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.LayerNorm(dim),
        )
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


# ─────────────────────────────────────────────────────────────────────────────

class FlowMapMLP(nn.Module):
    """
    Physics-Informed flow-map surrogate for the HORUS-2B re-entry vehicle.

    Maps the current normalised state + controls (x_t) to the next normalised
    state (x_{t+1}) in a single forward pass, replacing one step of numerical
    integration.

    Default dimensions match the PINN preprocessing pipeline:
      - in_dim  = 13  :  delta_t, r, Mach, lat, lon, heading, gamma, AoA,
                         bank, density, CD, CL, airspeed
      - out_dim = 10  :  Δr, Δlat, Δlon, Δheading, Δgamma, ΔCD, ΔCL,
                         ΔV, Δq_dot, Δg_load

    Architecture: Linear → Tanh → N residual blocks → Linear head
    The residual structure lets the network learn small corrections on top of
    a linear baseline, which works well for smooth dynamical systems.
    """

    def __init__(
        self,
        in_dim:  int   = 13,
        out_dim: int   = 10,
        hidden:  int   = 256,
        depth:   int   = 6,
        dropout: float = 0.05,
    ):
        super().__init__()

        # Initial projection from input space to hidden space
        layers = [
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
        ]

        # Residual trunk
        for _ in range(depth):
            layers.append(ResidualBlock(hidden, hidden, dropout))

        # Output head (no activation — regression task)
        layers.append(nn.Linear(hidden, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
