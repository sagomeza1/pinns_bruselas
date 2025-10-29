"""
Demostración práctica en PyTorch: entrenamiento de una capa GammaBias con WeightNorm
y registro de los parámetros v, g, gamma y bias por época.

Requisitos:
- torch
- pandas (solo si quieres guardar/leer el CSV de historia)
"""

import numpy as np
import torch
import torch.nn as nn
#from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import weight_norm
from torch.utils.data import TensorDataset, DataLoader

# 1) Definimos la capa equivalente en PyTorch
class GammaBiasLayerTorch(nn.Module):
    def __init__(self, in_features, units):
        super().__init__()
        dense = nn.Linear(in_features, units, bias=False)
        # Inicialización uniforme como en el ejemplo de TF
        nn.init.uniform_(dense.weight, -1.0, 1.0)
        # Aplicamos Weight Normalization; dim=0 porque normalizamos por salida
        self.w = weight_norm(dense, name='weight', dim=0)
        # Escala (gamma) y sesgo (bias) por unidad
        self.gamma = nn.Parameter(torch.ones(units))
        self.bias  = nn.Parameter(torch.zeros(units))

    def forward(self, x):
        z = self.w(x)          # (N, units)
        z = z * self.gamma     # broadcast por canal
        return z + self.bias   # (N, units)

# 2) Modelo simple 2->1 usando solo esta capa
class SimpleModel(nn.Module):
    def __init__(self, in_features=2, units=1):
        super().__init__()
        self.layer = GammaBiasLayerTorch(in_features, units)

    def forward(self, x):
        return self.layer(x)

def train_and_log(epochs=30, batch_size=32, lr=0.05, seed=0):
    torch.manual_seed(seed)

    # 3) Dataset sintético: y = 3*x1 - 2*x2 + 1
    N = 256
    X = torch.randn(N, 2)
    true_w = torch.tensor([3.0, -2.0]).view(2, 1)
    true_b = 1.0
    y = X @ true_w + true_b  # (N,1)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 4) Entrenamiento
    model = SimpleModel(in_features=2, units=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = []

    def snapshot(model, epoch, loss_value):
        with torch.no_grad():
            # Acceso a parámetros internos de weight_norm:
            # - model.layer.w.weight_g  -> g (shape: [out_features])
            # - model.layer.w.weight_v  -> v (shape: [out_features, in_features] porque dim=0)
          #  print(dir(model.layer.w))
         #   input()
            g = model.layer.w.weight_g.detach().cpu().numpy().copy().reshape(-1)
            v = model.layer.w.weight_v.detach().cpu().numpy().copy()
            v_norm = np.linalg.norm(v, axis=1)  # norma por salida
            gamma = model.layer.gamma.detach().cpu().numpy().copy().reshape(-1)
            bias  = model.layer.bias.detach().cpu().numpy().copy().reshape(-1)

            # Peso efectivo W = g * v / ||v||
            W_eff = (g[:, None] * (v / (v_norm[:, None] + 1e-12)))  # (out_features, in_features)

            row = {
                "epoch": epoch,
                "loss": float(loss_value),
                "g[0]": float(g[0]),
                "||v[0]||": float(v_norm[0]),
                "v[0,0]": float(v[0,0]),
                "v[0,1]": float(v[0,1]),
                "gamma[0]": float(gamma[0]),
                "bias[0]": float(bias[0]),
                "W_eff[0,0]": float(W_eff[0,0]),
                "W_eff[0,1]": float(W_eff[0,1]),
            }
            return row

    # Snapshot inicial antes de entrenar (del modelo real)
    with torch.no_grad():
        loss0 = loss_fn(model(X), y).item()
        history.append(snapshot(model, epoch=0, loss_value=loss0))

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred_full = model(X)
            loss_full = loss_fn(pred_full, y).item()
            history.append(snapshot(model, epoch=epoch, loss_value=loss_full))

    return model, history

if __name__ == "__main__":
    model, history = train_and_log()
    # Mostrar últimos 5 snapshots
    import pandas as pd
    df = pd.DataFrame(history)
    print(df.tail(5).to_string(index=False))
    # Guardar CSV
    df.to_csv("history_gamma_bias_wn.csv", index=False)
    print("\nCSV guardado en 'history_gamma_bias_wn.csv'")
