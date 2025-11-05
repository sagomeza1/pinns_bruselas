#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch rewrite of PINN_weather.py (Keras/TensorFlow)
- Replica de arquitectura (GammaBias + WeightNorm, tanh)
- Misma normalización/no dimensionalización
- Misma construcción de lotes y entrenamiento
- Misma función de pérdida física y de datos
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0")

import math
import numpy as np
import pandas as pd
import scipy.io
import datetime as dt

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from pathlib import Path

# -------------------------
# Configuración de dispositivo
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using PyTorch version: {torch.__version__}, device: {device}")

# -------------------------
# Carga y preprocesamiento de datos (idéntico en lógica al original)
# -------------------------
WS_data = scipy.io.loadmat('Weather_data.mat')

# Convertir fecha a segundos relativos
date_0 = WS_data['Date'][0]
date = []
for i in range(0, len(date_0)):
    date = np.append(date, str(date_0[i])[2: -2])
time_init = dt.datetime(int(date[0][0:4]), int(date[0][5:7]), int(date[0][8:10]), int(date[0][11:13]), int(date[0][14:16]))
T_nan_index = np.argwhere(pd.isna(date))
date = np.delete(date, T_nan_index[:, 0],  0)
print('Double-check for NaN in time sequence', np.sum(pd.isna(date)))

Seconds = np.zeros((date.shape[0], 1))
for index in range(date.shape[0]):
    Seconds[index, 0] = ((dt.datetime(int(date[index][0:4]), int(date[index][5:7]), int(date[index][8:10]), int(date[index][11:13]), int(date[index][14:16])) - time_init).total_seconds())
T_WS = Seconds

# Coordenadas cartesianas
X_WS = np.array(6378000 * np.sin(np.radians(WS_data['Lon'])))[0]
Y_WS = np.array(6378000 * np.sin(np.radians(WS_data['Lat'])))[0]
Z_WS = np.array(WS_data['Alt'])[0]
Temp_WS = np.array(WS_data['Temperature'])[0]

# Proyección del viento
U_WS = (WS_data['WindSpeed'] * WS_data['WindDirectionX'])[0]
V_WS = (WS_data['WindSpeed'] * WS_data['WindDirectionY'])[0]

# Presión mbar -> Pa
P_WS = WS_data['Pressure'][0] * 100

# Remover NaN por tiempo
X_WS = np.delete(X_WS, T_nan_index[:, 0],  0)
Y_WS = np.delete(Y_WS, T_nan_index[:, 0],  0)
Z_WS = np.delete(Z_WS, T_nan_index[:, 0],  0)
U_WS = np.delete(U_WS, T_nan_index[:, 0],  0)
V_WS = np.delete(V_WS, T_nan_index[:, 0],  0)
P_WS = np.delete(P_WS, T_nan_index[:, 0],  0)
Temp_WS = np.delete(Temp_WS, T_nan_index[:, 0],  0)

# Matrices: estaciones x snapshots
T_WS = np.reshape(T_WS, (int(T_WS.shape[0] / 21), 21)).T
X_WS = np.reshape(X_WS, (T_WS.shape[1], T_WS.shape[0])).T
Y_WS = np.reshape(Y_WS, (T_WS.shape[1], T_WS.shape[0])).T
Z_WS = np.reshape(Z_WS, (T_WS.shape[1], T_WS.shape[0])).T
U_WS = np.reshape(U_WS, (T_WS.shape[1], T_WS.shape[0])).T
V_WS = np.reshape(V_WS, (T_WS.shape[1], T_WS.shape[0])).T
P_WS = np.reshape(P_WS, (T_WS.shape[1], T_WS.shape[0])).T
Temp_WS = np.reshape(Temp_WS, (T_WS.shape[1], T_WS.shape[0])).T
print('Number of weather stations:', T_WS.shape[0])

# Remover NaN por ubicación
X_nan_index = np.argwhere(np.isnan(X_WS))
T_WS = np.delete(T_WS, X_nan_index[:, 0],  0)
P_WS = np.delete(P_WS, X_nan_index[:, 0],  0)
U_WS = np.delete(U_WS, X_nan_index[:, 0],  0)
V_WS = np.delete(V_WS, X_nan_index[:, 0],  0)
X_WS = np.delete(X_WS, X_nan_index[:, 0],  0)
Y_WS = np.delete(Y_WS, X_nan_index[:, 0],  0)
Z_WS = np.delete(Z_WS, X_nan_index[:, 0],  0)
Temp_WS = np.delete(Temp_WS, X_nan_index[:, 0],  0)
print('Double-check for NaN in location field', np.sum(np.isnan(X_WS)))

# Días para reconstrucción
n_days = 14
samples = int(144 * n_days)
T_WS = T_WS[:, :samples]
X_WS = X_WS[:, :samples]
Y_WS = Y_WS[:, :samples]
Z_WS = Z_WS[:, :samples]
U_WS = U_WS[:, :samples]
V_WS = V_WS[:, :samples]
P_WS = P_WS[:, :samples]
Temp_WS = Temp_WS[:, :samples]

# Ordenar por X ascendente por snapshot
for snap in range(0, T_WS.shape[1]):
    index_sort = np.argsort(X_WS[:, snap])
    T_WS[:, snap] = T_WS[index_sort, snap]
    X_WS[:, snap] = X_WS[index_sort, snap]
    Y_WS[:, snap] = Y_WS[index_sort, snap]
    Z_WS[:, snap] = Z_WS[index_sort, snap]
    U_WS[:, snap] = U_WS[index_sort, snap]
    V_WS[:, snap] = V_WS[index_sort, snap]
    P_WS[:, snap] = P_WS[index_sort, snap]
    Temp_WS[:, snap] = Temp_WS[index_sort, snap]

# Eliminar estaciones con NaN constantes en U,V,P
uvp_mean = np.nanmean(np.concatenate([U_WS, V_WS, P_WS], axis=1), axis=1)[:, None]
vel_nan_index = np.argwhere(np.isnan(uvp_mean))
T_WS = np.delete(T_WS, vel_nan_index[:, 0],  0)
P_WS = np.delete(P_WS, vel_nan_index[:, 0],  0)
U_WS = np.delete(U_WS, vel_nan_index[:, 0],  0)
V_WS = np.delete(V_WS, vel_nan_index[:, 0],  0)
X_WS = np.delete(X_WS, vel_nan_index[:, 0],  0)
Y_WS = np.delete(Y_WS, vel_nan_index[:, 0],  0)
Z_WS = np.delete(Z_WS, vel_nan_index[:, 0],  0)
Temp_WS = np.delete(Temp_WS, vel_nan_index[:, 0],  0)

# Corrección de presión al nivel del mar (ISA)
P_WS = P_WS * (1 - 0.0065 * Z_WS / (Temp_WS + 273.15 + 0.0065 * Z_WS)) ** (-5.257)

# Centrado
x_min = np.min(X_WS); x_max = np.max(X_WS); X_WS = X_WS - (x_min + x_max) / 2
y_min = np.min(Y_WS); y_max = np.max(Y_WS); Y_WS = Y_WS - (y_min + y_max) / 2
t_min = np.min(T_WS); t_max = np.max(T_WS); T_WS = T_WS - t_min

# Mallado PINN
T_PINN = T_WS[0:1, :]
R = 0.2
R_PINN = 6378000 * np.sin(np.radians(R))
x_PINN = np.arange(x_min - R_PINN, x_max + R_PINN, R_PINN)
y_PINN = np.arange(y_min - R_PINN, y_max + R_PINN, R_PINN)
x_PINN = x_PINN - (x_min + x_max) / 2
y_PINN = y_PINN - (y_min + y_max) / 2
X_PINN, Y_PINN = np.meshgrid(x_PINN, y_PINN)
X_PINN = X_PINN.flatten('F')[:, None]
Y_PINN = Y_PINN.flatten('F')[:, None]

dim_T_PINN = T_PINN.shape[1]
dim_N_PINN = X_PINN.shape[0]

T_PINN = np.tile(T_PINN, (dim_N_PINN, 1))
X_PINN = np.tile(X_PINN, dim_T_PINN)
Y_PINN = np.tile(Y_PINN, dim_T_PINN)

# No dimensionalización
L = math.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
W = math.sqrt(np.nanmax(abs(U_WS)) ** 2 + np.nanmax(abs(V_WS)) ** 2)
rho = 1.269
nu = 1.382e-5
Re = int(W * L / nu)
P0 = np.nanmean(P_WS)
print('L:', L, 'W', W, 'P0', P0, 'Re', Re)

X_WS = X_WS / L; Y_WS = Y_WS / L; T_WS = T_WS * W / L
P_WS = (P_WS - P0) / rho / (W ** 2)
U_WS = U_WS / W; V_WS = V_WS / W

X_PINN = X_PINN / L; Y_PINN = Y_PINN / L; T_PINN = T_PINN * W / L

# Validación (estaciones retiradas)
WS_val = np.array([1, 2, 3, 5, 7, 9, 10, 11, 13, 14, 15, 16, 19])
T_val = T_WS[WS_val, :]
P_val = P_WS[WS_val, :]
U_val = U_WS[WS_val, :]
V_val = V_WS[WS_val, :]
X_val = X_WS[WS_val, :]
Y_val = Y_WS[WS_val, :]
Z_val = Z_WS[WS_val, :]

T_WS = np.delete(T_WS, WS_val, 0)
P_WS = np.delete(P_WS, WS_val, 0)
U_WS = np.delete(U_WS, WS_val, 0)
V_WS = np.delete(V_WS, WS_val, 0)
X_WS = np.delete(X_WS, WS_val, 0)
Y_WS = np.delete(Y_WS, WS_val, 0)
print('Number of final weather stations available for training:', T_WS.shape[0])

dim_N_WS = X_WS.shape[0]
dim_T_WS = X_WS.shape[1]

# -------------------------
# Capa GammaBias (replica)
# -------------------------
class GammaBiasLayerTorch(nn.Module):
    def __init__(self, in_features, units):
        super().__init__()
        dense = nn.Linear(in_features, units, bias=False)
        nn.init.uniform_(dense.weight, -1.0, 1.0)
        self.w = weight_norm(dense, name='weight', dim=0)  # normaliza por salida
        self.gamma = nn.Parameter(torch.ones(units, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(units, dtype=torch.float32))

    def forward(self, x):
        y = self.w(x)              # [B, units]
        y = self.gamma * y + self.bias
        return y

# -------------------------
# Modelo MLP (mismas capas)
# -------------------------
class PINNNet(nn.Module):
    def __init__(self, num_inputs=3, num_outputs=3):
        super().__init__()
        neurons = 200 * num_outputs
        layers = [num_inputs] + (2 * (num_inputs + num_outputs)) * [neurons] + [num_outputs]

        mods = []
        # Primera capa
        mods.append(GammaBiasLayerTorch(layers[0], layers[1]))
        mods.append(nn.Tanh())
        # Capas intermedias (primera sección)
        for l in layers[2: 2 * int((len(layers) - 2) / 3)]:
            mods.append(GammaBiasLayerTorch(layers[1], l))
            mods.append(nn.Tanh())
        # Capas intermedias (segunda sección) repite tamaño layers[-2]
        for l in layers[2 * int((len(layers) - 2) / 3): -1]:
            mods.append(GammaBiasLayerTorch(layers[1], layers[-2]))
        # Salida
        mods.append(GammaBiasLayerTorch(layers[-2], layers[-1]))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)

# -------------------------
# Utilidades de derivadas
# -------------------------
def grad(outputs, inputs):
    """ grad vectorial; devuelve d(outputs)/d(inputs) con misma forma que inputs """
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]

# -------------------------
# Funciones de pérdida
# -------------------------
mse_loss = nn.MSELoss(reduction='mean')

def loss_NS_2D(model, t_eqns, x_eqns, y_eqns):
    # Asegurar gradientes con respecto a entradas
    for ten in (t_eqns, x_eqns, y_eqns):
        if not ten.requires_grad:
            ten.requires_grad_(True)

    X = torch.cat([t_eqns, x_eqns, y_eqns], dim=1)  # [N, 3]
    Y = model(X)                                     # [N, 3]
    u, v, p = Y[:, 0:1], Y[:, 1:2], Y[:, 2:3]

    # Derivadas
    u_t = grad(u, t_eqns)
    v_t = grad(v, t_eqns)

    u_x = grad(u, x_eqns)
    v_x = grad(v, x_eqns)
    p_x = grad(p, x_eqns)

    u_y = grad(u, y_eqns)
    v_y = grad(v, y_eqns)
    p_y = grad(p, y_eqns)

    # Residuales NS (no viscosos como en el original)
    e1 = (u_x + v_y)
    e2 = (u_t + (u * u_x + v * u_y) + p_x)
    e3 = (v_t + (u * v_x + v * v_y) + p_y)

    zero = torch.zeros_like(e1)
    return mse_loss(zero, e1) + mse_loss(zero, e2) + mse_loss(zero, e3)

def loss_u(model, t_data, x_data, y_data, u_true):
    X = torch.cat([t_data, x_data, y_data], dim=1)
    Y = model(X)
    u_pred = Y[:, 0:1]
    denom = torch.std(u_true, unbiased=False) ** 2 + 1e-12
    return mse_loss(u_true, u_pred) / denom

def loss_v(model, t_data, x_data, y_data, v_true):
    X = torch.cat([t_data, x_data, y_data], dim=1)
    Y = model(X)
    v_pred = Y[:, 1:2]
    denom = torch.std(v_true, unbiased=False) ** 2 + 1e-12
    return mse_loss(v_true, v_pred) / denom

def loss_p(model, t_data, x_data, y_data, p_true):
    X = torch.cat([t_data, x_data, y_data], dim=1)
    Y = model(X)
    p_pred = Y[:, 2:3]
    denom = torch.std(p_true, unbiased=False) ** 2 + 1e-12
    return mse_loss(p_true, p_pred) / denom

def loss_total(model,
               t_u, x_u, y_u, u_u,
               t_v, x_v, y_v, v_v,
               t_p, x_p, y_p, p_p,
               t_eq_ref, x_eq_ref, y_eq_ref,
               t_eq, x_eq, y_eq,
               lamb):
    NS_eqns = lamb * loss_NS_2D(model, t_eq, x_eq, y_eq)
    NS_data = lamb * loss_NS_2D(model, t_eq_ref, x_eq_ref, y_eq_ref)
    P_e = loss_p(model, t_p, x_p, y_p, p_p)
    U_e = loss_u(model, t_u, x_u, y_u, u_u)
    V_e = loss_v(model, t_v, x_v, y_v, v_v)

    total_e = NS_eqns + NS_data + U_e + V_e + P_e
    # Fórmula idéntica al original
    return (NS_eqns ** 2 + NS_data ** 2 + U_e ** 2 + V_e ** 2 + P_e ** 2) / (total_e + 1e-12), (NS_eqns, P_e, U_e, V_e)

# -------------------------
# Entrenamiento
# -------------------------
# Modelo y optimizador
model = PINNNet(num_inputs=3, num_outputs=3).to(device)
print(model)

num_epochs = 1000
lamb = 2.0

# Batch sizes
dim_T_PINN = T_PINN.shape[1]
dim_N_PINN = X_PINN.shape[0]
dim_N_data = dim_N_WS = X_WS.shape[0]
dim_T_data = dim_T_WS = X_WS.shape[1]
dim_T_eqns = dim_T_PINN
dim_N_eqns = dim_N_PINN

R = 0.2  # usado arriba también
batch_PINN = int(np.ceil((dim_N_PINN * dim_T_PINN / n_days * R)))
batch_WS = int(np.ceil(dim_N_WS * dim_T_WS / n_days * R)) 

# Acumuladores de métricas
train_loss_results = []
NS_loss_results = []
P_loss_results = []
U_loss_results = []
V_loss_results = []

# Optimizador (se recrea según el LR adaptativo, como en TF)
def make_optimizer(lr):
    return torch.optim.Adam(model.parameters(), lr=lr)

optimizer = make_optimizer(1e-3)

to_torch = lambda a: torch.from_numpy(a).float().to(device)

for epoch in range(num_epochs):
    # Mezcla y muestreo (idéntico)
    idx_t = np.random.choice(dim_T_WS, dim_T_data, replace=False)
    idx_x = np.random.choice(dim_N_WS, dim_N_data, replace=False)
    t_u = T_WS[:, idx_t][idx_x, :].flatten()[:, None]
    x_u = X_WS[:, idx_t][idx_x, :].flatten()[:, None]
    y_u = Y_WS[:, idx_t][idx_x, :].flatten()[:, None]
    z_u = Z_WS[:, idx_t][idx_x, :].flatten()[:, None]
    u_u = U_WS[:, idx_t][idx_x, :].flatten()[:, None]
    v_u = V_WS[:, idx_t][idx_x, :].flatten()[:, None]
    p_u = P_WS[:, idx_t][idx_x, :].flatten()[:, None]

    idx_t = np.random.choice(dim_T_WS, dim_T_data, replace=False)
    idx_x = np.random.choice(dim_N_WS, dim_N_data, replace=False)
    t_v = T_WS[:, idx_t][idx_x, :].flatten()[:, None]
    x_v = X_WS[:, idx_t][idx_x, :].flatten()[:, None]
    y_v = Y_WS[:, idx_t][idx_x, :].flatten()[:, None]
    z_v = Z_WS[:, idx_t][idx_x, :].flatten()[:, None]
    u_v = U_WS[:, idx_t][idx_x, :].flatten()[:, None]
    v_v = V_WS[:, idx_t][idx_x, :].flatten()[:, None]
    p_v = P_WS[:, idx_t][idx_x, :].flatten()[:, None]

    idx_t = np.random.choice(P_WS.shape[1], P_WS.shape[1], replace=False)
    idx_x = np.random.choice(P_WS.shape[0], P_WS.shape[0], replace=False)
    t_p = T_WS[:, idx_t][idx_x, :].flatten()[:, None]
    x_p = X_WS[:, idx_t][idx_x, :].flatten()[:, None]
    y_p = Y_WS[:, idx_t][idx_x, :].flatten()[:, None]
    z_p = Z_WS[:, idx_t][idx_x, :].flatten()[:, None]
    u_p = U_WS[:, idx_t][idx_x, :].flatten()[:, None]
    v_p = V_WS[:, idx_t][idx_x, :].flatten()[:, None]
    p_p = P_WS[:, idx_t][idx_x, :].flatten()[:, None]

    idx_t = np.random.choice(dim_T_PINN, dim_T_eqns, replace=False)
    idx_x = np.random.choice(dim_N_PINN, dim_N_eqns, replace=False)
    t_eqns = T_PINN[:, idx_t][idx_x, :].flatten()[:, None]
    x_eqns = X_PINN[:, idx_t][idx_x, :].flatten()[:, None]
    y_eqns = Y_PINN[:, idx_t][idx_x, :].flatten()[:, None]

    idx_t = np.random.choice(dim_T_WS, dim_T_data, replace=False)
    idx_x = np.random.choice(dim_N_WS, dim_N_data, replace=False)
    t_eqns_ref = T_WS[:, idx_t][idx_x, :].flatten()[:, None]
    x_eqns_ref = X_WS[:, idx_t][idx_x, :].flatten()[:, None]
    y_eqns_ref = Y_WS[:, idx_t][idx_x, :].flatten()[:, None]

    # Reordenar por lotes completos (igual)
    def shuffle_all(ts, xs, ys, vals):
        idx = np.random.choice(ts.shape[0], ts.shape[0], replace=False)
        return ts[idx, :], xs[idx, :], ys[idx, :], vals[idx, :]

    t_u, x_u, y_u, u_u = shuffle_all(t_u, x_u, y_u, u_u)
    t_v, x_v, y_v, v_v = shuffle_all(t_v, x_v, y_v, v_v)
    t_p, x_p, y_p, p_p = shuffle_all(t_p, x_p, y_p, p_p)
    t_eqns, x_eqns, y_eqns = [arr[np.random.choice(arr.shape[0], arr.shape[0], replace=False), :] for arr in (t_eqns, x_eqns, y_eqns)]
    t_eqns_ref, x_eqns_ref, y_eqns_ref = [arr[np.random.choice(arr.shape[0], arr.shape[0], replace=False), :] for arr in (t_eqns_ref, x_eqns_ref, y_eqns_ref)]

    # Remover NaN remanentes
    def drop_nan(t, x, y, target):
        nan_idx = np.argwhere(np.isnan(target))
        if nan_idx.size > 0:
            t = np.delete(t, nan_idx[:, 0], 0)
            x = np.delete(x, nan_idx[:, 0], 0)
            y = np.delete(y, nan_idx[:, 0], 0)
            target = np.delete(target, nan_idx[:, 0], 0)
        return t, x, y, target

    t_u, x_u, y_u, u_u = drop_nan(t_u, x_u, y_u, u_u)
    t_v, x_v, y_v, v_v = drop_nan(t_v, x_v, y_v, v_v)
    t_p, x_p, y_p, p_p = drop_nan(t_p, x_p, y_p, p_p)

    # Divisiones en lotes
    div_u = range(0, len(x_u), batch_WS)
    div_v = range(0, len(x_v), batch_WS)
    div_p = range(0, len(x_p), batch_WS)
    div_eqns = range(0, len(x_eqns_ref), batch_WS)
    div_PINN = range(0, len(x_eqns), batch_PINN)
    min_div = min([len(div_u), len(div_v), len(div_p), len(div_eqns), len(div_PINN)])

    model.train()
    epoch_loss_vals = []
    epoch_NS_vals = []
    epoch_P_vals = []
    epoch_U_vals = []
    epoch_V_vals = []

    for i in range(min_div):
        iu = div_u[i]; iv = div_v[i]; ip = div_p[i]; ie = div_eqns[i]; ipn = div_PINN[i]

        # Tensores torch
        t_u_b = to_torch(t_u[iu: iu + batch_WS]); x_u_b = to_torch(x_u[iu: iu + batch_WS])
        y_u_b = to_torch(y_u[iu: iu + batch_WS]); u_u_b = to_torch(u_u[iu: iu + batch_WS])

        t_v_b = to_torch(t_v[iv: iv + batch_WS]); x_v_b = to_torch(x_v[iv: iv + batch_WS])
        y_v_b = to_torch(y_v[iv: iv + batch_WS]); v_v_b = to_torch(v_v[iv: iv + batch_WS])

        t_p_b = to_torch(t_p[ip: ip + batch_WS]); x_p_b = to_torch(x_p[ip: ip + batch_WS])
        y_p_b = to_torch(y_p[ip: ip + batch_WS]); p_p_b = to_torch(p_p[ip: ip + batch_WS])

        t_eq_ref_b = to_torch(t_eqns_ref[ie: ie + batch_WS]); x_eq_ref_b = to_torch(x_eqns_ref[ie: ie + batch_WS])
        y_eq_ref_b = to_torch(y_eqns_ref[ie: ie + batch_WS])

        t_eq_b = to_torch(t_eqns[ipn: ipn + batch_PINN]); x_eq_b = to_torch(x_eqns[ipn: ipn + batch_PINN])
        y_eq_b = to_torch(y_eqns[ipn: ipn + batch_PINN])

        optimizer.zero_grad()
        loss_train, (NS_loss, P_loss, U_loss, V_loss) = loss_total(
            model,
            t_u_b, x_u_b, y_u_b, u_u_b,
            t_v_b, x_v_b, y_v_b, v_v_b,
            t_p_b, x_p_b, y_p_b, p_p_b,
            t_eq_ref_b, x_eq_ref_b, y_eq_ref_b,
            t_eq_b, x_eq_b, y_eq_b,
            lamb
        )
        loss_train.backward()
        optimizer.step()

        # Registros
        epoch_loss_vals.append(loss_train.detach().cpu().item())
        epoch_NS_vals.append(NS_loss.detach().cpu().item())
        epoch_P_vals.append(P_loss.detach().cpu().item())
        epoch_U_vals.append(U_loss.detach().cpu().item())
        epoch_V_vals.append(V_loss.detach().cpu().item())

    # Promedios de la época
    ep_loss = float(np.mean(epoch_loss_vals)) if len(epoch_loss_vals) else float('nan')
    ep_NS = float(np.mean(epoch_NS_vals)) if len(epoch_NS_vals) else float('nan')
    ep_P = float(np.mean(epoch_P_vals)) if len(epoch_P_vals) else float('nan')
    ep_U = float(np.mean(epoch_U_vals)) if len(epoch_U_vals) else float('nan')
    ep_V = float(np.mean(epoch_V_vals)) if len(epoch_V_vals) else float('nan')

    train_loss_results.append(ep_loss)
    NS_loss_results.append(ep_NS)
    P_loss_results.append(ep_P)
    U_loss_results.append(ep_U)
    V_loss_results.append(ep_V)

    # Scheduler adaptativo (recrea el optimizador como en TF)
    if ep_loss > 1e-1:
        optimizer = make_optimizer(1e-3)
    elif ep_loss > 3e-2:
        optimizer = make_optimizer(1e-4)
    elif ep_loss > 3e-3:
        optimizer = make_optimizer(1e-5)
    else:
        optimizer = make_optimizer(1e-6)

    print(f"Epoch: {epoch:4d} | Loss training: {ep_loss:10.4e} | NS_Loss: {ep_NS:10.4e} | P_Loss: {ep_P:10.4e} | U_Loss: {ep_U:10.4e} | V_Loss: {ep_V:10.4e} | lr: {optimizer.param_groups[0]['lr']:.1e}")

    # ---------------- Guardado final como en el original ----------------
    if (epoch + 1) % num_epochs == 0:
        model.eval()
        with torch.no_grad():
            # Salidas en grilla
            U_PINN = np.zeros_like(X_PINN)
            V_PINN = np.zeros_like(X_PINN)
            P_PINN = np.zeros_like(X_PINN)
            # Predicciones en WS (train)
            U_WS_pred = np.zeros_like(X_WS)
            V_WS_pred = np.zeros_like(X_WS)
            P_WS_pred = np.zeros_like(X_WS)
            # Predicciones en validación
            U_val_pred = np.zeros_like(X_val)
            V_val_pred = np.zeros_like(X_val)
            P_val_pred = np.zeros_like(X_val)

            # Pred grilla PINN
            for snap in range(0, dim_T_PINN):
                t_out = to_torch(T_PINN[:, snap: snap + 1])
                x_out = to_torch(X_PINN[:, snap: snap + 1])
                y_out = to_torch(Y_PINN[:, snap: snap + 1])
                X_out = torch.cat([t_out, x_out, y_out], dim=1)
                Y_out = model(X_out)
                u_pred, v_pred, p_pred = Y_out[:, 0:1], Y_out[:, 1:2], Y_out[:, 2:3]
                U_PINN[:, snap: snap + 1] = u_pred.cpu().numpy()
                V_PINN[:, snap: snap + 1] = v_pred.cpu().numpy()
                P_PINN[:, snap: snap + 1] = p_pred.cpu().numpy()

            # Pred en WS (train)
            for snap in range(0, dim_T_WS):
                t_out = to_torch(T_WS[:, snap: snap + 1])
                x_out = to_torch(X_WS[:, snap: snap + 1])
                y_out = to_torch(Y_WS[:, snap: snap + 1])
                X_out = torch.cat([t_out, x_out, y_out], dim=1)
                Y_out = model(X_out)
                u_pred, v_pred, p_pred = Y_out[:, 0:1], Y_out[:, 1:2], Y_out[:, 2:3]
                U_WS_pred[:, snap: snap + 1] = u_pred.cpu().numpy()
                V_WS_pred[:, snap: snap + 1] = v_pred.cpu().numpy()
                P_WS_pred[:, snap: snap + 1] = p_pred.cpu().numpy()

            # Pred en WS de validación
            for snap in range(0, T_val.shape[1]):
                t_out = to_torch(T_val[:, snap: snap + 1])
                x_out = to_torch(X_val[:, snap: snap + 1])
                y_out = to_torch(Y_val[:, snap: snap + 1])
                X_out = torch.cat([t_out, x_out, y_out], dim=1)
                Y_out = model(X_out)
                u_pred, v_pred, p_pred = Y_out[:, 0:1], Y_out[:, 1:2], Y_out[:, 2:3]
                U_val_pred[:, snap: snap + 1] = u_pred.cpu().numpy()
                V_val_pred[:, snap: snap + 1] = v_pred.cpu().numpy()
                P_val_pred[:, snap: snap + 1] = p_pred.cpu().numpy()

        # Guardado .mat
        scipy.io.savemat('Brussels_%s_lambda_%s_R_%s_envelope_torch.mat' % (str(epoch + 1), str(lamb), str(R)),
                         {'T_PINN': T_PINN, 'X_PINN': X_PINN, 'Y_PINN': Y_PINN, 'U_PINN': U_PINN, 'V_PINN': V_PINN, 'P_PINN': P_PINN,
                          'T_WS': T_WS, 'X_WS': X_WS, 'Y_WS': Y_WS, 'U_WS': U_WS, 'V_WS': V_WS, 'P_WS': P_WS,
                          'U_WS_pred': U_WS_pred, 'V_WS_pred': V_WS_pred, 'P_WS_pred': P_WS_pred,
                          'T_val': T_val, 'X_val': X_val, 'Y_val': Y_val, 'U_val': U_val, 'V_val': V_val, 'P_val': P_val,
                          'U_val_pred': U_val_pred, 'V_val_pred': V_val_pred, 'P_valt_pred': P_val_pred,
                          'Train_loss': np.array(train_loss_results), 'NS_loss': np.array(NS_loss_results),
                          'P_loss': np.array(P_loss_results), 'U_loss': np.array(U_loss_results), 'V_loss': np.array(V_loss_results)})

        # Guardado del modelo (estado y checkpoint)
        model_filename = 'PINN_model_epoch_%s_lambda_%s_R_%s_torch' % (str(epoch + 1), str(lamb), str(R))
        torch.save(model.state_dict(), model_filename + "_state_dict.pth")
        torch.save({'model': model, 'state_dict': model.state_dict()}, model_filename + "_full.pth")
        print(f"Modelo guardado en: {model_filename}_*.pth")

print('Process completed')

# -------------------------
# (Opcional) Visualización en PyTorch/Matplotlib (comentado)
# import matplotlib.pyplot as plt
# plt.figure(); plt.plot(train_loss_results); plt.title('Train Loss'); plt.show()
# -------------------------
