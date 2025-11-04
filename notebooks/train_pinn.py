# %%

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import scipy.io as sio
from pathlib import Path
# from torch.nn.utils import weight_norm
from torch.nn.utils.parametrizations import weight_norm
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# %%
# ! python --version

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando PyTorch {torch.__version__} | GPU disponible: {torch.cuda.is_available()}")

# %%
# (Opcional) ligeras optimizaciones en GPU
torch.backends.cudnn.benchmark = True

# %% [markdown]
# # Rutas

# %%
BASE_DIR = Path().resolve().parents[0]
DATA_PROC_DIR = BASE_DIR / "data" / "processed"

# %% [markdown]
# # Carga de datos

# %%
path_data = DATA_PROC_DIR / "weather_data_processed.mat"
WS_data = sio.loadmat(path_data)

# %%
# Malla PINN
X_PINN = WS_data["X_PINN"]
Y_PINN = WS_data["Y_PINN"]
T_PINN = WS_data["T_PINN"]
# Data WS
T_WS = WS_data["T_WS"]
P_WS = WS_data["P_WS"]
U_WS = WS_data["U_WS"]
V_WS = WS_data["V_WS"]
X_WS = WS_data["X_WS"]
Y_WS = WS_data["Y_WS"]
# Data val
WS_val = WS_data["WS_val"]
T_val = WS_data["T_val"]
P_val = WS_data["P_val"]
U_val = WS_data["U_val"]
V_val = WS_data["V_val"]
X_val = WS_data["X_val"]
Y_val = WS_data["Y_val"]
Z_val = WS_data["Z_val"]

L = WS_data["L"]
W = WS_data["W"]
P0 = WS_data["P0"]
Re = WS_data["Re"]

batch_PINN = WS_data["batch_PINN"][0][0]
batch_WS = WS_data["batch_WS"][0][0]

# %%
batch_PINN#.shape

# %% [markdown]
# # Utilidades

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"{device=}")

# %%
# Training
num_epochs = 1000 # number of epochs
lamb = 2 # Tuning of physics constraints

# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# %%
def _flatten_grid(T, X, Y, target=None):
    """
    Recibe matrices 2D (N_x by N_t) y devuelve vectores columna aplanados (N_x*N_t, 1).
    Aplica máscara de NaN con base en `target` si se provee.
    """
    # Aplanar
    t = T.reshape(-1, 1)
    x = X.reshape(-1, 1)
    y = Y.reshape(-1, 1)
    tgt = None if target is None else target.reshape(-1, 1)

    # Máscara de NaN según target si existe
    if tgt is not None:
        mask = ~np.isnan(tgt[:, 0])
        t, x, y = t[mask], x[mask], y[mask]
        tgt = tgt[mask]
    return t, x, y, tgt

# %%
def move_to_device(batch, device):
    if isinstance(batch, (list, tuple)):
        return [item.to(device, non_blocking=True) for item in batch]
    elif isinstance(batch, dict):
        return {key: val.to(device, non_blocking=True) for key, val in batch.items()}
    else:
        return batch.to(device, non_blocking=True)

# %% [markdown]
# # Datasets

# %%
class WSDataset(Dataset):
    """
    Dataset para un campo escalar (u, v o p) en las ubicaciones WS (t, x, y [,z]).
    Permite barajar con DataLoader + RandomSampler.
    """
    def __init__(self, T_WS, X_WS, Y_WS, target_WS):
        t, x, y, tgt = _flatten_grid(T_WS, X_WS, Y_WS, target_WS)
        self.t = torch.from_numpy(t.astype(np.float64))
        self.x = torch.from_numpy(x.astype(np.float64))
        self.y = torch.from_numpy(y.astype(np.float64))
        self.target = torch.from_numpy(tgt.astype(np.float64))

    def __len__(self):
        return self.t.shape[0]

    def __getitem__(self, idx):
        # return torch.concat([self.t[idx], self.x[idx], self.y[idx], self.target[idx]], axis=1)
        return self.t[idx], self.x[idx], self.y[idx], self.target[idx]


# %%
class WSEqnRefDataset(Dataset):
    """
    Dataset para (t, x, y) de referencia desde WS (para términos de ecuaciones que usan ref).
    """
    def __init__(self, T_WS, X_WS, Y_WS):
        t, x, y, _ = _flatten_grid(T_WS, X_WS, Y_WS)
        self.t = torch.from_numpy(t.astype(np.float64))
        self.x = torch.from_numpy(x.astype(np.float64))
        self.y = torch.from_numpy(y.astype(np.float64))

    def __len__(self):
        return self.t.shape[0]

    def __getitem__(self, idx):
        # return torch.concat([self.t[idx], self.x[idx], self.y[idx]], axis=1)
        return self.t[idx], self.x[idx], self.y[idx]

# %%
class PINNEqnDataset(Dataset):
    """
    Dataset de puntos de colación (PINN) provenientes de malla PINN (sin targets).
    """
    def __init__(self, T_PINN, X_PINN, Y_PINN):
        t, x, y, _ = _flatten_grid(T_PINN, X_PINN, Y_PINN)
        self.t = torch.from_numpy(t.astype(np.float64))
        self.x = torch.from_numpy(x.astype(np.float64))
        self.y = torch.from_numpy(y.astype(np.float64))
        print(self.t.shape)
        print(self.x.shape)
        print(self.y.shape)
        print(self.t.shape[0])
        print(torch.concat([self.t, self.x, self.y], axis=1))

    def __len__(self):
        return self.t.shape[0]

    def __getitem__(self, idx):
        # return torch.concat([self.t[idx,:], self.x[idx,:], self.y[idx,:]], axis=1)
        return self.t[idx], self.x[idx], self.y[idx]

# %% [markdown]
# # Preparación de datasets y dataloaders

# %%
# Dimensions
dim_N_WS = X_WS.shape[0]
dim_T_WS = X_WS.shape[1]
dim_N_PINN = X_PINN.shape[0]
dim_T_PINN = T_PINN.shape[1]

# %%
# Dimensions
dim_N_data = dim_N_WS
dim_T_data = dim_T_WS
dim_T_eqns = dim_T_PINN
dim_N_eqns = dim_N_PINN

num_samples_WS = int(dim_N_data * dim_T_data)
num_samples_PINN = int(dim_N_eqns * dim_T_eqns)

batch_PINN = int(batch_PINN)
batch_WS = int(batch_WS)

print(f"{batch_PINN=}, {batch_WS=}")

# %%
# Conjuntos
ds_u = WSDataset(T_WS, X_WS, Y_WS, U_WS)
ds_v = WSDataset(T_WS, X_WS, Y_WS, V_WS)
ds_p = WSDataset(T_WS, X_WS, Y_WS, P_WS)

ds_eqns_ref = WSEqnRefDataset(T_WS, X_WS, Y_WS)
ds_eqns = PINNEqnDataset(T_PINN, X_PINN, Y_PINN)

# %%
# Muestreo aleatorio por época
sampler_u = RandomSampler(ds_u, replacement=False, num_samples=num_samples_WS)
sampler_v = RandomSampler(ds_v, replacement=False, num_samples=num_samples_WS)
sampler_p = RandomSampler(ds_p, replacement=False, num_samples=num_samples_WS)
sampler_eqns_ref = RandomSampler(ds_eqns_ref, replacement=False, num_samples=num_samples_WS)
sampler_eqns = RandomSampler(ds_eqns, replacement=False, num_samples=num_samples_PINN)

# %%
# Carga de los dataloaders con el muestreo simple
loader_u = DataLoader(ds_u, batch_size=batch_WS, sampler=sampler_u, pin_memory=True)
loader_v = DataLoader(ds_v, batch_size=batch_WS, sampler=sampler_v, pin_memory=True)
loader_p = DataLoader(ds_p, batch_size=batch_WS, sampler=sampler_p, pin_memory=True)
loader_eqns_ref = DataLoader(ds_eqns_ref, batch_size=batch_WS, sampler=sampler_eqns_ref, pin_memory=True)
loader_eqns = DataLoader(ds_eqns, batch_size=batch_PINN, sampler=sampler_eqns, pin_memory=True)


# %%
# # Se cargan los batches a device
# for loader in [loader_u,loader_v,loader_p,loader_eqns_ref,loader_eqns,]:
#     for batch in loader:
#         batch = move_to_device(batch, device)

# %% [markdown]
# # Capa personalizada: GammaBiasLayer

# %%
class GammaBiasLayer(nn.Module):
    """
    Capa densa personalizada:
      y = gamma ⊙ (W_norm x) + bias
    - W_norm: Linear (sin bias) con Weight Normalization
    - gamma: parámetro de escala por-neurona
    - bias: sesgo por-neurona

    Args:
        in_features  (int): tamaño de entrada
        out_features (int): número de unidades (neuronas)
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # Linear sin bias para emular Dense(use_bias=False)
        linear = nn.Linear(in_features, out_features, bias=False)
        # Inicialización uniforme [-1, 1], como en tu RandomUniform
        nn.init.uniform_(linear.weight, a=-1.0, b=1.0)

        # Weight Normalization (equivalente a tfa.layers.WeightNormalization)
        self.w = weight_norm(linear)  # añade weight_g y weight_v internamente

        # Parámetros gamma y bias (forma [out_features])
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.bias  = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_features) -> out: (batch, out_features)
        y = self.w(x)
        # Broadcasting de gamma y bias sobre la dimensión batch
        return y * self.gamma + self.bias

# %% [markdown]
# # PINN

# %%
# class PINNNet(nn.Module):
#     def __init__(self, num_input_variables=3, num_output_variables=3):
#         super().__init__()

#         neurons = 200 * num_output_variables
#         layers_sizes = (
#             [num_input_variables]
#             + (2 * (num_input_variables + num_output_variables)) * [neurons]
#             + [num_output_variables]
#         )
#         # Guardamos para reproducir los mismos rangos de tu for
#         L = layers_sizes
#         # Índices de particiones como en tu código
#         mid_end = 2 * int((len(L) - 2) / 3)

#         # Construimos los módulos siguiendo el mismo patrón
#         mods = []

#         # Primer bloque: GammaBias(layers[1]) + tanh
#         mods.append(GammaBiasLayer(L[0], L[1]))
#         mods.append(nn.Tanh())

#         # Bloques intermedios: for l in layers[2 : mid_end]: GammaBias(l) + tanh
#         in_dim = L[1]
#         for l in L[2:mid_end]:
#             mods.append(GammaBiasLayer(in_dim, l))
#             mods.append(nn.Tanh())
#             in_dim = l

#         # Bloques finales (antes de salida): for l in layers[mid_end : -1]:
#         #   GammaBias(layers[-2])  (tal como en tu código original)
#         # Esto apila capas con anchura fija igual a L[-2]
#         penultimate = L[-2]
#         for _ in L[mid_end:-1]:
#             mods.append(GammaBiasLayer(in_dim, penultimate))
#             # OJO: el original no aplicaba activación aquí
#             in_dim = penultimate

#         # Capa de salida: GammaBias(layers[-1])
#         mods.append(GammaBiasLayer(in_dim, L[-1]))

#         self.net = nn.Sequential(*mods)

#     def forward(self, x):
#         # x: [N, 3] -> [N, 3]  (u, v, p)
#         return self.net(x)

# %%

class PINNNet(nn.Module):
    def __init__(self, num_input_variables=3, num_output_variables=3):
        super().__init__()
        neurons = 200 * num_output_variables 
        hidden_sizes = (2 * (num_input_variables + num_output_variables))*[neurons]
        layers = []
        last = num_input_variables
        for h in hidden_sizes + [num_output_variables]:
            layers.append(GammaBiasLayer(last, h))
            last = h
        self.layers = nn.ModuleList(layers)
        self.activation = nn.Tanh()

    def forward(self, x):
        len1 = int(len(self.layers)/3)
        len2 = int(2*len(self.layers)/3)
        for layer in self.layers[:len1]:
            x = self.activation(layer(x))
        for layer in self.layers[len1:]:
            x = layer(x)
        return x

# %%
model = PINNNet().to(device)

# %% [markdown]
# # Funciones de perdidas

# %%
mse_loss = nn.MSELoss()
# mse_loss = nn.MSELoss(reduction="mean")

# @torch.enable_grad()
def loss_NS_2D(model, t_eqns, x_eqns, y_eqns):
    """
    Calcula los residuales 2D (incompresible) aproximados:
      e1 = u_x + v_y
      e2 = u_t + (u u_x + v u_y) + p_x
      e3 = v_t + (u v_x + v v_y) + p_y
    Devuelve MSE(0, e1) + MSE(0, e2) + MSE(0, e3)
    """
    # Asegurar gradientes con respecto a entradas
    for ten in (t_eqns, x_eqns, y_eqns):
        ten.requires_grad_(True)

    X = torch.cat([t_eqns, x_eqns, y_eqns], dim=1)  # [N, 3]
    Y = model(X)                                     # [N, 3]
    u, v, p = Y[:, 0:1], Y[:, 1:2], Y[:, 2:3]

    ones_u = torch.ones_like(u)
    ones_v = torch.ones_like(v)
    ones_p = torch.ones_like(p)

    # Derivadas primeras
    u_t = torch.autograd.grad(u, t_eqns, grad_outputs=ones_u, create_graph=True)[0]
    v_t = torch.autograd.grad(v, t_eqns, grad_outputs=ones_v, create_graph=True)[0]

    u_x = torch.autograd.grad(u, x_eqns, grad_outputs=ones_u, create_graph=True)[0]
    v_x = torch.autograd.grad(v, x_eqns, grad_outputs=ones_v, create_graph=True)[0]
    p_x = torch.autograd.grad(p, x_eqns, grad_outputs=ones_p, create_graph=True)[0]

    u_y = torch.autograd.grad(u, y_eqns, grad_outputs=ones_u, create_graph=True)[0]
    v_y = torch.autograd.grad(v, y_eqns, grad_outputs=ones_v, create_graph=True)[0]
    p_y = torch.autograd.grad(p, y_eqns, grad_outputs=ones_p, create_graph=True)[0]

    # Residuales Navier–Stokes (sin términos viscosos, conforme a tu snippet)
    e1 = (u_x + v_y)
    e2 = (u_t + (u * u_x + v * u_y) + p_x)
    e3 = (v_t + (u * v_x + v * v_y) + p_y)

    zero = torch.zeros_like(e1)
    return (
        mse_loss(e1, zero) +
        mse_loss(e2, zero) +
        mse_loss(e3, zero)
    )

# @torch.no_grad()
def _safe_std(x: torch.Tensor, eps: float = 1e-8):
    # std con protección para divisiones por cero
    s = torch.std(x)
    return s.clamp_min(eps)

def loss_u(model, t_b, x_b, y_b, u_b):
    X = torch.cat([t_b, x_b, y_b], dim=1)
    Y = model(X)
    u_pred = Y[:, 0:1]
    return mse_loss(u_pred, u_b) / (_safe_std(u_b) ** 2)

def loss_v(model, t_b, x_b, y_b, v_b):
    X = torch.cat([t_b, x_b, y_b], dim=1)
    Y = model(X)
    v_pred = Y[:, 1:2]
    return mse_loss(v_pred, v_b) / (_safe_std(v_b) ** 2)

def loss_p(model, t_b, x_b, y_b, p_b):
    X = torch.cat([t_b, x_b, y_b], dim=1)
    Y = model(X)
    p_pred = Y[:, 2:3]
    return mse_loss(p_pred, p_b) / (_safe_std(p_b) ** 2)

def loss_total(
    model,
    # datos u
    t_u_b, x_u_b, y_u_b, u_u_b,
    # datos v
    t_v_b, x_v_b, y_v_b, v_v_b,
    # datos p
    t_p_b, x_p_b, y_p_b, p_p_b,
    # ecuaciones (referencia + ecuaciones)
    t_eqns_ref_b, x_eqns_ref_b, y_eqns_ref_b,
    t_eqns_b, x_eqns_b, y_eqns_b,
    lamb: float
):
    NS_eqns = lamb * loss_NS_2D(model, t_eqns_b,     x_eqns_b,     y_eqns_b,   )
    NS_data = lamb * loss_NS_2D(model, t_eqns_ref_b, x_eqns_ref_b, y_eqns_ref_b)
    P_e = loss_p(model, t_p_b, x_p_b, y_p_b, p_p_b)
    U_e = loss_u(model, t_u_b, x_u_b, y_u_b, u_u_b)
    V_e = loss_v(model, t_v_b, x_v_b, y_v_b, v_v_b)

    total_e = NS_eqns + NS_data + U_e + V_e + P_e
    # Misma forma de tu retorno: suma de cuadrados/total
#     print(f"""
# {NS_eqns=}
# {NS_data=}
# {U_e=}
# {V_e=}
# {P_e=}
# {total_e=}
#           """)
    # return total_e
    return (NS_eqns ** 2 + NS_data ** 2 + U_e ** 2 + V_e ** 2 + P_e ** 2) / total_e

# %% [markdown]
# # Gradiente y optimizador

# %%
model_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# %%
scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
# scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

def grad_amp(model, model_optimizer, scaler,
             t_u_batch, x_u_batch, y_u_batch, u_u_batch,
             t_v_batch, x_v_batch, y_v_batch, v_v_batch,
             t_p_batch, x_p_batch, y_p_batch, p_p_batch,
             t_eqns_ref_batch, x_eqns_ref_batch, y_eqns_ref_batch,
             t_eqns_batch, x_eqns_batch, y_eqns_batch,
             lamb):
    model.train()
    model_optimizer.zero_grad(set_to_none=True)

    # with torch.cuda.amp.autocast(enabled=False):
    with torch.amp.autocast(dtype=torch.float64, device_type=device.type):
    # with torch.cuda.amp.autocast(dtype=torch.float64):
        loss_value = loss_total(model,
                                t_u_batch, x_u_batch, y_u_batch, u_u_batch,
                                t_v_batch, x_v_batch, y_v_batch, v_v_batch,
                                t_p_batch, x_p_batch, y_p_batch, p_p_batch,
                                t_eqns_ref_batch, x_eqns_ref_batch, y_eqns_ref_batch,
                                t_eqns_batch, x_eqns_batch, y_eqns_batch,
                                lamb)

    scaler.scale(loss_value).backward()
    scaler.step(model_optimizer)
    scaler.update()

    # Si quieres inspeccionar gradientes:
    grads = [p.grad.detach().clone() if p.grad is not None else None
             for p in model.parameters() if p.requires_grad]

    return loss_value.detach(), grads

# %%
# def grad(model,
#          t_u_batch, x_u_batch, y_u_batch, u_u_batch,
#          t_v_batch, x_v_batch, y_v_batch, v_v_batch,
#          t_p_batch, x_p_batch, y_p_batch, p_p_batch,
#          t_eqns_ref_batch, x_eqns_ref_batch, y_eqns_ref_batch,
#          t_eqns_batch, x_eqns_batch, y_eqns_batch,
#          lamb):
    
#     model.train()  # Asegura que el modelo esté en modo entrenamiento
#     model.zero_grad()  # Limpia gradientes anteriores

#     # Forward pass
#     loss_value = loss_total(model,
#                             t_u_batch, x_u_batch, y_u_batch, u_u_batch,
#                             t_v_batch, x_v_batch, y_v_batch, v_v_batch,
#                             t_p_batch, x_p_batch, y_p_batch, p_p_batch,
#                             t_eqns_ref_batch, x_eqns_ref_batch, y_eqns_ref_batch,
#                             t_eqns_batch, x_eqns_batch, y_eqns_batch,
#                             lamb,
#                             training=True)

#     # Backward pass
#     loss_value.backward()

#     # Extraer gradientes
#     gradient_model = [p.grad.clone() if p.grad is not None else None
#                       for p in model.parameters() if p.requires_grad]

#     return loss_value.detach(), gradient_model


# %% [markdown]
# # Helpers de métricas simples por época (media acumulada)

# %%
class RunningMean:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        # value puede ser tensor o float
        v = float(value) if torch.is_tensor(value) else float(value)
        self.sum += v * n
        self.count += n

    @property
    def result(self):
        return self.sum / max(self.count, 1)

# %% [markdown]
# # Entrenamiento

# %%
print(device)

# %%
def adjust_learning_rate(optimizer, epoch_loss):
    if epoch_loss > 1e-1:
        new_lr = 1e-3
    elif epoch_loss > 3e-2:
        new_lr = 1e-4
    elif epoch_loss > 3e-3:
        new_lr = 1e-5
    else:
        new_lr = 1e-6

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    return new_lr

# %%
def train_eval_model(loader_u, loader_v, loader_p, loader_eqns_ref, loader_eqns):

    for idx,(\
        (t_u_b, x_u_b, y_u_b, u_u_b), \
        (t_v_b, x_v_b, y_v_b, v_v_b), \
        (t_p_b, x_p_b, y_p_b, p_p_b), \
        (t_eq_ref_b, x_eq_ref_b, y_eq_ref_b), \
        (t_eq_b, x_eq_b, y_eq_b)) in enumerate(zip(loader_u, loader_v, loader_p, loader_eqns_ref, loader_eqns)):

        # Se envian tensores a device
        t_u_b = t_u_b.to(device)
        x_u_b = x_u_b.to(device)
        y_u_b = y_u_b.to(device)
        u_u_b = u_u_b.to(device)
        t_v_b = t_v_b.to(device)
        x_v_b = x_v_b.to(device)
        y_v_b = y_v_b.to(device)
        v_v_b = v_v_b.to(device)
        t_p_b = t_p_b.to(device)
        x_p_b = x_p_b.to(device)
        y_p_b = y_p_b.to(device)
        p_p_b = p_p_b.to(device)
        t_eq_ref_b = t_eq_ref_b.to(device)
        x_eq_ref_b = x_eq_ref_b.to(device)
        y_eq_ref_b = y_eq_ref_b.to(device)
        t_eq_b = t_eq_b.to(device)
        x_eq_b = x_eq_b.to(device)
        y_eq_b = y_eq_b.to(device)

        with torch.enable_grad():
            loss_train, grads = grad_amp(model, model_optimizer, scaler,
                                         t_u_b, x_u_b, y_u_b, u_u_b,
                                         t_v_b, x_v_b, y_v_b, v_v_b,
                                         t_p_b, x_p_b, y_p_b, p_p_b,
                                         t_eq_ref_b, x_eq_ref_b, y_eq_ref_b,
                                         t_eq_b, x_eq_b, y_eq_b, lamb)
            epoch_loss_avg.update(loss_train)


        model.eval()
        # with torch.no_grad():
        NS_loss = loss_NS_2D(model, t_eq_b, x_eq_b, y_eq_b)
        P_loss = loss_u(model, t_u_b, x_u_b, y_u_b, u_u_b)
        U_loss = loss_v(model, t_v_b, x_v_b, y_v_b, v_v_b)
        V_loss = loss_p(model, t_p_b, x_p_b, y_p_b, p_p_b)

        epoch_NS_loss_avg.update(NS_loss)
        epoch_P_loss_avg.update(P_loss)
        epoch_U_loss_avg.update(U_loss)
        epoch_V_loss_avg.update(V_loss)

        return loss_train, NS_loss, P_loss, U_loss, V_loss
        
        
        # # End epoch
        # epoch_loss_avg.update(loss_train)
        # epoch_NS_loss_avg.update(NS_loss)
        # epoch_P_loss_avg.update(P_loss)
        # epoch_U_loss_avg.update(U_loss)
        # epoch_V_loss_avg.update(V_loss)





    ...

# %%
# Obten la mejor pérdida 
major_loss_validation = float('inf')

# Keep results for plotting
train_loss_results = []
NS_loss_results = []
P_loss_results = []
U_loss_results = []
V_loss_results = []

for epoch in range(1, num_epochs + 1):

    # Inicializamos registros de las funciones de perdida
    epoch_loss_avg = RunningMean()
    epoch_NS_loss_avg = RunningMean()
    epoch_P_loss_avg = RunningMean()
    epoch_U_loss_avg = RunningMean()
    epoch_V_loss_avg = RunningMean()

    # Train
    loss_train, NS_loss, P_loss, U_loss, V_loss = train_eval_model(loader_u, loader_v, loader_p, loader_eqns_ref, loader_eqns)
    # End epoch
    train_loss_results.append(epoch_loss_avg.result)
    NS_loss_results.append(epoch_NS_loss_avg.result)
    P_loss_results.append(epoch_P_loss_avg.result)
    U_loss_results.append(epoch_U_loss_avg.result)
    V_loss_results.append(epoch_V_loss_avg.result)

    # Update learning rate
    new_lr = adjust_learning_rate(model_optimizer, loss_train)
    
    print(f"Epoch: {epoch:4} | "
          f"Loss training: {loss_train:10.4e} | "
          f"NS_Loss: {NS_loss:10.4e} | "
          f"P_Loss: {P_loss:10.4e} | "
          f"U_Loss: {U_loss:10.4e} | "
          f"V_Loss: {V_loss:10.4e} | "
          f"learning rate: {new_lr:10.4e} | "
          )


# %%
# print(f"|{0.0001234:10.4e}|")

# %%
# for epoch in range(10):
#     for (t_u,x_u,y_u,u_u), (t_v,x_v,y_v,v_v) in zip(loader_u, loader_v):
#         # print(f"{x_u.shape=}")
#         # print(f"{y_u=}")
#         # print(f"{t_u=}")
#         # print(f"{u_u=}")
#         # print(torch.concat([t_u,x_u,y_u,u_u], axis=1))
#         # print("*"*10)
#         # print(f"{x_v.shape=}")
#         # print(f"{y_v=}")
#         # print(f"{t_v=}")
#         # print(f"{v_v=}")
#         print(torch.concat([t_v,x_v,y_v,v_v], axis=1)[:5,:])


#         ans = input("stop?")
#         if ans == "y":
#             break
#     if ans == "y":
#         break

# %%
# train_loss_results = []
# NS_loss_results = []
# P_loss_results = []
# U_loss_results = []
# V_loss_results = []

# for epoch in range(num_epochs):
#     model.train()
#     epoch_loss_avg = RunningMean()
#     epoch_NS_loss_avg = RunningMean()
#     epoch_P_loss_avg = RunningMean()
#     epoch_U_loss_avg = RunningMean()
#     epoch_V_loss_avg = RunningMean()

#     # zip se detiene en el DataLoader más corto (equivale a tu min_div)
#     for (t_u_b, x_u_b, y_u_b, u_u_b), \
#         (t_v_b, x_v_b, y_v_b, v_v_b), \
#         (t_p_b, x_p_b, y_p_b, p_p_b), \
#         (t_eq_ref_b, x_eq_ref_b, y_eq_ref_b), \
#         (t_eq_b, x_eq_b, y_eq_b) in zip(loader_u, loader_v, loader_p, loader_eqns_ref, loader_eqns):

#         # Enviar a device
#         t_u_b, x_u_b, y_u_b, u_u_b = to_dev(t_u_b, x_u_b, y_u_b, u_u_b, device=device)
#         t_v_b, x_v_b, y_v_b, v_v_b = to_dev(t_v_b, x_v_b, y_v_b, v_v_b, device=device)
#         t_p_b, x_p_b, y_p_b, p_p_b = to_dev(t_p_b, x_p_b, y_p_b, p_p_b, device=device)
#         t_eq_ref_b, x_eq_ref_b, y_eq_ref_b = to_dev(t_eq_ref_b, x_eq_ref_b, y_eq_ref_b, device=device)
#         t_eq_b, x_eq_b, y_eq_b = to_dev(t_eq_b, x_eq_b, y_eq_b, device=device)

#         # Paso de entrenamiento
#         model_optimizer.zero_grad(set_to_none=True)
#         loss_train = loss_total(
#             model,
#             # u
#             t_u_b, x_u_b, y_u_b, u_u_b,
#             # v
#             t_v_b, x_v_b, y_v_b, v_v_b,
#             # p
#             t_p_b, x_p_b, y_p_b, p_p_b,
#             # eqns ref
#             t_eq_ref_b, x_eq_ref_b, y_eq_ref_b,
#             # eqns
#             t_eq_b, x_eq_b, y_eq_b,
#             lamb, training=True
#         )
#         loss_train.backward()
#         model_optimizer.step()

#         # Métricas por lote (sin gradiente, modo eval para consistencia)
#         model.eval()
#         with torch.no_grad():
#             NS_loss = loss_NS_2D(model, t_eq_b.clone(), x_eq_b.clone(), y_eq_b.clone(), training=False)
#             P_loss = loss_p(model, t_p_b, x_p_b, y_p_b, p_p_b, training=False)
#             U_loss = loss_u(model, t_u_b, x_u_b, y_u_b, u_u_b, training=False)
#             V_loss = loss_v(model, t_v_b, x_v_b, y_v_b, v_v_b, training=False)

#         # Acumular promedios
#         bs = t_u_b.shape[0]  # cualquier batch size como peso
#         epoch_loss_avg.update(loss_train, n=bs)
#         epoch_NS_loss_avg.update(NS_loss, n=bs)
#         epoch_P_loss_avg.update(P_loss, n=bs)
#         epoch_U_loss_avg.update(U_loss, n=bs)
#         epoch_V_loss_avg.update(V_loss, n=bs)

#         model.train()  # volver a train para el siguiente batch

#     # --- fin de la época: guardar resultados de métricas ---
#     train_loss_results.append(epoch_loss_avg.result)
#     NS_loss_results.append(epoch_NS_loss_avg.result)
#     P_loss_results.append(epoch_P_loss_avg.result)
#     U_loss_results.append(epoch_U_loss_avg.result)
#     V_loss_results.append(epoch_V_loss_avg.result)

#     # --- actualizar LR como en tu scheduler por umbrales ---
#     avg = epoch_loss_avg.result
#     if avg > 1e-1:
#         new_lr = 1e-3
#     elif avg > 3e-2:
#         new_lr = 1e-4
#     elif avg > 3e-3:
#         new_lr = 1e-5
#     else:
#         new_lr = 1e-6
#     for g in model_optimizer.param_groups:
#         g['lr'] = new_lr

#     print(f"Epoch: {epoch:4d} "
#           f"Loss_training: {epoch_loss_avg.result:.3e} "
#           f"NS_loss: {epoch_NS_loss_avg.result:.3e} "
#           f"P_loss: {epoch_P_loss_avg.result:.3e} "
#           f"U_loss: {epoch_U_loss_avg.result:.3e} "
#           f"V_loss: {epoch_V_loss_avg.result:.3e}  "
#           f"(lr={new_lr:.0e})")

#     # ------------------ Guardado de predicciones/modelo ------------------
#     if (epoch + 1) % num_epochs == 0:
#         model.eval()
#         with torch.no_grad():
#             # Salidas de alta resolución (PINN grid)
#             U_PINN = np.zeros_like(X_PINN)
#             V_PINN = np.zeros_like(X_PINN)
#             P_PINN = np.zeros_like(X_PINN)

#             for snap in range(0, X_PINN.shape[1]):
#                 t_out = torch.as_tensor(T_PINN[:, snap:snap+1], dtype=torch.get_default_dtype(), device=device)
#                 x_out = torch.as_tensor(X_PINN[:, snap:snap+1], dtype=torch.get_default_dtype(), device=device)
#                 y_out = torch.as_tensor(Y_PINN[:, snap:snap+1], dtype=torch.get_default_dtype(), device=device)
#                 X_out = torch.cat([t_out, x_out, y_out], dim=1)  # [N,3]
#                 Y_out = model(X_out)                              # [N,3]
#                 u_pred, v_pred, p_pred = Y_out[:,0:1], Y_out[:,1:2], Y_out[:,2:3]
#                 U_PINN[:, snap:snap+1] = u_pred.cpu().numpy()
#                 V_PINN[:, snap:snap+1] = v_pred.cpu().numpy()
#                 P_PINN[:, snap:snap+1] = p_pred.cpu().numpy()

#             # Predicciones en WS
#             U_WS_pred = np.zeros_like(X_WS)
#             V_WS_pred = np.zeros_like(X_WS)
#             P_WS_pred = np.zeros_like(X_WS)

#             for snap in range(0, X_WS.shape[1]):
#                 t_out = torch.as_tensor(T_WS[:, snap:snap+1], dtype=torch.get_default_dtype(), device=device)
#                 x_out = torch.as_tensor(X_WS[:, snap:snap+1], dtype=torch.get_default_dtype(), device=device)
#                 y_out = torch.as_tensor(Y_WS[:, snap:snap+1], dtype=torch.get_default_dtype(), device=device)
#                 X_out = torch.cat([t_out, x_out, y_out], dim=1)
#                 Y_out = model(X_out)
#                 u_pred, v_pred, p_pred = Y_out[:,0:1], Y_out[:,1:2], Y_out[:,2:3]
#                 U_WS_pred[:, snap:snap+1] = u_pred.cpu().numpy()
#                 V_WS_pred[:, snap:snap+1] = v_pred.cpu().numpy()
#                 P_WS_pred[:, snap:snap+1] = p_pred.cpu().numpy()

#             # Predicciones en validación
#             U_val_pred = np.zeros_like(X_val)
#             V_val_pred = np.zeros_like(X_val)
#             P_val_pred = np.zeros_like(X_val)

#             for snap in range(0, X_val.shape[1]):
#                 t_out = torch.as_tensor(T_val[:, snap:snap+1], dtype=torch.get_default_dtype(), device=device)
#                 x_out = torch.as_tensor(X_val[:, snap:snap+1], dtype=torch.get_default_dtype(), device=device)
#                 y_out = torch.as_tensor(Y_val[:, snap:snap+1], dtype=torch.get_default_dtype(), device=device)
#                 X_out = torch.cat([t_out, x_out, y_out], dim=1)
#                 Y_out = model(X_out)
#                 u_pred, v_pred, p_pred = Y_out[:,0:1], Y_out[:,1:2], Y_out[:,2:3]
#                 U_val_pred[:, snap:snap+1] = u_pred.cpu().numpy()
#                 V_val_pred[:, snap:snap+1] = v_pred.cpu().numpy()
#                 P_val_pred[:, snap:snap+1] = p_pred.cpu().numpy()

#         # Guardar .mat
#         scipy.io.savemat(
#             f'Brussels_{epoch+1}_lambda_{lamb}_R_{R}_envelope.mat',
#             {
#                 'T_PINN': T_PINN, 'X_PINN': X_PINN, 'Y_PINN': Y_PINN,
#                 'U_PINN': U_PINN, 'V_PINN': V_PINN, 'P_PINN': P_PINN,
#                 'T_WS': T_WS, 'X_WS': X_WS, 'Y_WS': Y_WS,
#                 'U_WS': U_WS, 'V_WS': V_WS, 'P_WS': P_WS,
#                 'U_WS_pred': U_WS_pred, 'V_WS_pred': V_WS_pred, 'P_WS_pred': P_WS_pred,
#                 'T_val': T_val, 'X_val': X_val, 'Y_val': Y_val,
#                 'U_val': U_val, 'V_val': V_val, 'P_val': P_val,
#                 'U_val_pred': U_val_pred, 'V_val_pred': V_val_pred, 'P_val_pred': P_val_pred,
#                 'Train_loss': np.array(train_loss_results, dtype=float),
#                 'NS_loss': np.array(NS_loss_results, dtype=float),
#                 'P_loss': np.array(P_loss_results, dtype=float),
#                 'U_loss': np.array(U_loss_results, dtype=float),
#                 'V_loss': np.array(V_loss_results, dtype=float),
#             }
#         )

#         # Guardar el modelo (state dict + versión trazada opcional)
#         model_filename = f'PINN_model_epoch_{epoch+1}_lambda_{lamb}_R_{R}'
#         torch.save(model.state_dict(), model_filename + ".pth")
#         try:
#             # Trazeo con un input dummy (ajusta el tamaño según tu caso)
#             dummy = torch.zeros(1, 3, device=device, dtype=torch.get_default_dtype())
#             traced = torch.jit.trace(model, dummy)
#             traced.save(model_filename + "_traced.pt")
#         except Exception as e:
#             print("Aviso: no se pudo trazar el modelo (torch.jit.trace).", e)
#         print(f"Modelo guardado en: {model_filename}.pth (y traced si fue posible)")

# %%
# 

# %%
# 

# %%
# 

# %%
# 

# %% [markdown]
# # X

# %%
# import torch
# from torch.utils.data import Dataset, DataLoader, RandomSampler

# # 1. Define a custom Dataset
# class CustomDataset(Dataset):
#     def __init__(self, data):
#         self.data = data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]

# # Create some dummy data
# dummy_data = [f"sample_{i}" for i in range(10)]
# dataset = CustomDataset(dummy_data)

# # 2. Create a RandomSampler
# # By default, replacement is False (sampling without replacement)
# # You can also specify num_samples if you want to sample a subset
# sampler = RandomSampler(dataset, replacement=False) 

# # If you want to sample with replacement and specify a number of samples:
# # sampler_with_replacement = RandomSampler(dataset, replacement=True, num_samples=20) 

# # 3. Create a DataLoader using the sampler
# # When a sampler is provided, the 'shuffle' argument in DataLoader should be False
# dataloader = DataLoader(dataset, batch_size=5, sampler=sampler)

# # 4. Iterate through the DataLoader to get random batches
# print("Randomly sampled batches:")
# for batch in dataloader:
#     print(batch)

# # Example with a fixed random seed for reproducibility
# print("\nRandomly sampled batches with fixed seed:")
# generator = torch.Generator()
# generator.manual_seed(42) # Set a seed for reproducibility
# sampler_seeded = RandomSampler(dataset, generator=generator)
# dataloader_seeded = DataLoader(dataset, batch_size=2, sampler=sampler_seeded)

# for batch in dataloader_seeded:
#     print(batch)

# %%
# np.random.choice(10, 8, replace=False)

# %%
# 


