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
X_WS = WS_data["X_WS"]
Y_WS = WS_data["Y_WS"]
U_WS = WS_data["U_WS"]
V_WS = WS_data["V_WS"]
P_WS = WS_data["P_WS"]
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

# %%
print(f"{np.nanmax(T_WS)=} , {np.nanmin(T_WS)=}")
print(f"{np.nanmax(P_WS)=} , {np.nanmin(P_WS)=}")
print(f"{np.nanmax(U_WS)=} , {np.nanmin(U_WS)=}")
print(f"{np.nanmax(T_WS)=} , {np.nanmin(T_WS)=}")
print(f"{np.nanmax(X_WS)=} , {np.nanmin(X_WS)=}")
print(f"{np.nanmax(Y_WS)=} , {np.nanmin(Y_WS)=}")

# %% [markdown]
# # Utilidades

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"{device=}")

# %%
# Training
num_epochs = 1000 # number of epochs
lamb = 2 # Tuning of physics constraints
# dtype = np.float64
dtype = np.float32

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
        self.t = torch.from_numpy(t.astype(dtype))
        self.x = torch.from_numpy(x.astype(dtype))
        self.y = torch.from_numpy(y.astype(dtype))
        self.target = torch.from_numpy(tgt.astype(dtype))

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
        self.t = torch.from_numpy(t.astype(dtype))
        self.x = torch.from_numpy(x.astype(dtype))
        self.y = torch.from_numpy(y.astype(dtype))

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
        self.t = torch.from_numpy(t.astype(dtype))
        self.x = torch.from_numpy(x.astype(dtype))
        self.y = torch.from_numpy(y.astype(dtype))
        # print(self.t.shape)
        # print(self.x.shape)
        # print(self.y.shape)
        # print(self.t.shape[0])
        # print(torch.concat([self.t, self.x, self.y], axis=1))

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
#         hidden_sizes = (2 * (num_input_variables + num_output_variables))*[neurons]
#         layers = []
#         last = num_input_variables
#         for h in hidden_sizes + [num_output_variables]:
#                      layers.append(GammaBiasLayer(last, h))
#                      last = h
#         self.layers = nn.ModuleList(layers)
#         self.activation = nn.Tanh()

#     def forward(self, x):
#         # Aplicar activación solo en las primeras 5 capas ocultas
#         for i, layer in enumerate(self.layers):
#             x = layer(x)
#             if i < 8:  # activación en capas 0 a 4
#                          x = self.activation(x)
#         return x

# %%

class PINNNet(nn.Module):
    def __init__(self, num_input_variables=3, num_output_variables=3):
        super().__init__()
        self.activation = nn.Tanh()
        self.l01 = GammaBiasLayer(3, 600)
        self.l02 = GammaBiasLayer(600, 600)
        self.l03 = GammaBiasLayer(600, 600)
        self.l04 = GammaBiasLayer(600, 600)
        self.l05 = GammaBiasLayer(600, 600)
        self.l06 = GammaBiasLayer(600, 600)
        self.l07 = GammaBiasLayer(600, 600)
        self.l08 = GammaBiasLayer(600, 600)
        self.l09 = GammaBiasLayer(600, 600)
        self.l10 = GammaBiasLayer(600, 600)
        self.l11 = GammaBiasLayer(600, 600)
        self.l12 = GammaBiasLayer(600, 600)
        self.lfi = GammaBiasLayer(600, 3)

    def forward(self, x):
        
        a01 = self.activation(self.l01(x))
        a02 = self.activation(self.l02(a01))
        a03 = self.activation(self.l03(a02))
        a04 = self.activation(self.l04(a03))
        a05 = self.activation(self.l05(a04))
        a06 = self.activation(self.l06(a05))
        a07 = self.activation(self.l07(a06))
        a08 = self.activation(self.l08(a07))
        a09 = self.l09(a08)
        a10 = self.l10(a09)
        a11 = self.l11(a10)
        a12 = self.l12(a11)
        afi = self.lfi(a12)


        return afi, (a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, afi, )



# %%
# model = PINNNet().to(device).double()
model = PINNNet().to(device)

# %%
#  'named_parameters',
#  'parameters',
def info_model(model):
    for i in model.named_parameters(): 
        print(20*"#")
        print(f"Parámetro: {i[0]:10}")
        print(f"Dim parámetro: {i[1].shape}")
        print(f"Dim parámetro: {i[1][:5]}")
        print("")

# %%
info_model(model)

# %% [markdown]
# # Funciones de perdidas

# %%
mse_loss = nn.MSELoss()
# mse_loss = nn.MSELoss(reduction="mean")

# @torch.enable_grad()
# def loss_NS_2D(model, t_eqns, x_eqns, y_eqns):
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
    Y, actv = model(X)                                     # [N, 3]
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

# def loss_u(t_b, x_b, y_b, u_b):
def loss_u(model, t_b, x_b, y_b, u_b):
    X = torch.cat([t_b, x_b, y_b], dim=1)
    Y, actv = model(X)
    u_pred = Y[:, 0:1]
    for i, a in enumerate(actv):
        print(f"Capa {i:02d}: shape={tuple(a.shape)}, mean={a.mean():+.4f}, std={a.std():.4f}")
    # print(f"{X[:5,:]=}")
    # print(f"{Y[:5,:]=}")
    # print(f"{_safe_std(u_b) ** 2:=.2f}")
    return mse_loss(u_pred, u_b) / (_safe_std(u_b) ** 2)

# def loss_v(t_b, x_b, y_b, v_b):
def loss_v(model, t_b, x_b, y_b, v_b):
    X = torch.cat([t_b, x_b, y_b], dim=1)
    Y, actv = model(X)
    v_pred = Y[:, 1:2]
    return mse_loss(v_pred, v_b) / (_safe_std(v_b) ** 2)

# def loss_p(t_b, x_b, y_b, p_b):
def loss_p(model, t_b, x_b, y_b, p_b):
    X = torch.cat([t_b, x_b, y_b], dim=1)
    Y, actv = model(X)
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

    return (NS_eqns ** 2 + NS_data ** 2 + U_e ** 2 + V_e ** 2 + P_e ** 2) / total_e
    # return total_e

# %% [markdown]
# # Gradiente y optimizador

# %%
model_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# %%
scaler = torch.amp.GradScaler("cuda")

def grad_amp(model, model_optimizer, scaler,
             t_u_batch, x_u_batch, y_u_batch, u_u_batch,
             t_v_batch, x_v_batch, y_v_batch, v_v_batch,
             t_p_batch, x_p_batch, y_p_batch, p_p_batch,
             t_eqns_ref_batch, x_eqns_ref_batch, y_eqns_ref_batch,
             t_eqns_batch, x_eqns_batch, y_eqns_batch,
             lamb):
    
    use_amp = False
    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
        loss_value = loss_total(model,
                                t_u_batch, x_u_batch, y_u_batch, u_u_batch,
                                t_v_batch, x_v_batch, y_v_batch, v_v_batch,
                                t_p_batch, x_p_batch, y_p_batch, p_p_batch,
                                t_eqns_ref_batch, x_eqns_ref_batch, y_eqns_ref_batch,
                                t_eqns_batch, x_eqns_batch, y_eqns_batch,
                                lamb)

    # Escalar la pérdida antes del backward
    scaler.scale(loss_value).backward()

    # Revisa gradientes ANTES del step
    all_finite = True
    for n, p in model.named_parameters():
        if p.grad is None: 
            continue
        if not torch.isfinite(p.grad).all():
            print(f"[WARN] grad no finito en {n}")
            all_finite = False
            break

    # (Opcional) Clipping y chequeo de gradientes
    scaler.unscale_(model_optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Paso del optimizador de forma segura
    scaler.step(model_optimizer)

    # Actualiza el factor de escala automáticamente
    scaler.update()

    # # Si quieres inspeccionar gradientes:
    # grads = [p.grad.detach().clone() if p.grad is not None else None
    #          for p in model.parameters() if p.requires_grad]

    return loss_value.detach()#, grads

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
def train(loader_u, loader_v, loader_p, loader_eqns_ref, loader_eqns):

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


        model_optimizer.zero_grad(set_to_none=True)

        use_amp = False
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            loss_value = loss_total(model, t_u_b, x_u_b, y_u_b, u_u_b,
                                    t_v_b, x_v_b, y_v_b, v_v_b,
                                    t_p_b, x_p_b, y_p_b, p_p_b,
                                    t_eq_ref_b, x_eq_ref_b, y_eq_ref_b,
                                    t_eq_b, x_eq_b, y_eq_b,
                                    lamb)

        # Escalar la pérdida antes del backward
        scaler.scale(loss_value).backward()

        # Revisa gradientes ANTES del step
        all_finite = True
        for n, p in model.named_parameters():
            if p.grad is None: 
                continue
            if not torch.isfinite(p.grad).all():
                print(f"[WARN] grad no finito en {n}")
                all_finite = False
                break

        # (Opcional) Clipping y chequeo de gradientes
        scaler.unscale_(model_optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Paso del optimizador de forma segura
        scaler.step(model_optimizer)

        # Actualiza el factor de escala automáticamente
        scaler.update()
        
        return loss_value

# %%
def eval(loader_u, loader_v, loader_p, loader_eqns_ref, loader_eqns):
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
        
        NS_loss = loss_NS_2D(model, t_eq_b, x_eq_b, y_eq_b)
        
        with torch.inference_mode():
            
            U_loss  = loss_u(model, t_u_b, x_u_b, y_u_b, u_u_b)
            V_loss  = loss_v(model, t_v_b, x_v_b, y_v_b, v_v_b)
            P_loss  = loss_p(model, t_p_b, x_p_b, y_p_b, p_p_b)
            
        return NS_loss, U_loss, V_loss, P_loss
            

        

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

    # --------- ENTRENAMIENTO ---------
    model.train()
    loss_train = train(loader_u, loader_v, loader_p, loader_eqns_ref, loader_eqns)
    
     # --------- VALIDACIÓN (SIN GRADIENTES) ---------
    model.eval()
    NS_loss, U_loss, V_loss, P_loss = eval(loader_u, loader_v, loader_p, loader_eqns_ref, loader_eqns)
    
    # End epoch
    train_loss_results.append(loss_train)
    NS_loss_results.append(NS_loss)
    U_loss_results.append(U_loss)
    V_loss_results.append(V_loss)
    P_loss_results.append(P_loss)

    # Update learning rate
    new_lr = adjust_learning_rate(model_optimizer, loss_train)
    
    
    print(f"Epoch: {epoch:4} | "
          f"Loss training: {loss_train:10.4e} | "
          f"NS_Loss: {NS_loss:10.4e} | "
          f"U_Loss: {U_loss:10.4e} | "
          f"V_Loss: {V_loss:10.4e} | "
          f"P_Loss: {P_loss:10.4e} | "
          f"learning rate: {new_lr:10.4e} | "
          )

    print("\n=== GRADIENTS ===")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name:20s} | grad.mean={param.grad.mean():+.4e} | grad.std={param.grad.std():.4e}")
        else:
            print(f"{name:20s} | grad=None")

    info_model(model)
    input()


