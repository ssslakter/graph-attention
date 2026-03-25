import os
import math
import random
import wfdb
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import tqdm
import warnings

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.signal import butter, filtfilt
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.benchmark = True

def clean_ecg_signal(data, fs=360.0):
    nyq = 0.5 * fs
    low = 0.5 / nyq
    high = 45.0 / nyq
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, data)

class ECGDataset(Dataset):
    def __init__(self, records, data_dir, window=256, is_train=False):
        self.samples = []
        self.is_train = is_train
        half = window // 2
        for rec in records:
            sig, _ = wfdb.rdsamp(os.path.join(data_dir, rec))
            ann = wfdb.rdann(os.path.join(data_dir, rec), "atr")
            raw_ecg = sig[:, 0]
            
            clean_ecg = clean_ecg_signal(raw_ecg, fs=360.0)
            
            
            for pos, sym in zip(ann.sample, ann.symbol):
                if pos < half or pos + half >= len(clean_ecg):
                    continue

                segment = clean_ecg[pos-half : pos+half]
                segment = (segment - segment.mean()) / (segment.std() + 1e-8)
                
                if sym == "N":
                    label = 0
                elif sym in ["V","A","L","R","F"]:
                    label = 1
                else:
                    continue

                self.samples.append((segment.astype(np.float32), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        if self.is_train:
            shift = random.randint(-10, 10)
            if shift != 0:
                x = np.roll(x, shift)
                if shift > 0:
                    x[:shift] = 0
                else:
                    x[shift:] = 0
                    
            scale = random.uniform(0.9, 1.1)
            x = x * scale
            noise = np.random.normal(0, 0.05, size=x.shape).astype(np.float32)
            x = x + noise

        return torch.from_numpy(x), torch.tensor(y)

def sparsity_schedule(l, L, smax=0.2, smin=0.8, alpha=3.0):
    return smin + (smax - smin) * math.exp(-alpha * l / L)

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss) 
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class GraphConstructor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.scale = math.sqrt(dim) 

    def forward(self, X):
        tau = self.temperature.clamp(0.1, 5.0)
        scores = (X @ X.transpose(-1, -2)) / (self.scale * tau)
        return scores

class GraphFilter(nn.Module):
    def __init__(self, dim, K, separate_W=True):
        super().__init__()
        self.K = K
        self.separate_W = separate_W
        self.alpha_logits = nn.Parameter(torch.zeros(K + 1))
        
        if separate_W:
            self.W = nn.ModuleList([
                nn.Linear(dim, dim, bias=False)
                for _ in range(K + 1)
            ])
        else:
            self.W = nn.Linear(dim, dim, bias=False)

    def forward(self, A, X):
        alpha = self.alpha_logits
        
        P_k = X 
        if self.separate_W:
            H = alpha[0] * self.W[0](P_k)
        else:
            H = alpha[0] * self.W(P_k)
            
        for k in range(1, self.K + 1):
            P_k = A @ P_k # A^k X
            
            # Apply W_{l,k}
            if self.separate_W:
                proj = self.W[k](P_k)
            else:
                proj = self.W(P_k)
                
            H = H + alpha[k] * proj
            
        return H

class AGFL(nn.Module):
    def __init__(self, dim, heads, K, separate_W=True):
        super().__init__()
        self.heads = heads
        self.dim_h = dim // heads
        
        self.builders = nn.ModuleList([
            GraphConstructor(self.dim_h)
            for _ in range(heads)
        ])
        self.filters = nn.ModuleList([
            GraphFilter(self.dim_h, K, separate_W)
            for _ in range(heads)
        ])
        self.proj = nn.Linear(dim, dim)
        self.last_adj = None

    def forward(self, X, layer_idx, L):
        B, N, D = X.shape

        X = X.view(B, N, self.heads, self.dim_h).transpose(1, 2)
        

        sparsity = sparsity_schedule(layer_idx, L) 
        k_val = max(1, int((1 - sparsity) * N))
        
        outs = []
        adjs = []
        
        for h in range(self.heads):
            Xh = X[:, h]
            S = self.builders[h](Xh)
            
            if k_val < N:
                threshold = torch.topk(S, k_val, dim=-1).values[..., -1:]
                mask = S < threshold
                S = S.masked_fill(mask, float('-inf'))
            
            A_sparse = torch.softmax(S, dim=-1)
            
            outs.append(self.filters[h](A_sparse, Xh))
            adjs.append(A_sparse.detach())

        self.last_adj = torch.stack(adjs)
        
        out = torch.cat(outs, dim=-1)

        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, dim, expansion=4, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class ConvModule(nn.Module):
    def __init__(self, dim, kernel_size=15):
        super().__init__()

        self.pointwise1 = nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.depthwise = nn.Conv1d(
            dim, dim, kernel_size,
            padding=kernel_size // 2,
            groups=dim
        )
        self.batchnorm = nn.BatchNorm1d(dim)
        self.pointwise2 = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        # x: (B, N, D)
        x = x.transpose(1, 2)

        x = self.pointwise1(x)
        x = F.glu(x, dim=1)

        x = self.depthwise(x)
        x = self.batchnorm(x)
        x = F.silu(x)

        x = self.pointwise2(x)

        return x.transpose(1, 2)

class StandardAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=heads, 
            dropout=dropout,
            batch_first=True
        )
        self.last_attn = None

    def forward(self, x):
        out, attn_weights = self.attn(x, x, x, need_weights=True)
        self.last_attn = attn_weights.detach() 
        return out

class NoAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ff = nn.Linear(dim, dim)

    def forward(self, x):
        return self.ff(x)

class ConformerBlock(nn.Module):
    def __init__(self, dim, heads, K, mode="agfl", separate_W=True):
        super().__init__()

        self.mode = mode

        self.ff1 = FeedForward(dim)
        self.ff2 = FeedForward(dim)

        if mode == "agfl":
            self.attn = AGFL(dim, heads, K, separate_W)
        elif mode == "standard":
            self.attn = StandardAttention(dim, heads)
        else:
            self.attn = NoAttention(dim)

        self.conv = ConvModule(dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)

    def forward(self, x, layer_idx=None, L=None):
        x = x + 0.5 * self.ff1(self.norm1(x))
        if self.mode == "agfl":
            x = x + self.attn(self.norm2(x), layer_idx, L)
        else:
            x = x + self.attn(self.norm2(x))

        x = x + self.conv(self.norm3(x))
        x = x + 0.5 * self.ff2(self.norm4(x))

        return x

class ConformerModel(nn.Module):
    def __init__(
        self,
        dim=64,
        depth=3,
        heads=4,
        K=2,
        mode="agfl",
        separate_W=True
    ):
        super().__init__()
        self.mode = mode
        self.input = nn.Linear(1, dim)
        self.layers = nn.ModuleList([
            ConformerBlock(dim, heads, K, mode, separate_W)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.cls = nn.Linear(dim, 2)
        self.pos_emb = nn.Parameter(torch.randn(1, 256, dim))

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.input(x)
        x = x + 0.1 * self.pos_emb[:, :x.size(1)]
        for i, layer in enumerate(self.layers):

            if self.mode == "agfl":
                x = layer(x, i, len(self.layers))
            else:
                x = layer(x)

        x = self.norm(x)

        x_mean = x.mean(dim=1)
        x_max = x.max(dim=1).values
        x_pooled = x_mean + x_max
        return self.cls(x_pooled)


def compute_class_weights(train_loader, device):
    all_labels = []
    for _, y in train_loader:
        all_labels.extend(y.numpy())
    
    all_labels = np.array(all_labels)
    class_counts = np.bincount(all_labels)
    total_samples = len(all_labels)
    num_classes = len(class_counts)
    
    weights = total_samples / (num_classes * class_counts)
    
    print(f"Class counts (0: Normal, 1: Abnormal): {class_counts}")
    print(f"Applied weights: {weights}")
    
    return torch.tensor(weights, dtype=torch.float32).to(device)

def train_eval(model, train_loader, val_loader, device, epochs=50):
    model.to(device)
    lr = 3e-4
    
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3) 
    class_weights = compute_class_weights(train_loader, device)
    loss_fn = FocalLoss(weight=class_weights, gamma=1.0)
    
    warmup_epochs = 10
    warmup_scheduler = LinearLR(opt, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(opt, T_max=epochs - warmup_epochs)
    scheduler = SequentialLR(opt, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    pbar = tqdm.trange(epochs, desc="Training Model", unit="epoch")
    history = {'loss': [], 'val_acc': [], 'lr': []}
    
    for epoch in pbar:
        model.train()
        epoch_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            
            epoch_loss += loss.item()
            
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        model.eval()
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for x_val, y_val in val_loader:
                logits_val = model(x_val.to(device))
                val_preds.extend(logits_val.argmax(-1).cpu().numpy())
                val_targets.extend(y_val.numpy())
                
        current_acc = accuracy_score(val_targets, val_preds)
        
        history['loss'].append(avg_loss)
        history['val_acc'].append(current_acc)
        history['lr'].append(opt.param_groups[0]['lr'])

        pbar.set_postfix({
            "Loss": f"{avg_loss:.4f}", 
            "Val_Acc": f"{current_acc:.4f}",
            "LR": f"{opt.param_groups[0]['lr']:.2e}"
        })

    model.eval()
    preds, targets, probs = [], [], []

    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x.to(device))
            preds.extend(logits.argmax(-1).cpu().numpy())
            probabilities = F.softmax(logits, dim=-1)
            probs.extend(probabilities[:, 1].cpu().numpy())
            targets.extend(y.numpy())

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='macro') 
    auc = roc_auc_score(targets, probs)

    return acc, f1, auc, history

def plot_training_curves(history, name, save_path):
    epochs = range(1, len(history['loss']) + 1)
    
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    
    axs[0].plot(epochs, history['loss'], marker='o', color='#d62728', linewidth=2, label='Train Loss')
    axs[0].set_ylabel('Loss', fontsize=12)
    axs[0].set_title(f'{name} Training Trajectory', fontsize=14, fontweight='bold')
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].legend()

    axs[1].plot(epochs, history['val_acc'], marker='o', color='#1f77b4', linewidth=2, label='Val Accuracy')
    axs[1].set_ylabel('Accuracy', fontsize=12)
    axs[1].grid(True, linestyle='--', alpha=0.7)
    axs[1].legend()
    
    axs[2].plot(epochs, history['lr'], marker='o', color='#2ca02c', linewidth=2, label='Learning Rate')
    axs[2].set_xlabel('Epochs', fontsize=12)
    axs[2].set_ylabel('Learning Rate', fontsize=12)
    axs[2].grid(True, linestyle='--', alpha=0.7)
    axs[2].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    axs[2].legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/training_curves_{name}.png", dpi=300)
    plt.close()

def attention_entropy(A):
    eps = 1e-8

    if isinstance(A, np.ndarray):
        A = torch.from_numpy(A)

    return -(A * torch.log(A + eps)).sum(dim=-1).mean().item()

def plot_comparison(results, save_path):
    plt.figure(figsize=(8, 5))
    names = list(results.keys())
    vals = [metrics['Accuracy'] for metrics in results.values()]
    bars = plt.bar(names, vals, color='#4C72B0')
    
    plt.title("Model Comparison (Accuracy)", fontsize=14, fontweight='bold')
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0.0, 1.0)
    plt.bar_label(bars, fmt='%.4f', padding=3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/comparison.png", dpi=300)
    plt.close()

def plot_entropy(models, x, save_path):
    plt.figure()
    for name, model in models.items():
        with torch.no_grad():
            _ = model(x)
        ent = []
        for layer in model.layers:
            if hasattr(layer.attn, "last_adj") and layer.attn.last_adj is not None:
                A = layer.attn.last_adj
                if A.ndim == 4:
                    A = A.mean(0).mean(0).cpu().numpy()
                elif A.ndim == 3:
                    A = A.mean(0).cpu().numpy()
                ent.append(attention_entropy(A))
            elif hasattr(layer.attn, "last_attn") and layer.attn.last_attn is not None:
                A = layer.attn.last_attn
                if A.ndim == 4:
                    A = A.mean(0).mean(0).cpu().numpy()
                elif A.ndim == 3:
                    A = A.mean(0).cpu().numpy()
                ent.append(attention_entropy(A))
        if ent:
            plt.plot(ent, marker='o', label=name)

    plt.legend()
    plt.title("Attention Entropy")
    plt.savefig(f"{save_path}/entropy.png")
    plt.close()


def plot_sparsity(models, x, save_path):
    plt.figure()
    for name, model in models.items():
        with torch.no_grad():
            _ = model(x)
        sp = []
        for layer in model.layers:
            A = None
            if hasattr(layer.attn, "last_adj") and layer.attn.last_adj is not None:
                A = layer.attn.last_adj
                if A.ndim == 4:
                    A = A.mean(0).mean(0)
                elif A.ndim == 3:
                    A = A.mean(0)
            elif hasattr(layer.attn, "last_attn") and layer.attn.last_attn is not None:
                A = layer.attn.last_attn
                if A.ndim == 4:
                    A = A.mean(0).mean(0)
                elif A.ndim == 3:
                    A = A.mean(0)

            if A is not None:
                sp.append((A > 1e-3).float().mean().item())

        if sp:
            plt.plot(sp, marker='o', label=name)

    plt.legend()
    plt.title("Sparsity")
    plt.savefig(f"{save_path}/sparsity.png")
    plt.close()

def plot_layer_heatmaps(models, x, save_path, save_prefix="heatmap"):
    for name, model in models.items():
        with torch.no_grad():
            _ = model(x)

        fig, axes = plt.subplots(1, len(model.layers), figsize=(4*len(model.layers), 4))
        if len(model.layers) == 1:
            axes = [axes]

        for i, layer in enumerate(model.layers):
            A = None
            if hasattr(layer.attn, "last_adj") and layer.attn.last_adj is not None:
                A = layer.attn.last_adj
                if A.ndim == 4:
                    A = A.mean(0).mean(0).cpu().numpy()
                elif A.ndim == 3:
                    A = A.mean(0).cpu().numpy()
            elif hasattr(layer.attn, "last_attn") and layer.attn.last_attn is not None:
                A = layer.attn.last_attn
                if A.ndim == 4:
                    A = A.mean(0).mean(0).cpu().numpy()
                elif A.ndim == 3:
                    A = A.mean(0).cpu().numpy()

            if A is not None and A.ndim == 2:
                im = axes[i].imshow(A, aspect='auto', cmap='viridis')
                axes[i].set_title(f"Layer {i}")
                axes[i].set_xlabel("Node")
                axes[i].set_ylabel("Node")
                fig.colorbar(im, ax=axes[i])

        plt.suptitle(f"{name} Layer-wise Attention/Adjacency Heatmaps")
        plt.tight_layout()
        plt.savefig(f"{save_path}/{save_prefix}_{name}.png")
        plt.close()

def plot_embedding_space(models, x, y, save_path, method="tsne", save_prefix="embedding"):
    y_np = y.cpu().numpy() if torch.is_tensor(y) else np.array(y)
    
    for name, model in models.items():
        with torch.no_grad():
            x_dev = x.to(next(model.parameters()).device)
            out = model.input(x_dev.unsqueeze(-1))
            
            out = out + 0.1 * model.pos_emb[:, :out.size(1)]
            
            for i, layer in enumerate(model.layers):
                if model.mode == "agfl":
                    out = layer(out, i, len(model.layers))
                else:
                    out = layer(out)
                    
            out = model.norm(out)
            embeddings = out.mean(dim=1).cpu().numpy()

        if method.lower() == "tsne":
            reducer = TSNE(n_components=2, random_state=0)
        else:
            reducer = umap.UMAP(n_components=2, random_state=0)

        emb2d = reducer.fit_transform(embeddings)

        plt.figure(figsize=(6,6))
        
        for c in np.unique(y_np):
            plt.scatter(emb2d[y_np==c, 0], emb2d[y_np==c, 1], label=f"Class {c}", alpha=0.7)
            
        plt.legend()
        plt.title(f"{name} Embedding {method.upper()}")
        plt.savefig(f"{save_path}/{save_prefix}_{name}_{method}.png")
        plt.close()

def plot_ecg_attention(x, models, save_path, save_prefix="ecg_attention"):
    x_sample = x[0].unsqueeze(0) 
    
    for name, model in models.items():
        device = next(model.parameters()).device
        x_dev = x_sample.to(device)
        
        with torch.no_grad():
            _ = model(x_dev)

        for i, layer in enumerate(model.layers):
            A = None
            
            if hasattr(layer.attn, "last_adj") and layer.attn.last_adj is not None:
                A = layer.attn.last_adj
            elif hasattr(layer.attn, "last_attn") and layer.attn.last_attn is not None:
                A = layer.attn.last_attn

            if A is not None:
                if A.ndim == 4:
                    A = A.mean(0).mean(0).cpu().numpy()
                elif A.ndim == 3:
                    A = A.mean(0).cpu().numpy()

            if A is not None and A.ndim == 2:
                fig, ax1 = plt.subplots(figsize=(10, 4))
                
                im = ax1.imshow(A, aspect='auto', cmap='viridis', alpha=0.85)
                ax1.set_xlabel("Time (Samples)", fontsize=11)
                ax1.set_ylabel("Attention Nodes (Queries)", fontsize=11)
                
                cbar = fig.colorbar(im, ax=ax1, pad=0.1)
                cbar.set_label("Attention Weight", rotation=270, labelpad=15)

                ax2 = ax1.twinx()
                ecg_signal = x_sample[0].cpu().numpy()
                
                ax2.plot(ecg_signal, color='#ff3333', linewidth=2.0, label="ECG Signal")
                ax2.set_ylabel("ECG Amplitude", fontsize=11, color='#ff3333')
                ax2.tick_params(axis='y', labelcolor='#ff3333')
                
                ax1.set_xlim(0, len(ecg_signal) - 1)
                
                plt.title(f"{name} - Layer {i} Attention Mapping", fontsize=13, fontweight='bold')
                
                fig.tight_layout()
                plt.savefig(f"{save_path}/{save_prefix}_{name}_layer{i}.png", dpi=300)
                plt.close(fig)


def plot_sparsity_vs_accuracy(models, val_loader, device, save_path):
    accs = {}
    sparsities = {}

    for name, model in models.items():
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds.extend(logits.argmax(-1).cpu().numpy())
                targets.extend(y.cpu().numpy())
        accs[name] = accuracy_score(targets, preds)

        sparsity_list = []
        x_vis, _ = next(iter(val_loader))
        x_vis = x_vis.to(device)
        with torch.no_grad():
            _ = model(x_vis)
            for layer in model.layers:
                A = None
                if hasattr(layer.attn, "last_adj") and layer.attn.last_adj is not None:
                    A = layer.attn.last_adj
                    if A.ndim == 4:
                        A = A.mean(0).mean(0).cpu().numpy()
                    elif A.ndim == 3:
                        A = A.mean(0).cpu().numpy()
                elif hasattr(layer.attn, "last_attn") and layer.attn.last_attn is not None:
                    A = layer.attn.last_attn
                    if A.ndim == 4:
                        A = A.mean(0).mean(0).cpu().numpy()
                    elif A.ndim == 3:
                        A = A.mean(0).cpu().numpy()
                if A is not None:
                    sparsity_list.append((A > 1e-3).mean().item())
        if sparsity_list:
            sparsities[name] = np.mean(sparsity_list)
        else:
            sparsities[name] = 0.0

    plt.figure(figsize=(6,6))
    for name in models.keys():
        plt.scatter(sparsities[name], accs[name], label=name, s=100)
    plt.xlabel("Average Sparsity")
    plt.ylabel("Validation Accuracy")
    plt.title("Sparsity vs Accuracy")
    plt.legend()
    plt.savefig(f"{save_path}/sparsity_vs_acc.png")
    plt.close()



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = "./out_attention4"

    data_dir = "./mit-bih-arrhythmia-database-1.0.0"
    train_records = [
        '101', '106', '108', '109', '112', '114', '115', '116', 
        '118', '119', '122', '124', '201', '203', '205', '207', 
        '208', '209', '215', '220', '223', '230'
    ] 
    
    val_records = [
        '100', '103', '104', '105', '111', '113', '117', '121', 
        '200', '202', '210', '212', '213', '214', '219', '221', 
        '222', '228', '231', '232', '233', '234'
    ]

    train_loader = DataLoader(ECGDataset(train_records, data_dir, is_train=True), batch_size=64, shuffle=True)
    val_loader = DataLoader(ECGDataset(val_records, data_dir), batch_size=64)

    configs = [
        ("AGFL", "agfl"),
        #("No Attention", "none"),
        ("Standard", "standard"),
    ]

    results = {}
    models = {}

    for name, mode in configs:
        print(f"{name} is on\n")
        model = ConformerModel(mode=mode).to(device)
        
        acc, f1, auc, history = train_eval(model, train_loader, val_loader, device)
        
        print(f"{name}:")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1 (Macro): {f1:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}\n")

        results[name] = {
            'Accuracy': acc,
            'F1_Score': f1,
            'ROC_AUC': auc
        }
        models[name] = model
        plot_training_curves(history, name, save_path)

    x_vis, y_vis = next(iter(val_loader))
    x_vis = x_vis.to(device)
    y_vis = y_vis.cpu().numpy()

    plot_comparison(results, save_path)
    plot_layer_heatmaps(models, x_vis, save_path)
    plot_embedding_space(models, x_vis, y_vis, save_path, method="tsne")
    plot_embedding_space(models, x_vis, y_vis, save_path, method="umap")
    plot_ecg_attention(x_vis, models, save_path)
    plot_sparsity_vs_accuracy(models, val_loader, device, save_path)
    plot_entropy(models, x_vis, save_path)
    plot_sparsity(models, x_vis, save_path)


if __name__ == "__main__":
    main()
