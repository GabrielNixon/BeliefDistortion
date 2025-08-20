import math, random, time, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def set_seed(seed=123):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.benchmark = False
epochs       = 1
batch_size   = 256
lr           = 1e-3
label_noise  = 0.2
seed         = 123
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(seed)

tf_train = transforms.Compose([transforms.ToTensor()])
tf_test  = transforms.Compose([transforms.ToTensor()])
train = datasets.CIFAR10(root="./data", train=True, download=True, transform=tf_train)
test  = datasets.CIFAR10(root="./data", train=False, download=True, transform=tf_test)

def corrupt_labels(dataset, p=0.2, num_classes=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = len(dataset.targets)
    idx = torch.randperm(n, generator=g)[:int(p*n)]
    for i in idx:
        y = int(dataset.targets[i])
        r = random.randrange(num_classes-1)
        dataset.targets[i] = (y + 1 + r) % num_classes
    return idx

if label_noise > 0: corrupt_labels(train, p=label_noise, num_classes=10, seed=seed)

class IndexedDataset(Dataset):
    def __init__(self, base): self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]; return i, x, y

tr = IndexedDataset(train); te = IndexedDataset(test)
train_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader  = DataLoader(te, batch_size=512, shuffle=False, num_workers=0)

class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128*4*4, 256)
        self.fc2 = nn.Linear(256, num_classes)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def ece_score(probs, labels, n_bins=15):
    device = probs.device
    conf, preds = probs.max(1)
    acc = preds.eq(labels).float()
    bins = torch.linspace(0, 1, n_bins+1, device=device)
    e = []
    for b in range(n_bins):
        m = (conf > bins[b]) & (conf <= bins[b+1])
        if m.any():
            e.append((acc[m].mean() - conf[m].mean()).abs() * m.float().mean())
    return 100*torch.stack(e).sum().item()

def nll_score(probs, labels):
    p = probs.gather(1, labels.view(-1,1)).clamp_min(1e-12).squeeze(1)
    return -p.log().mean().item()

def mean_entropy(probs):
    return (-probs.clamp_min(1e-12).log()*probs).mul(probs).sum(1).mean().item()

@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()
    tot = 0; corr = 0; all_p, all_y = [], []
    for _, x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(1)
        corr += pred.eq(y).sum().item(); tot += y.size(0)
        all_p.append(probs.cpu()); all_y.append(y.cpu())
    acc = 100.0 * corr / tot
    P = torch.cat(all_p, 0); L = torch.cat(all_y, 0)
    ece = ece_score(P,L); nll = nll_score(P,L); ent = mean_entropy(P)
    return {"clean": {"acc": acc, "ECE": ece, "NLL": nll, "H": ent}}

def brief(name, res):
    m = res["clean"]
    print(f"{name:11s} | Acc: {m['acc']:5.1f}  ECE: {m['ECE']:6.2f}  NLL: {m['NLL']:7.4f}  H: {m['H']:6.4f}")

def run_baseline(model, loader, device, epochs=1, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        for _, x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
    return model

def run_temp(model, loader, device, epochs=1, lr=1e-3, temp=1.3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        for _, x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits / temp, y)
            opt.zero_grad(); loss.backward(); opt.step()
    return model

def run_trust(model, loader, device, epochs=1, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = len(loader.dataset); C = 10
    p_mem = torch.full((N, C), 1.0/C, device=device)
    state = {"U": torch.tensor(0.0, device=device),
             "M": torch.zeros((), device=device),
             "V": torch.zeros((), device=device),
             "phi": torch.tensor(0.0, device=device)}
    a,b,c,d0 = 1.5,1.0,0.8,-1.0; delta=0.01; k=3.0
    lam_min, lam_max = 0.5, 4.0; tau_min, tau_max = 0.05, 1.5
    alpha_eta = 0.5; eps_label = 0.05
    for ep in range(epochs):
        model.train()
        for idx, x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            with torch.no_grad():
                p_y = probs.gather(1, y.view(-1,1)).clamp_min(1e-12).squeeze(1)
                u = -p_y.log(); u_mean = u.mean()
                beta_p=0.02
                state["U"] = (1-beta_p)*state["U"] + beta_p*u_mean
                state["M"] = (1-beta_p)*state["M"] + beta_p*probs.mean()
                state["V"] = (1-beta_p)*state["V"] + beta_p*(probs - probs.mean()).pow(2).mean()
                state["phi"] = torch.maximum(torch.tensor(0.0, device=device), state["phi"] + (u_mean - state["U"]) - delta)
                m = 1 - torch.exp(-k*torch.minimum(torch.tensor(1.0, device=device), state["phi"]))
                U_t = torch.clamp((state["U"]-0.0)/(1.5+1e-9), 0, 1)
                V_t = torch.clamp((state["V"]-0.0)/(0.25+1e-9), 0, 1)
                S = torch.sigmoid(a*U_t + b*V_t + c*m + d0).clamp(0.0,1.0)
                lam = lam_min + (lam_max-lam_min)*S
                tau = tau_min + (tau_max-tau_min)*S
                T = lam + tau
                alpha_w = lam / T
                eta = (1.0/(1.0+lam))**alpha_eta
            p_prior = p_mem[idx]
            y_onehot = torch.zeros_like(p_prior).scatter_(1, y[:,None], 1.0)
            r_obs = (1.0 - eps_label)*y_onehot + eps_label*(1.0-y_onehot)/(C-1)
            q_logits = alpha_w*torch.log(p_prior.clamp_min(1e-12)) + (1.0/T)*torch.log(r_obs.clamp_min(1e-12))
            q = torch.softmax(q_logits, dim=1).detach()
            kl = F.kl_div(F.log_softmax(logits, dim=1), q, reduction="batchmean")
            ce = F.cross_entropy(logits, y)
            loss = 0.7*kl + 0.3*ce
            opt.zero_grad(); loss.backward(); opt.step()
            p_mem[idx] = (1-eta)*p_prior + eta*q
    return model

base = TinyCNN().to(device)
base = run_baseline(base, train_loader, device, epochs=epochs, lr=lr)
res_base = evaluate(base, test_loader, device)
temp = TinyCNN().to(device)
temp = run_temp(temp, train_loader, device, epochs=epochs, lr=lr, temp=1.3)
res_temp = evaluate(temp, test_loader, device)
trust = TinyCNN().to(device)
trust = run_trust(trust, train_loader, device, epochs=epochs, lr=lr)
res_trust = evaluate(trust, test_loader, device)

print("\n=== One-epoch sanity ===")
brief("Baseline", res_base)
brief("Temp-Bayes", res_temp)
brief("Trust-aware", res_trust)

EPOCHS_DEMO = 10
def curve_runner(mode):
    accs, eces = [], []
    model = TinyCNN().to(device)
    for ep in range(EPOCHS_DEMO):
        if mode=="baseline":
            model = run_baseline(model, train_loader, device, epochs=1, lr=lr)
        elif mode=="temp":
            model = run_temp(model, train_loader, device, epochs=1, lr=lr, temp=1.3)
        else:
            model = run_trust(model, train_loader, device, epochs=1, lr=lr)
        res = evaluate(model, test_loader, device)
        accs.append(res["clean"]["acc"])
        eces.append(res["clean"]["ECE"])
    return accs, eces

acc_b, ece_b = curve_runner("baseline")
acc_t, ece_t = curve_runner("temp")
acc_r, ece_r = curve_runner("trust")

xs = np.arange(1, EPOCHS_DEMO+1)
plt.figure()
plt.plot(xs, acc_b, label="Baseline")
plt.plot(xs, acc_t, label="Temp-Bayes")
plt.plot(xs, acc_r, label="Trust-aware")
plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("Accuracy vs Epoch")
plt.legend(); plt.grid(True); plt.show()

plt.figure()
plt.plot(xs, ece_b, label="Baseline")
plt.plot(xs, ece_t, label="Temp-Bayes")
plt.plot(xs, ece_r, label="Trust-aware")
plt.xlabel("Epoch"); plt.ylabel("ECE"); plt.title("ECE vs Epoch")
plt.legend(); plt.grid(True); plt.show()
