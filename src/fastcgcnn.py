# fastcgcnn.py: helpers shared by the later cells (written to your Drive by Cell 5)
import os, sys, json, time, subprocess, warnings
from functools import partial
import numpy as np, pandas as pd

DRIVE = "/content/drive/MyDrive/cgcnn_run"
REPO = "/content/cgcnn"
RADIUS, MAXN = 8.0, 12                      # CGCNN defaults: 12 nearest neighbours within 8 A

def ensure_repo():
    # clone the CGCNN repo if this runtime does not have it, and make it importable
    if not os.path.exists(REPO):
        subprocess.run(["git", "clone", "-q", "https://github.com/txie-93/cgcnn.git", REPO], check=True)
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    return f"{REPO}/data/sample-regression/atom_init.json"

def _build_one(cid, root):
    # same graph CGCNN builds: 12 nearest neighbours within 8 A, padded with (index 0, distance 9) if fewer
    from pymatgen.core import Structure
    s = Structure.from_file(f"{root}/{cid}.cif")
    Z = np.array([site.specie.Z for site in s], dtype=np.int16)
    idx = np.zeros((len(s), MAXN), dtype=np.int32)
    dist = np.full((len(s), MAXN), RADIUS + 1.0, dtype=np.float32)
    for i, nb in enumerate(s.get_all_neighbors(RADIUS, include_index=True)):
        nb = sorted(nb, key=lambda x: x[1])[:MAXN]
        idx[i, :len(nb)] = [x[2] for x in nb]
        dist[i, :len(nb)] = [x[1] for x in nb]
    return Z, idx, dist

def build_graphs(root, out):
    # one-time job: build every crystal graph in `root` (CIFs + id_prop.csv) and save them in one file
    if os.path.exists(out):
        print("graphs already built, skipping:", out); return
    if not os.path.exists(f"{root}/id_prop.csv"):
        raise FileNotFoundError(f"{root}/id_prop.csv not found: run the cell that creates {root} first")
    from multiprocessing import get_context
    from tqdm.auto import tqdm
    warnings.filterwarnings("ignore")
    lab = pd.read_csv(f"{root}/id_prop.csv", header=None, names=["id", "y"])
    with get_context("fork").Pool(os.cpu_count()) as p:
        res = list(tqdm(p.imap(partial(_build_one, root=root), lab.id, chunksize=64), total=len(lab)))
    Z, idx, dist = (np.concatenate([r[k] for r in res]) for k in range(3))
    start = np.concatenate([[0], np.cumsum([len(r[0]) for r in res])])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp.npz"                  # write to a temp name first, so a half-written file is never mistaken for a finished one
    np.savez(tmp, ids=lab.id.values.astype(str), y=lab.y.values.astype(np.float32), start=start, Z=Z, idx=idx, dist=dist)
    os.replace(tmp, out)
    print(len(lab), "structures,", len(Z), "atoms ->", out)

def train(graphs, split, run, epochs=150, bs=256, lr=1e-3, seed=0):
    # train CGCNN on the split in `split` (json with train/val/test ids); checkpoints + predictions go to the folder `run`
    import torch, torch.nn as nn
    atom_init = ensure_repo()
    from cgcnn.model import CrystalGraphConvNet
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(run, exist_ok=True); torch.manual_seed(seed); np.random.seed(seed)
    print("device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "(NO GPU: this will be slow)")

    # ---- everything lives on the GPU: atoms, neighbour lists, targets ----
    G = np.load(graphs); ids = G["ids"]; pos = {c: i for i, c in enumerate(ids)}
    T = lambda a, dt=None: torch.tensor(a, device=dev, dtype=dt)
    start, Zall = T(G["start"], torch.long), T(G["Z"], torch.long)
    idxall, distall, Y = T(G["idx"], torch.long), T(G["dist"], torch.float32), T(G["y"], torch.float32)
    sj = json.load(open(split))
    assert not (set(sj["train"]) & set(sj["test"])) and not (set(sj["val"]) & set(sj["test"])) and not (set(sj["train"]) & set(sj["val"])), "train/val/test overlap!"
    sp = {k: T([pos[c] for c in sj[k]], torch.long) for k in ("train", "val", "test")}
    print("train / val / test =", len(sp["train"]), len(sp["val"]), len(sp["test"]), "(no overlap, checked)")

    ai = json.load(open(atom_init))
    table = torch.zeros(101, 92)
    for k, v in ai.items(): table[int(k)] = torch.tensor(v)
    table = table.to(dev)
    filt = T(np.arange(0, 8.0 + 0.2, 0.2), torch.float32)   # CGCNN's 41 Gaussian centres (width 0.2)

    def batch(cids):
        # assemble a batch of crystals on the GPU, no Python loops
        s = start[cids]; n = start[cids + 1] - s
        off = torch.cumsum(n, 0) - n                                   # first atom of each crystal inside the batch
        bvec = torch.repeat_interleave(torch.arange(len(cids), device=dev), n)   # crystal of every atom
        a = s[bvec] + torch.arange(len(bvec), device=dev) - off[bvec]  # global atom index
        nbr_fea = torch.exp(-((distall[a].unsqueeze(-1) - filt) ** 2) / 0.2 ** 2)
        return table[Zall[a]], nbr_fea, idxall[a] + off[bvec].unsqueeze(1), (bvec, n)

    class FastCGCNN(CrystalGraphConvNet):                              # the repo's own model; only pooling is vectorised
        def pooling(self, atom_fea, crystal_atom_idx):
            bvec, n = crystal_atom_idx
            out = torch.zeros(len(n), atom_fea.shape[1], device=atom_fea.device, dtype=atom_fea.dtype)
            return out.index_add_(0, bvec, atom_fea) / n.unsqueeze(1).to(atom_fea.dtype)

    model = FastCGCNN(92, len(filt), atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1).to(dev)
    mu, sd = Y[sp["train"]].mean(), Y[sp["train"]].std()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    @torch.no_grad()
    def evaluate(cids, ebs=1024):
        model.eval()
        p = torch.cat([model(*batch(cids[i:i + ebs])).squeeze(1) * sd + mu for i in range(0, len(cids), ebs)])
        return (p - Y[cids]).abs().mean().item(), p

    hist, best, ep0, ck = [], 1e9, 0, f"{run}/last.pt"
    if os.path.exists(ck):                                             # resume after a disconnect
        c = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
        hist, best, ep0 = c["hist"], c["best"], c["epoch"] + 1
        print("resumed at epoch", ep0)

    tr = sp["train"]
    for ep in range(ep0, epochs):
        t0 = time.time(); model.train(); tot = torch.zeros((), device=dev)
        perm = tr[torch.randperm(len(tr), device=dev)]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            if len(b) < 8: continue                                    # BatchNorm needs more than a couple of atoms
            loss = nn.functional.mse_loss(model(*batch(b)).squeeze(1), (Y[b] - mu) / sd)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.detach() * len(b)
        sched.step()
        vmae, _ = evaluate(sp["val"])
        hist.append(dict(epoch=ep, train_loss=tot.item() / len(tr), val_mae=vmae, lr=sched.get_last_lr()[0], sec=time.time() - t0))
        if vmae < best:
            best = vmae; torch.save(model.state_dict(), f"{run}/best.pt")
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), hist=hist, best=best, epoch=ep), ck)
        if ep < 3 or ep % 5 == 0 or ep == epochs - 1:
            print(f"epoch {ep:3d} | val MAE {vmae:.4f} (best {best:.4f}) | {hist[-1]['sec']:.0f}s")

    # ---- final: test MAE with the best-validation weights, plus predictions for error analysis ----
    model.load_state_dict(torch.load(f"{run}/best.pt", map_location=dev))
    tmae, p = evaluate(sp["test"])
    print(f"TEST MAE {tmae:.4f} eV/atom   (best validation MAE {best:.4f})")
    pd.DataFrame(dict(id=ids[sp["test"].cpu().numpy()], y_true=Y[sp["test"]].cpu().numpy(), y_pred=p.cpu().numpy())).to_csv(f"{run}/test_predictions.csv", index=False)
    pd.DataFrame(hist).to_csv(f"{run}/history.csv", index=False)
    return dict(test_mae=tmae, best_val=best)
