#!/usr/bin/env python3
"""
aerial-landuse-resolution-shift
A one-day study: how do different transfer learning strategies hold up when the
pixel size (ground sampling distance, GSD) of aerial images changes?

Public data only:
  - UC Merced Land Use (2,100 aerial tiles, 21 classes, 0.3 m pixels)
  - EuroSAT RGB (27,000 Sentinel-2 tiles, 10 classes, 10 m pixels)

Commands (run in this order, or use `all`):
  python aerial_resolution_shift.py prepare   # download, check, make example figure
  python aerial_resolution_shift.py exp_a     # strategies S1-S4 vs simulated GSD shift
  python aerial_resolution_shift.py exp_b     # simulated shift vs real satellite shift
  python aerial_resolution_shift.py exp_c     # how many target images close the gap?
  python aerial_resolution_shift.py figures   # make all charts from the CSV files

Add --smoke to any command to run a fast test on synthetic data (no downloads,
no pretrained weights, CPU is fine). Always do this first.
"""
import argparse
import glob
import hashlib
import json
import os
import random
import sys, zlib
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(os.environ.get("RS_ROOT", "./work"))
DATA, RES, FIG = ROOT / "data", ROOT / "results", ROOT / "figures"

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
UCM_URL = "https://huggingface.co/datasets/torchgeo/ucmerced/resolve/main/UCMerced_LandUse.zip"
UCM_MD5 = "5b7ec56793786b6dc8a908e8854ac0e4"  # as listed in the torchgeo source
UCM_CLASSES = ["agricultural", "airplane", "baseballdiamond", "beach", "buildings", "chaparral",
               "denseresidential", "forest", "freeway", "golfcourse", "harbor", "intersection",
               "mediumresidential", "mobilehomepark", "overpass", "parkinglot", "river", "runway",
               "sparseresidential", "storagetanks", "tenniscourt"]
BASE_GSD_M = 0.3                       # UC Merced pixel size in metres
FACTORS = [1, 2, 4, 8, 16]            # coarsening factors -> 0.3, 0.6, 1.2, 2.4, 4.8 m
S4_TRAIN_FACTORS = [1, 2, 4, 8]       # factor 16 is deliberately NOT seen in training
GSD_LABELS = [f"{BASE_GSD_M * f:.1f} m" for f in FACTORS]

# Shared classes between UC Merced and EuroSAT (merged where needed)
UCM_TO_5 = {"agricultural": "agriculture", "forest": "forest", "freeway": "highway", "river": "river",
            "denseresidential": "residential", "mediumresidential": "residential",
            "sparseresidential": "residential"}
EURO_TO_5 = {"annualcrop": "agriculture", "permanentcrop": "agriculture", "forest": "forest",
             "highway": "highway", "river": "river", "residential": "residential"}
FIVE = ["agriculture", "forest", "highway", "residential", "river"]

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# colour per strategy (colour-blind-safe set) + marker so identity is never colour-only
STYLE = {"S0": ("#7a7a75", "x", ":"), "S1": ("#2a78d6", "o", "-"), "S2": ("#eb6834", "s", "-"),
         "S3": ("#1baf7a", "^", "-"), "S4": ("#4a3aa7", "D", "-")}
NAMES = {"S0": "S0 from scratch", "S1": "S1 linear probe (frozen)", "S2": "S2 fine-tune last stage",
         "S3": "S3 fine-tune all", "S4": "S4 fine-tune all + GSD augmentation"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def set_seed(s):
    import torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ----------------------------------------------------------------------------
# Resolution simulation
# ----------------------------------------------------------------------------
def degrade(im: Image.Image, factor: int) -> Image.Image:
    """Imitate a coarser sensor (or a higher flight): average pixels in factor x factor
    blocks (like a larger pixel would), then enlarge back so the network sees the same
    input size. Simulates pixel size only: not viewing angle, sun, haze or sensor noise."""
    if factor == 1:
        return im
    w, h = im.size
    small = im.resize((max(1, w // factor), max(1, h // factor)), Image.BOX)
    return small.resize((w, h), Image.BILINEAR)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    log("downloading", url)
    urllib.request.urlretrieve(url, dest)


def _synth(label_name, size, rng):
    """Synthetic stand-in image for smoke tests. Class identity is a base colour + stripe frequency."""
    base_rng = np.random.default_rng(zlib.crc32(label_name.encode()))
    colour = base_rng.uniform(0.15, 0.85, 3)
    freq = base_rng.integers(2, 9)
    yy, xx = np.mgrid[0:size, 0:size] / size
    stripes = 0.15 * np.sin(2 * np.pi * freq * (xx + yy))[..., None]
    arr = colour[None, None, :] + stripes + rng.normal(0, 0.05, (size, size, 3))
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


def load_ucm(smoke=False):
    """Returns (list of PIL images 256x256, list of UC Merced class names per image)."""
    if smoke:
        rng = np.random.default_rng(0)
        imgs, names = [], []
        for c in UCM_CLASSES:
            tgt = UCM_TO_5.get(c, c)           # same look for classes that merge
            for _ in range(12):
                imgs.append(_synth(tgt, 256, rng)); names.append(c)
        return imgs, names
    zpath = DATA / "UCMerced_LandUse.zip"
    download(UCM_URL, zpath)
    got = md5(zpath)
    if got != UCM_MD5:
        log("WARNING: md5 differs from the value listed by torchgeo:", got)
    out = DATA / "ucmerced"
    if not out.exists():
        with zipfile.ZipFile(zpath) as z:
            z.extractall(out)
    files = sorted(glob.glob(str(out / "**" / "*.tif"), recursive=True))
    imgs, names = [], []
    for f in files:
        cls = Path(f).parent.name.lower()
        if cls not in UCM_CLASSES:
            continue
        im = Image.open(f).convert("RGB")
        if im.size != (256, 256):
            im = im.resize((256, 256), Image.BILINEAR)    # 44 images are not 256x256
        imgs.append(im); names.append(cls)
    log(f"UC Merced: {len(imgs)} images, {len(set(names))} classes (expected 2100 and 21)")
    return imgs, names


def load_eurosat(n_per_class, seed, smoke=False, eurosat_dir=None):
    """Returns dict: five_class_name -> list of PIL images (64x64), n_per_class sampled per FIVE class."""
    rng = np.random.default_rng(seed)
    pools = {c: [] for c in FIVE}
    if smoke:
        for raw, five in list(EURO_TO_5.items()):
            for _ in range(40):
                pools[five].append(_synth(five, 64, rng))
    else:
        root = Path(eurosat_dir) if eurosat_dir else DATA / "eurosat"
        has_files = root.exists() and any(root.rglob("*." + e) for e in ("jpg", "png", "tif"))
        if not has_files:
            log("downloading EuroSAT RGB through torchvision (falls back to manual instructions on failure)")
            try:
                import torchvision
                torchvision.datasets.EuroSAT(root=str(root), download=True)
            except Exception as e:
                sys.exit("EuroSAT download failed: %s\nDownload the RGB version manually (Zenodo record 7711810 or "
                         "github.com/phelber/EuroSAT), unzip it so that folders like Forest/ River/ exist, "
                         "then run again with --eurosat-dir <that folder>." % e)
        files = [f for ext in ("jpg", "png", "tif") for f in root.rglob("*." + ext)]
        by_class = {}
        for f in files:
            raw = f.parent.name.lower()
            if raw in EURO_TO_5:
                by_class.setdefault(EURO_TO_5[raw], []).append(f)
        for five, fl in by_class.items():
            fl = sorted(fl)
            rng.shuffle(fl)
            pools[five] = [Image.open(f).convert("RGB") for f in fl[: n_per_class]]
    for c in FIVE:
        pools[c] = pools[c][: n_per_class]
    return pools


def stratified_split(y, seed, fr=(0.7, 0.1, 0.2)):
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx); ntr = int(round(fr[0] * n)); nva = int(round(fr[1] * n))
        tr += list(idx[:ntr]); va += list(idx[ntr:ntr + nva]); te += list(idx[ntr + nva:])
    return np.array(tr), np.array(va), np.array(te)


# ----------------------------------------------------------------------------
# Torch pieces (imported lazily so that `figures` works without torch)
# ----------------------------------------------------------------------------
def torch_bits():
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms.functional as TF
    return torch, nn, torchvision, TF


class ImgDS:
    def __init__(self, imgs, labels, factors=(1,), train=False, size=224):
        self.imgs, self.labels, self.factors, self.train, self.size = imgs, list(labels), list(factors), train, size
        self.TF = torch_bits()[3]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        im = self.imgs[i]
        if self.train:
            if random.random() < 0.5: im = im.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5: im = im.transpose(Image.FLIP_TOP_BOTTOM)
            k = random.randint(0, 3)
            if k: im = im.rotate(90 * k)
        f = random.choice(self.factors) if len(self.factors) > 1 else self.factors[0]
        im = degrade(im, f)
        im = im.resize((self.size, self.size), Image.BILINEAR)
        x = self.TF.normalize(self.TF.to_tensor(im), MEAN, STD)
        return x, self.labels[i]


def make_loader(ds, bs, shuffle, smoke):
    torch = torch_bits()[0]
    drop = shuffle and len(ds) > bs and len(ds) % bs == 1      # avoid a batch of 1 (BatchNorm error)
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=drop,
                                       num_workers=0 if smoke else 2, pin_memory=False)


def device():
    torch = torch_bits()[0]
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


CFG = {  # epochs, learning rate
    "S0": dict(epochs=30, lr=1e-3), "S1": dict(epochs=15, lr=1e-3), "S2": dict(epochs=12, lr=3e-4),
    "S3": dict(epochs=12, lr=1e-4), "S4": dict(epochs=12, lr=1e-4),
}


def build_model(strategy, n_classes, smoke):
    torch, nn, torchvision, _ = torch_bits()
    arch = "resnet18" if smoke else "resnet50"
    pretrained = (strategy != "S0") and not smoke
    weights = None
    if pretrained:
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    m = getattr(torchvision.models, arch)(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    for p in m.parameters():
        p.requires_grad = True
    if strategy == "S1":
        for n_, p in m.named_parameters():
            p.requires_grad = n_.startswith("fc.")
    elif strategy == "S2":
        for n_, p in m.named_parameters():
            p.requires_grad = n_.startswith("fc.") or n_.startswith("layer4.")
    return m.to(device())


def set_train_mode(m, strategy):
    """Frozen parts stay in eval mode so their BatchNorm statistics do not drift."""
    if strategy == "S1":
        m.eval(); m.fc.train()
    elif strategy == "S2":
        m.eval(); m.layer4.train(); m.fc.train()
    else:
        m.train()


def train_model(m, strategy, imgs, labels, epochs, lr, factors, smoke, bs=32, val=None, tag=""):
    torch, nn, _, _ = torch_bits()
    dev = device()
    ds = ImgDS(imgs, labels, factors=factors, train=True)
    dl = make_loader(ds, bs, True, smoke)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    steps = max(1, epochs * len(dl))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        set_train_mode(m, strategy)
        tot, n = 0.0, 0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(m(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(y); n += len(y)
        msg = f"  {tag}{strategy} epoch {ep + 1}/{epochs} loss {tot / max(n, 1):.3f}"
        if val is not None and (ep + 1) % 4 == 0:
            msg += f" val acc {evaluate(m, val[0], val[1], 1, smoke)[0]:.3f}"
        log(msg)
    return m


def predict(m, imgs, labels, factor, smoke):
    torch = torch_bits()[0]
    dev = device()
    dl = make_loader(ImgDS(imgs, labels, factors=(factor,), train=False), 64, False, smoke)
    m.eval(); preds = []
    with torch.no_grad():
        for x, _ in dl:
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                preds.append(m(x.to(dev)).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def evaluate(m, imgs, labels, factor, smoke):
    from sklearn.metrics import f1_score
    p = predict(m, imgs, labels, factor, smoke)
    y = np.array(labels)
    return float((p == y).mean()), float(f1_score(y, p, average="macro"))


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------
def cmd_prepare(a):
    for d in (DATA, RES, FIG):
        d.mkdir(parents=True, exist_ok=True)
    imgs, names = load_ucm(a.smoke)
    counts = pd.Series(names).value_counts().sort_index()
    counts.to_csv(RES / "ucm_class_counts.csv", header=["images"])
    log("class counts saved; min/max per class:", counts.min(), counts.max())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = ["forest", "denseresidential", "freeway", "river"]
    fig, axs = plt.subplots(len(show), len(FACTORS), figsize=(9, 7.2))
    for r, c in enumerate(show):
        im = imgs[names.index(c)]
        for k, f in enumerate(FACTORS):
            axs[r, k].imshow(degrade(im, f), interpolation="nearest")
            axs[r, k].set_xticks([]); axs[r, k].set_yticks([])
            if r == 0:
                axs[r, k].set_title(f"{BASE_GSD_M * f:.1f} m pixels", fontsize=10)
            if k == 0:
                axs[r, k].set_ylabel(c, fontsize=10)
    fig.suptitle("Same tile at five simulated pixel sizes (pixel size only is simulated)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_gsd_examples.png", dpi=170)
    log("saved", FIG / "fig1_gsd_examples.png")


def cmd_exp_a(a):
    for d in (RES, FIG):
        d.mkdir(parents=True, exist_ok=True)
    imgs, names = load_ucm(a.smoke)
    y = np.array([UCM_CLASSES.index(n) for n in names])
    strategies = ["S1", "S2", "S3", "S4"] + (["S0"] if a.with_scratch else [])
    rows = []
    for seed in a.seeds:
        set_seed(seed)
        tr, va, te = stratified_split(y, seed)
        log(f"seed {seed}: train {len(tr)}, val {len(va)}, test {len(te)}")
        for s in strategies:
            set_seed(seed)
            cfg = dict(CFG[s]);
            if a.smoke: cfg["epochs"] = 1
            m = build_model(s, len(UCM_CLASSES), a.smoke)
            fac = S4_TRAIN_FACTORS if s == "S4" else [1]
            train_model(m, s, [imgs[i] for i in tr], y[tr].tolist(), cfg["epochs"], cfg["lr"], fac, a.smoke,
                        val=([imgs[i] for i in va], y[va].tolist()), tag=f"[seed {seed}] ")
            for f in FACTORS:
                acc, f1 = evaluate(m, [imgs[i] for i in te], y[te].tolist(), f, a.smoke)
                rows.append(dict(seed=seed, strategy=s, factor=f, gsd_m=BASE_GSD_M * f, acc=acc, macro_f1=f1))
                log(f"    {s} @ {BASE_GSD_M * f:.1f} m: acc {acc:.3f}  macro-F1 {f1:.3f}")
            del m
        pd.DataFrame(rows).to_csv(RES / "exp_a.csv", index=False)
    log("saved", RES / "exp_a.csv")


def ucm_five_class(smoke, seed):
    imgs, names = load_ucm(smoke)
    rng = np.random.default_rng(seed)
    by = {c: [] for c in FIVE}
    for im, n in zip(imgs, names):
        if n in UCM_TO_5:
            by[UCM_TO_5[n]].append(im)
    n_min = min(len(v) for v in by.values())            # = 100 -> balance by sampling 100 residential of 300
    X, y = [], []
    for k, c in enumerate(FIVE):
        idx = rng.permutation(len(by[c]))[:n_min]
        X += [by[c][i] for i in idx]; y += [k] * len(idx)
    return X, np.array(y)


def euro_xy(pools, per_class):
    X, y = [], []
    for k, c in enumerate(FIVE):
        X += pools[c][:per_class]; y += [k] * len(pools[c][:per_class])
    return X, np.array(y)


def cmd_exp_b(a):
    for d in (RES, FIG):
        d.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in a.seeds:
        set_seed(seed)
        X, y = ucm_five_class(a.smoke, seed)
        tr, va, te = stratified_split(y, seed)
        pools = load_eurosat(400 if not a.smoke else 20, seed, a.smoke, a.eurosat_dir)
        Xe, ye = euro_xy(pools, 400 if not a.smoke else 20)
        log(f"seed {seed}: UCM-5 train {len(tr)} test {len(te)}; EuroSAT test {len(Xe)}")
        for s in ["S1", "S3", "S4"]:
            set_seed(seed)
            cfg = dict(CFG[s])
            if a.smoke: cfg["epochs"] = 1
            m = build_model(s, len(FIVE), a.smoke)
            fac = S4_TRAIN_FACTORS if s == "S4" else [1]
            train_model(m, s, [X[i] for i in tr], y[tr].tolist(), cfg["epochs"], cfg["lr"], fac, a.smoke,
                        tag=f"[B seed {seed}] ")
            Xt, yt = [X[i] for i in te], y[te].tolist()
            for name, (xs, ys, f) in {"aerial 0.3 m (same domain)": (Xt, yt, 1),
                                      "aerial simulated 4.8 m": (Xt, yt, 16),
                                      "satellite 10 m (real, EuroSAT)": (Xe, ye.tolist(), 1)}.items():
                p = predict(m, xs, ys, f, a.smoke)
                acc = float((p == np.array(ys)).mean())
                rows.append(dict(seed=seed, strategy=s, test_set=name, acc=acc))
                log(f"    {s} on {name}: acc {acc:.3f}")
                if name.startswith("satellite"):
                    np.savez(RES / f"exp_b_euro_preds_{s}_seed{seed}.npz", y_true=np.array(ys), y_pred=p)
            del m
        pd.DataFrame(rows).to_csv(RES / "exp_b.csv", index=False)
    log("saved", RES / "exp_b.csv")


def cmd_exp_c(a):
    import copy
    for d in (RES, FIG):
        d.mkdir(parents=True, exist_ok=True)
    ks = [0, 5, 10, 20, 50]
    rows = []
    for seed in a.seeds:
        set_seed(seed)
        X, y = ucm_five_class(a.smoke, seed)
        tr, va, te = stratified_split(y, seed)
        pools_all = load_eurosat(450 if not a.smoke else 40, seed, a.smoke, a.eurosat_dir)
        test_pools = {c: pools_all[c][:400 if not a.smoke else 20] for c in FIVE}          # same as exp_b
        adapt_pools = {c: pools_all[c][400 if not a.smoke else 20:] for c in FIVE}         # disjoint pool
        Xe, ye = euro_xy(test_pools, 10 ** 6)
        cfg = dict(CFG["S3"])
        if a.smoke: cfg["epochs"] = 1
        base = build_model("S3", len(FIVE), a.smoke)
        train_model(base, "S3", [X[i] for i in tr], y[tr].tolist(), cfg["epochs"], cfg["lr"], [1], a.smoke,
                    tag=f"[C seed {seed}] ")
        for k in ks:
            k_eff = min(k, len(adapt_pools[FIVE[0]]))
            Xa, ya = euro_xy(adapt_pools, k_eff) if k_eff > 0 else ([], np.array([], dtype=int))
            for init in ("aerial-pretrained (UC Merced then adapt)", "ImageNet only (adapt directly)"):
                if init.startswith("ImageNet") and k == 0:
                    continue                                  # an untrained head is just chance
                set_seed(seed)
                if init.startswith("aerial"):
                    m = copy.deepcopy(base)
                else:
                    m = build_model("S3", len(FIVE), a.smoke)
                if k_eff > 0:
                    train_model(m, "S3", Xa, ya.tolist(), 1 if a.smoke else 20, 1e-4, [1], a.smoke, bs=16,
                                tag=f"[C k={k} {init[:6]}] ")
                acc = evaluate(m, Xe, ye.tolist(), 1, a.smoke)[0]
                rows.append(dict(seed=seed, k_per_class=k, init=init, acc=acc))
                log(f"    k={k:>2} {init}: acc {acc:.3f}")
                del m
        pd.DataFrame(rows).to_csv(RES / "exp_c.csv", index=False)
    log("saved", RES / "exp_c.csv")


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e5e5e0", lw=0.8)
    ax.set_axisbelow(True)


def cmd_figures(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    if (RES / "exp_a.csv").exists():
        df = pd.read_csv(RES / "exp_a.csv")
        g = df.groupby(["strategy", "factor"])["acc"].agg(["mean", "std", "count"]).reset_index()
        g["std"] = g["std"].fillna(0)
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        xs = np.arange(len(FACTORS))
        for s in sorted(g.strategy.unique()):
            d = g[g.strategy == s].sort_values("factor")
            col, mk, ls = STYLE[s]
            ax.plot(xs, d["mean"] * 100, color=col, marker=mk, ls=ls, lw=2, ms=7, label=NAMES[s])
            ax.fill_between(xs, (d["mean"] - d["std"]) * 100, (d["mean"] + d["std"]) * 100, color=col, alpha=0.12, lw=0)
        ax.axvline(len(S4_TRAIN_FACTORS) - 0.5, color="#999", ls="--", lw=1)
        ax.text(len(S4_TRAIN_FACTORS) - 0.45, 4, "S4 never saw\nthis pixel size", fontsize=8.5, color="#666")
        ax.set_xticks(xs); ax.set_xticklabels(GSD_LABELS)
        ax.set_xlabel("Pixel size of the test image (simulated, coarser = higher flight)")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_ylim(0, 100)
        n_seeds = int(g["count"].max())
        ax.set_title(f"Experiment A: accuracy as pixel size grows (UC Merced, mean and 1 SD over {n_seeds} seeds)", fontsize=10.5)
        ax.legend(frameon=False, fontsize=8.5, loc="lower left")
        _style(ax); fig.tight_layout(); fig.savefig(FIG / "fig2_expA_accuracy_vs_gsd.png", dpi=170); plt.close(fig)
        piv = g.pivot(index="strategy", columns="factor", values="mean") * 100
        piv.columns = GSD_LABELS
        piv["drop_0.3_to_4.8_pts"] = piv[GSD_LABELS[0]] - piv[GSD_LABELS[-1]]
        piv.round(1).to_csv(RES / "exp_a_summary.csv")
        log("saved fig2 and exp_a_summary.csv")
    if (RES / "exp_b.csv").exists():
        d = pd.read_csv(RES / "exp_b.csv")
        order = ["aerial 0.3 m (same domain)", "aerial simulated 4.8 m", "satellite 10 m (real, EuroSAT)"]
        strategies = ["S1", "S3", "S4"]
        g = d.groupby(["strategy", "test_set"])["acc"].agg(["mean", "std"]).fillna(0).reset_index()
        fig, ax = plt.subplots(figsize=(8.2, 4.6))
        w = 0.25
        for i, s in enumerate(strategies):
            vals = [g[(g.strategy == s) & (g.test_set == t)]["mean"].values[0] * 100 for t in order]
            errs = [g[(g.strategy == s) & (g.test_set == t)]["std"].values[0] * 100 for t in order]
            b = ax.bar(np.arange(3) + (i - 1) * w, vals, w * 0.92, yerr=errs, color=STYLE[s][0], capsize=3,
                       label=NAMES[s], error_kw=dict(lw=1, ecolor="#444"))
            for rect, v in zip(b, vals):
                ax.text(rect.get_x() + rect.get_width() / 2, v + 2.5, f"{v:.0f}", ha="center", fontsize=8.5)
        ax.axhline(20, color="#999", ls="--", lw=1)
        ax.text(2.45, 21.5, "chance (5 classes)", fontsize=8, color="#666", ha="right")
        ax.set_xticks(range(3)); ax.set_xticklabels(["same domain\n(aerial, 0.3 m)", "simulated shift\n(aerial, 4.8 m)", "real shift\n(satellite, 10 m)"])
        ax.set_ylabel("Accuracy (%) on 5 shared classes"); ax.set_ylim(0, 108)
        ax.set_title("Experiment B: simulated pixel-size shift versus a real sensor shift", fontsize=10.5)
        ax.legend(frameon=False, fontsize=8.5, loc="upper right", bbox_to_anchor=(1.0, 0.93))
        _style(ax); fig.tight_layout(); fig.savefig(FIG / "fig3_expB_sim_vs_real.png", dpi=170); plt.close(fig)
        npz = sorted(RES.glob("exp_b_euro_preds_S3_seed*.npz"))
        if npz:
            from sklearn.metrics import confusion_matrix
            z = np.load(npz[0]); cm = confusion_matrix(z["y_true"], z["y_pred"], labels=range(len(FIVE)))
            cmn = cm / cm.sum(1, keepdims=True)
            fig, ax = plt.subplots(figsize=(5.4, 4.6))
            im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            for i in range(len(FIVE)):
                for j in range(len(FIVE)):
                    ax.text(j, i, f"{cmn[i, j] * 100:.0f}", ha="center", va="center", fontsize=9,
                            color="white" if cmn[i, j] > 0.5 else "black")
            ax.set_xticks(range(len(FIVE))); ax.set_xticklabels(FIVE, rotation=35, ha="right")
            ax.set_yticks(range(len(FIVE))); ax.set_yticklabels(FIVE)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True (EuroSAT, 10 m)")
            ax.set_title("S3 on real satellite tiles: row-normalised (%), first seed", fontsize=10)
            fig.tight_layout(); fig.savefig(FIG / "fig4_expB_confusion_eurosat.png", dpi=170); plt.close(fig)
        log("saved fig3 and fig4")
    if (RES / "exp_c.csv").exists():
        d = pd.read_csv(RES / "exp_c.csv")
        g = d.groupby(["init", "k_per_class"])["acc"].agg(["mean", "std"]).fillna(0).reset_index()
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        cols = {"aerial-pretrained (UC Merced then adapt)": STYLE["S3"], "ImageNet only (adapt directly)": STYLE["S1"]}
        for init, (col, mk, ls) in cols.items():
            q = g[g.init == init].sort_values("k_per_class")
            if q.empty:
                continue
            ax.plot(q.k_per_class, q["mean"] * 100, color=col, marker=mk, lw=2, ms=7, label=init)
            ax.fill_between(q.k_per_class, (q["mean"] - q["std"]) * 100, (q["mean"] + q["std"]) * 100, color=col, alpha=0.12, lw=0)
        ax.axhline(20, color="#999", ls="--", lw=1)
        ax.set_xlabel("Labelled satellite images per class used for adaptation")
        ax.set_ylabel("Accuracy (%) on EuroSAT test tiles"); ax.set_ylim(0, 100)
        ax.set_title("Experiment C: how many target images does it take to bridge the gap?", fontsize=10.5)
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")
        _style(ax); fig.tight_layout(); fig.savefig(FIG / "fig5_expC_adaptation.png", dpi=170); plt.close(fig)
        log("saved fig5")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["prepare", "exp_a", "exp_b", "exp_c", "figures", "all"])
    p.add_argument("--smoke", action="store_true", help="fast test on synthetic data, no downloads")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--with-scratch", action="store_true", help="also train S0 from scratch in Experiment A")
    p.add_argument("--eurosat-dir", default=None, help="folder with an already downloaded EuroSAT RGB copy")
    a = p.parse_args()
    global ROOT, DATA, RES, FIG
    if a.smoke:
        ROOT = Path(os.environ.get("RS_ROOT", "./work_smoke"))
        DATA, RES, FIG = ROOT / "data", ROOT / "results", ROOT / "figures"
        a.seeds = a.seeds[:2]
    cmds = {"prepare": cmd_prepare, "exp_a": cmd_exp_a, "exp_b": cmd_exp_b, "exp_c": cmd_exp_c, "figures": cmd_figures}
    if a.command == "all":
        for c in ["prepare", "exp_a", "exp_b", "exp_c", "figures"]:
            cmds[c](a)
    else:
        cmds[a.command](a)


if __name__ == "__main__":
    main()
