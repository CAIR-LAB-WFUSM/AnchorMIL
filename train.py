"""Train AnchorMIL. Hyper-parameters default to Table 2 of the paper."""
import argparse
import json
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from anchormil import AnchorMIL, AnchorMILLoss
from anchormil.data import FeatureBagDataset, collate_bags
from anchormil.metrics import evaluate


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, rs, ps = [], [], []
    for x, m, y in loader:
        o = model(x.to(device), m.to(device))
        ys.append(y.numpy())
        rs.append(o["reg"].cpu().numpy())
        ps.append(o["prob"].cpu().numpy())
    return np.concatenate(ys), np.concatenate(rs), np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="slide_id,patient_id,odx,split (train/val/test)")
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--out_dir", default="runs/anchormil")
    ap.add_argument("--in_dim", type=int, default=768)
    ap.add_argument("--anchors", type=int, default=128)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=0.7)
    ap.add_argument("--tau", type=float, default=25.0)
    ap.add_argument("--max_patches", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out_dir, exist_ok=True)

    tr = FeatureBagDataset(a.csv, a.feat_dir, "train", a.max_patches)
    va = FeatureBagDataset(a.csv, a.feat_dir, "val")
    te = FeatureBagDataset(a.csv, a.feat_dir, "test")
    tl = DataLoader(tr, a.batch_size, sampler=tr.risk_sampler(a.tau), collate_fn=collate_bags)
    vl = DataLoader(va, a.batch_size, collate_fn=collate_bags)
    el = DataLoader(te, a.batch_size, collate_fn=collate_bags)

    model = AnchorMIL(a.in_dim, a.in_dim, a.anchors, a.heads, a.blocks, a.dropout).to(dev)
    crit = AnchorMILLoss(a.beta, a.tau)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)

    def lr_at(ep):  # linear warmup, then cosine decay to min_lr
        if ep < a.warmup:
            return a.lr * (ep + 1) / a.warmup
        t = (ep - a.warmup) / max(1, a.epochs - a.warmup)
        return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * t))

    best, ckpt = -1.0, os.path.join(a.out_dir, "best.pt")
    for ep in range(a.epochs):
        for g in opt.param_groups:
            g["lr"] = lr_at(ep)
        model.train()
        tot = 0.0
        for x, m, y in tl:
            x, m, y = x.to(dev), m.to(dev), y.to(dev)
            loss, _ = crit(model(x, m), y, model.anchor_scores)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
        v = evaluate(*predict(model, vl, dev), tau=a.tau)
        print(f"ep {ep + 1:02d} loss {tot / len(tr):.4f} val auc {v['auc']:.3f} mae {v['mae']:.2f}")
        if v["auc"] >= best:
            best = v["auc"]
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=dev))
    res = evaluate(*predict(model, el, dev), tau=a.tau)
    print("test:", json.dumps(res, indent=2))
    with open(os.path.join(a.out_dir, "test_metrics.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
