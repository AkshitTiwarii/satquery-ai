"""DOFA on reBEN Lithuania/Summer: the one clean run, with the rows that make it readable.

    python scripts/dofa_reben.py --lmdb <dir> --parquet metadata.parquet --out <prefix>
                                 [--epochs 5] [--smoke]

What it measures (all on the parquet's own test split of this cell, 2,151 patches, 19-way
multi-label, BCE), as average precision:

    prior          AP of predicting each class's train prevalence - the floor
    probe_random   frozen RANDOM-INIT DOFA + linear head       - the control: if the
                   pretrained probe does not beat this, the weights are not being used
    probe          frozen pretrained DOFA + linear head         - what the encoder knows
    finetune       whole DOFA fine-tuned, 12 bands, native 10 m - what it learns here
    finetune@20m / @40m   the fine-tuned model on 2x / 4x downsampled test tiles - the
                   only scale rung 10 m data can test; finer rungs are not measurable

Macro AP is reported over classes with >= 20 test positives (17 of 19 classes exist in this
cell and five have < 100 patches in total); micro AP and every per-class AP are in the JSON.

Data facts, computed on 7 Sep from the files: the LMDB holds 8,775 Sentinel-2 entries keyed by
patch id, twelve bands each (B01-B12 + B8A, no B10), picked BY NAME; metadata.parquet
(Zenodo 10891137) has labels + split for every one of them (train 4,206 / val 2,418 /
test 2,151). The join is asserted, not assumed.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CLASSES = [
    'Agro-forestry areas', 'Arable land', 'Beaches, dunes, sands', 'Broad-leaved forest',
    'Coastal wetlands', 'Complex cultivation patterns', 'Coniferous forest',
    'Industrial or commercial units', 'Inland waters', 'Inland wetlands',
    'Land principally occupied by agriculture, with significant areas of natural vegetation',
    'Marine waters', 'Mixed forest', 'Moors, heathland and sclerophyllous vegetation',
    'Natural grassland and sparsely vegetated areas', 'Pastures', 'Permanent crops',
    'Transitional woodland, shrub', 'Urban fabric']
# Sentinel-2 central wavelengths in micrometres, in the band order we stack.
BANDS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
WAVES = [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.945, 1.610, 2.190]
SIZE = 120     # cache resolution (10 m native); the model sees 224
IMG = 224


def pick(keys, wanted):
    for k in keys:
        if k.upper().replace('0', '') == wanted.upper().replace('0', ''):
            return k
    raise KeyError('band %s not among %s' % (wanted, keys))


def load_labels(parquet):
    import pyarrow.parquet as pq
    rows = pq.read_table(parquet, columns=['patch_id', 'labels', 'split', 'country']).to_pylist()
    return {r['patch_id']: (r['labels'], r['split']) for r in rows if r['country'] == 'Lithuania'}


def load_lmdb(lmdb_dir, labels, limit=0):
    """-> ids, X uint16 [N,12,120,120], Y float32 [N,19], split list. Every band resized to
    120 px (the 20 m and 60 m bands are stored smaller) with bilinear, BY NAME."""
    import glob
    import lmdb
    import torch
    import torch.nn.functional as F
    from safetensors.numpy import load as st_load
    cands = [p for p in glob.glob(os.path.join(lmdb_dir, '**', '*'), recursive=True)
             if os.path.isdir(p) and os.path.exists(os.path.join(p, 'data.mdb'))]
    if os.path.exists(os.path.join(lmdb_dir, 'data.mdb')):
        cands.insert(0, lmdb_dir)
    assert cands, 'no data.mdb under ' + lmdb_dir
    env = lmdb.open(cands[0], readonly=True, lock=False, readahead=False, meminit=False)
    ids, X, Y, split = [], [], [], []
    cidx = {c: i for i, c in enumerate(CLASSES)}
    unjoined = 0
    with env.begin() as txn:
        cur = txn.cursor()
        if not cur.set_range(b'S2'):
            raise RuntimeError('no S2 entries')
        for k, v in cur:
            pid = k.decode()
            if not pid.startswith('S2'):
                break
            if pid not in labels:
                unjoined += 1
                continue
            d = st_load(bytes(v))
            keys = list(d.keys())
            planes = []
            for b in BANDS:
                a = np.asarray(d[pick(keys, b)]).astype(np.float32)
                t = torch.from_numpy(a)[None, None]
                if a.shape != (SIZE, SIZE):
                    t = F.interpolate(t, size=(SIZE, SIZE), mode='bilinear', align_corners=False)
                planes.append(t[0, 0].numpy())
            X.append(np.clip(np.stack(planes), 0, 65535).astype(np.uint16))
            y = np.zeros(len(CLASSES), np.float32)
            for l in labels[pid][0]:
                y[cidx[l]] = 1.0
            Y.append(y)
            ids.append(pid)
            split.append(labels[pid][1])
            if limit and len(ids) >= limit:
                break
    assert len(ids) > 0, 'nothing joined - LMDB keys and parquet patch_id disagree'
    if not limit:
        assert unjoined == 0, '%d S2 entries had no parquet row' % unjoined
    return ids, np.stack(X), np.stack(Y), np.array(split)


def average_precision(y_true, y_score):
    """AP per class as sklearn defines it: sum over DISTINCT score thresholds of
    (recall step) x precision. Ties are one threshold, so a constant score gives exactly
    the prevalence instead of an order-dependent number. No sklearn dependency."""
    aps = []
    for c in range(y_true.shape[1]):
        t, s = y_true[:, c].astype(np.float64), y_score[:, c]
        n_pos = t.sum()
        if n_pos == 0:
            aps.append(float('nan'))
            continue
        order = np.argsort(-s, kind='stable')
        t, s = t[order], s[order]
        last = np.r_[s[1:] != s[:-1], True]          # index of the last item at each threshold
        tp = np.cumsum(t)[last]
        prec = tp / (np.flatnonzero(last) + 1)
        rec = tp / n_pos
        rec_prev = np.r_[0.0, rec[:-1]]
        aps.append(float(((rec - rec_prev) * prec).sum()))
    return np.array(aps)


def micro_ap(y_true, y_score):
    return float(average_precision(y_true.reshape(-1, 1), y_score.reshape(-1, 1))[0])


def summarise(name, y_true, y_score, min_pos=20):
    ap = average_precision(y_true, y_score)
    npos = y_true.sum(0)
    keep = npos >= min_pos
    return {
        'row': name,
        'macro_ap_%dpos' % min_pos: float(np.nanmean(ap[keep])),
        'classes_in_macro': int(keep.sum()),
        'micro_ap': micro_ap(y_true, y_score),
        'per_class_ap': {c: (None if np.isnan(a) else round(float(a), 4)) for c, a in zip(CLASSES, ap)},
        'test_positives': {c: int(n) for c, n in zip(CLASSES, npos)},
    }


class Prep:
    """uint16 [B,12,120,120] on device -> standardised float [B,12,224,224]."""
    def __init__(self, mean, std, device):
        import torch
        self.mean = torch.tensor(mean, device=device).view(1, -1, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, -1, 1, 1)

    def __call__(self, xb, downsample=1, train=False):
        import torch
        import torch.nn.functional as F
        x = xb.float()
        if downsample > 1:   # simulate a coarser GSD: average-pool, then back up
            x = F.avg_pool2d(x, downsample)
        x = (x - self.mean) / self.std
        if train:
            if torch.rand(1).item() < 0.5:
                x = x.flip(-1)
            if torch.rand(1).item() < 0.5:
                x = x.flip(-2)
            k = int(torch.randint(0, 4, (1,)).item())
            x = torch.rot90(x, k, dims=(-2, -1))
        return F.interpolate(x, size=(IMG, IMG), mode='bilinear', align_corners=False)


def build(pretrained, device):
    import torch
    from dofa_model import DOFABase16_Weights, dofa_base_patch16_224
    w = DOFABase16_Weights.DOFA_MAE if pretrained else None
    m = dofa_base_patch16_224(weights=w, num_classes=len(CLASSES), global_pool=True)
    return m.to(device)


def features(model, X, prep, device, bs=64):
    import torch
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=device == 'cuda'):
        for i in range(0, len(X), bs):
            xb = prep(torch.from_numpy(X[i:i + bs]).to(device))
            out.append(model.forward_features(xb, WAVES).float().cpu())
    return torch.cat(out).numpy()


def probe(F_tr, Y_tr, F_te, epochs=60, device='cpu'):
    import torch
    mu, sd = F_tr.mean(0, keepdims=True), F_tr.std(0, keepdims=True) + 1e-6
    ftr = torch.from_numpy((F_tr - mu) / sd).to(device)
    fte = torch.from_numpy((F_te - mu) / sd).to(device)
    ytr = torch.from_numpy(Y_tr).to(device)
    head = torch.nn.Linear(ftr.shape[1], Y_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(epochs):
        perm = torch.randperm(len(ftr), device=device)
        for i in range(0, len(ftr), 256):
            idx = perm[i:i + 256]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(head(ftr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return torch.sigmoid(head(fte)).cpu().numpy()


def predict(model, X, prep, device, downsample=1, bs=64):
    import torch
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=device == 'cuda'):
        for i in range(0, len(X), bs):
            xb = prep(torch.from_numpy(X[i:i + bs]).to(device), downsample=downsample)
            out.append(torch.sigmoid(model(xb, WAVES)).float().cpu())
    return torch.cat(out).numpy()


def finetune(model, X_tr, Y_tr, X_va, Y_va, prep, device, epochs, bs=32, log=print):
    import math
    import torch
    head = [p for n, p in model.named_parameters() if n.startswith('head') or n.startswith('fc_norm')]
    body = [p for n, p in model.named_parameters() if not (n.startswith('head') or n.startswith('fc_norm'))]
    opt = torch.optim.AdamW([{'params': body, 'lr': 5e-5}, {'params': head, 'lr': 1e-3}], weight_decay=0.05)
    steps = epochs * math.ceil(len(X_tr) / bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    scaler = torch.amp.GradScaler('cuda', enabled=device == 'cuda')
    Ytr = torch.from_numpy(Y_tr)
    # Select on micro AP: macro is NaN whenever no class has 20 val positives (the smoke run,
    # any small cell), and a NaN never compares greater, so the "best" state stayed None and
    # the reload crashed (v2, 7 Sep). Start from the initial weights so there is always one.
    best = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(len(X_tr))
        tot, n = 0.0, 0
        t0 = time.time()
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            xb = prep(torch.from_numpy(X_tr[idx]).to(device), train=True)
            yb = Ytr[idx].to(device)
            with torch.autocast('cuda', dtype=torch.float16, enabled=device == 'cuda'):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(xb, WAVES).float(), yb)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(idx); n += len(idx)
        va = summarise('va', Y_va, predict(model, X_va, prep, device))
        m = va['micro_ap']
        log('epoch %d  loss %.4f  val macroAP %.4f microAP %.4f  %.0fs' % (ep + 1, tot / n, va['macro_ap_20pos'], m, time.time() - t0))
        if m > best:
            best, best_state = m, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--lmdb', required=True)
    ap.add_argument('--parquet', required=True)
    ap.add_argument('--out', default='dofa_reben')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--smoke', action='store_true', help='128 patches, 1 epoch, prove the path')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    T0 = time.time()
    log_lines = []

    def log(s):
        print(s, flush=True); log_lines.append(s)

    labels = load_labels(a.parquet)
    log('parquet: %d Lithuania rows' % len(labels))
    ids, X, Y, split = load_lmdb(a.lmdb, labels, limit=128 if a.smoke else 0)
    log('joined: %d patches, X %s, positives/class min %d max %d, splits %s' % (
        len(ids), X.shape, int(Y.sum(0).min()), int(Y.sum(0).max()),
        {s: int((split == s).sum()) for s in np.unique(split)}))
    if not a.smoke:
        assert len(ids) == 8775, len(ids)
    tr, va, te = split == 'train', split == 'validation', split == 'test'
    if a.smoke and (tr.sum() < 8 or te.sum() < 8):
        tr = np.arange(len(ids)) < len(ids) // 2; te = ~tr; va = te
    epochs = 1 if a.smoke else a.epochs
    mean = X[tr].reshape(len(X[tr]), 12, -1).mean((0, 2)).tolist()
    std = (X[tr].reshape(len(X[tr]), 12, -1).astype(np.float32).std((0, 2)) + 1e-6).tolist()
    prep = Prep(mean, std, device)
    rows = []

    prior = np.tile(Y[tr].mean(0, keepdims=True), (te.sum(), 1))
    rows.append(summarise('prior', Y[te], prior))

    m0 = build(False, device)
    F0_tr, F0_te = features(m0, X[tr], prep, device), features(m0, X[te], prep, device)
    rows.append(summarise('probe_random', Y[te], probe(F0_tr, Y[tr], F0_te, device=device)))
    del m0

    m = build(True, device)
    F_tr, F_te = features(m, X[tr], prep, device), features(m, X[te], prep, device)
    rows.append(summarise('probe', Y[te], probe(F_tr, Y[tr], F_te, device=device)))
    log('probes done %.0fs' % (time.time() - T0))

    best_va = finetune(m, X[tr], Y[tr], X[va], Y[va], prep, device, epochs, log=log)
    preds = {'finetune': predict(m, X[te], prep, device),
             'finetune@20m': predict(m, X[te], prep, device, downsample=2),
             'finetune@40m': predict(m, X[te], prep, device, downsample=4)}
    for name, p in preds.items():
        rows.append(summarise(name, Y[te], p))
    # raw test predictions, so any row can be re-scored offline without the GPU
    np.savez_compressed(a.out + '_test_preds.npz', ids=np.array(ids)[te], y_true=Y[te], prior=prior,
                        **{k.replace('@', '_'): v for k, v in preds.items()})

    log('')
    log('%-14s %9s %9s %s' % ('row', 'macroAP', 'microAP', '(macro over classes with >=20 test positives)'))
    for r in rows:
        log('%-14s %9.3f %9.3f  %d classes' % (r['row'], r['macro_ap_20pos'], r['micro_ap'], r['classes_in_macro']))
    log('best val microAP during fine-tune (checkpoint selection): %.3f' % best_va)
    out = {'rows': rows, 'n': {'train': int(tr.sum()), 'val': int(va.sum()), 'test': int(te.sum())},
           'epochs': epochs, 'smoke': a.smoke, 'bands': BANDS, 'wavelengths_um': WAVES,
           'band_mean': mean, 'band_std': std, 'weights': 'torchgeo DOFABase16_Weights.DOFA_MAE',
           'device': torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu',
           'elapsed_min': round((time.time() - T0) / 60, 1), 'log': log_lines}
    with open(a.out + '.json', 'w') as fh:
        json.dump(out, fh, indent=1)
    if not a.smoke:
        torch.save(m.state_dict(), a.out + '_finetuned.pth')
    log('wrote %s.json in %.1f min' % (a.out, out['elapsed_min']))


if __name__ == '__main__':
    main()
