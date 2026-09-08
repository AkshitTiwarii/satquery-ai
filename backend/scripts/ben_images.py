"""reBEN LMDB -> RGB PIL, the Qwen loader, shared by the harnesses and the
training notebook (it is embedded there, so it cannot import satquery).

MEASURED, not assumed: an LMDB value is a SAFETENSORS DICT keyed by band
name. The Lithuania/Summer LMDB (re-measured 5 Sep) holds TWELVE Sentinel-2
bands per S2 entry (B01-B12 + B8A, the 60 m ones downsampled), at native
resolution - so RGB is picked BY NAME (B04/B03/B02), never by index, then
percentile-stretched and cast to 8-bit. Get that wrong and nothing raises;
you train on black or false-colour tiles and find out at the eval.

It ALSO holds one Sentinel-1 entry per patch, keyed by the annotation table's
`s1_name`, value {'VV', 'VH'}: 120x120 float32, already in dB (VV ~ -18, VH
~ -28 on this cell). 8,775 of each, one-to-one. That is co-registered
optical + SAR for every patch - the fusion capability's training data.
"""

# Fixed dB windows for the SAR pseudo-RGB. FIXED, not per-tile percentiles:
# a per-tile stretch would make a calm lake and a rough sea look the same,
# and the whole point of SAR for water/urban is the absolute backscatter.
# Ranges from this cell's measured min/max (VV -29..-12, VH -42..-20) with
# margin so other cells fit.
SAR_DB_WINDOW = {"VV": (-25.0, 0.0), "VH": (-32.0, -5.0), "RATIO": (0.0, 20.0)}

import glob
import os

import lmdb
import numpy as np
from PIL import Image
from safetensors.numpy import load as st_load


def _pick(keys, wanted):
    for k in keys:
        if k.upper().replace("0", "") == wanted.upper().replace("0", ""):
            return k
    return None


class BENImages:
    def __init__(self, lmdb_dir):
        cands = [p for p in glob.glob(os.path.join(lmdb_dir, "**", "*"), recursive=True)
                 if os.path.isdir(p) and os.path.exists(os.path.join(p, "data.mdb"))]
        if os.path.exists(os.path.join(lmdb_dir, "data.mdb")):
            cands.insert(0, lmdb_dir)
        if not cands:
            raise FileNotFoundError("no data.mdb under %s" % lmdb_dir)
        self.path = cands[0]
        self.env = lmdb.open(self.path, readonly=True, lock=False, readahead=False, meminit=False)
        # MEASURED 4 Sep: the Lithuania/Summer LMDB holds Sentinel-1 entries
        # too - keyed by s1_name, value {'VV', 'VH'} - interleaved with the
        # Sentinel-2 ones. The first entry was one of them, and a band picker
        # that trusted it refused every optical patch. So pick the RGB names
        # from the first entry that HAS B04, and keep the SAR keys for fusion.
        # Keys sort alphabetically, so every S1A_/S1B_ entry precedes every
        # S2A_/S2B_ one: a linear scan of the first few thousand sees only
        # SAR. Seek by prefix instead.
        self.s2_keys, self.s1_keys = None, None
        with self.env.begin() as txn:
            self.n = txn.stat()["entries"]
            cur = txn.cursor()
            if cur.set_range(b"S2"):
                self.s2_keys = sorted(st_load(bytes(cur.value())))
            if cur.set_range(b"S1"):
                keys = sorted(st_load(bytes(cur.value())))
                if _pick(keys, "VV"):
                    self.s1_keys = keys
        assert self.s2_keys and _pick(self.s2_keys, "B04"), (
            "no S2 entry with B04 - not a reBEN S2 LMDB? first S2 keys: %s" % self.s2_keys)
        keys = self.s2_keys
        self.rgb = [_pick(keys, "B04"), _pick(keys, "B03"), _pick(keys, "B02")]
        assert all(self.rgb), "no B04/B03/B02 among %s - refusing positional fallback" % keys

    def get_sar_rgb(self, s1_name, size=224):
        """Sentinel-1 as a 3-channel 8-bit PIL for the VLM: (VV dB, VH dB,
        VV-VH dB) each stretched over a FIXED window. Or None."""
        sar = self.get_sar(s1_name)
        if sar is None:
            return None
        return sar_to_rgb(*sar, size=size)

    def get_sar(self, s1_name):
        """(VV, VH) float32 arrays for a Sentinel-1 patch, or None. As stored,
        which on reBEN is already dB; speckle filtering is the tool's business."""
        with self.env.begin() as txn:
            raw = txn.get(s1_name.encode())
        if raw is None:
            return None
        d = st_load(bytes(raw))
        vv, vh = _pick(sorted(d), "VV"), _pick(sorted(d), "VH")
        if not (vv and vh):
            return None
        return d[vv].astype("float32"), d[vh].astype("float32")

    def has(self, patch_id):
        with self.env.begin() as txn:
            return txn.get(patch_id.encode()) is not None

    def get(self, patch_id, size=224):
        with self.env.begin() as txn:
            raw = txn.get(patch_id.encode())
        if raw is None:
            return None
        d = st_load(bytes(raw))
        out = []
        for b in self.rgb:
            ch = d[b].astype("float32")
            if ch.ndim == 3:
                ch = ch[0]
            lo, hi = np.percentile(ch, (2, 98))
            out.append(np.clip((ch - lo) / max(hi - lo, 1e-6), 0, 1))
        img = Image.fromarray((np.dstack(out) * 255).astype("uint8"), "RGB")
        return img.resize((size, size), Image.BILINEAR) if size else img


def sar_to_rgb(vv, vh, size=224):
    """VV/VH dB arrays -> 8-bit RGB PIL (VV, VH, VV-VH), fixed dB windows.

    Shared by the harness, the training notebook and the engine's fusion tool,
    so the adapter sees SAR rendered one way everywhere. If the input is
    linear intensity (all positive), convert to dB first; reBEN stores dB.
    """
    vv = np.asarray(vv, dtype="float32")
    vh = np.asarray(vh, dtype="float32")
    if vv.ndim == 3:
        vv = vv[0]
    if vh.ndim == 3:
        vh = vh[0]
    if vv.min() >= 0 and vh.min() >= 0:                 # linear -> dB
        vv = 10 * np.log10(np.clip(vv, 1e-6, None))
        vh = 10 * np.log10(np.clip(vh, 1e-6, None))
    chans = []
    for arr, key in ((vv, "VV"), (vh, "VH"), (vv - vh, "RATIO")):
        lo, hi = SAR_DB_WINDOW[key]
        chans.append(np.clip((arr - lo) / (hi - lo), 0, 1))
    img = Image.fromarray((np.dstack(chans) * 255).astype("uint8"), "RGB")
    return img.resize((size, size), Image.BILINEAR) if size else img
