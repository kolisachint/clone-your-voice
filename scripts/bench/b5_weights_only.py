#!/usr/bin/env python3
"""B5: does torch.load(weights_only=True) load real RVC v2 / Applio .pth files?

For each given .pth or .zip (zips: only members ending in .pth are read, into
memory/temp, never extractall) it:
  1. statically scans the pickle (pickletools, no execution) for GLOBAL /
     STACK_GLOBAL imports -> which classes the pickle references;
  2. torch.load(weights_only=True) in-process (inputs are untrusted but this
     mode refuses arbitrary callables; run in a throwaway env anyway);
  3. lists top-level keys + value types, recursively counts non-tensor,
     non-primitive objects;
  4. does the DESIGN 6.1 sanitize round trip (re-save tensors + plain config,
     reload with weights_only=True) and checks tensors are bit-identical.

Run:  python -I b5_weights_only.py FILE_OR_ZIP [...] [--json out.json]
Deps: torch (CPU is fine).
"""

import argparse
import io
import json
import pathlib
import pickletools
import zipfile

import torch

PRIM = (int, float, str, bool, type(None), bytes)


def pickle_globals(pth_bytes):
    out = set()
    with zipfile.ZipFile(io.BytesIO(pth_bytes)) as z:
        names = [n for n in z.namelist() if n.endswith("data.pkl")]
        for n in names:
            strs = []
            for op, arg, _ in pickletools.genops(z.read(n)):
                if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8"):
                    strs.append(arg)
                if op.name == "GLOBAL":
                    out.add(arg.replace(" ", "."))
                elif op.name == "STACK_GLOBAL" and len(strs) >= 2:
                    out.add(f"{strs[-2]}.{strs[-1]}")
    return sorted(out)


def walk(o, path="", stats=None):
    stats = stats if stats is not None else {"tensors": 0, "prims": 0, "containers": 0, "other": {}}
    if isinstance(o, torch.Tensor):
        stats["tensors"] += 1
    elif isinstance(o, PRIM):
        stats["prims"] += 1
    elif isinstance(o, dict):
        stats["containers"] += 1
        for k, v in o.items():
            walk(v, f"{path}/{k}", stats)
    elif isinstance(o, (list, tuple)):
        stats["containers"] += 1
        for i, v in enumerate(o):
            walk(v, f"{path}[{i}]", stats)
    else:
        stats["other"][f"{path}:{type(o).__module__}.{type(o).__qualname__}"] = 1
    return stats


def sanitize(ck):
    def clean(o):
        if isinstance(o, torch.Tensor):
            return o.detach().cpu().contiguous()
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, PRIM):
            return o
        raise TypeError(type(o))

    return clean(ck)


def same(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def test(label, data):
    r = {"file": label, "bytes": len(data), "pickle_globals": pickle_globals(data)}
    try:
        ck = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except Exception as e:
        r["load_ok"] = False
        r["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        return r
    r["load_ok"] = True
    r["top_level"] = {
        k: (type(v).__name__ if not isinstance(v, dict) else f"dict[{len(v)}]")
        for k, v in ck.items()
    }
    st = walk(ck)
    r["n_tensors"], r["n_primitives"] = st["tensors"], st["prims"]
    r["non_tensor_non_primitive"] = sorted(st["other"])
    r["config"] = ck.get("config")
    for k in ("version", "sr", "f0", "info"):
        if k in ck:
            r[k] = ck[k]
    san = sanitize(ck)
    buf = io.BytesIO()
    torch.save(san, buf)
    buf.seek(0)
    back = torch.load(buf, weights_only=True)
    r["sanitize_roundtrip_identical"] = same(ck, back)
    r["sanitized_bytes"] = buf.getbuffer().nbytes
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    res = []
    for f in a.files:
        p = pathlib.Path(f)
        if p.suffix == ".zip":
            with zipfile.ZipFile(p) as z:
                for n in z.namelist():
                    if n.endswith(".pth") and not n.endswith("/"):
                        res.append(test(f"{p.name}!{n}", z.read(n)))
        else:
            res.append(test(p.name, p.read_bytes()))
    res.append({"torch": torch.__version__})
    print(json.dumps(res, indent=1, default=str))
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
