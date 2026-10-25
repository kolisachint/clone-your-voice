from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import stat
import sys
import zipfile
from pathlib import Path

import pytest
import torch

from cyv import packs, paths
from cyv.packs import PackError

PY = Path(sys.executable)
SCHEMA = Path(packs.__file__).parent / "voice_schema.json"


def applio_ckpt() -> dict:
    """Shaped like rvc/train/process/extract_model.py output."""
    return {
        "weight": {
            "emb_g.weight": torch.zeros(1, 4).half(),
            "dec.conv_pre.weight": torch.ones(2, 2),
        },
        "config": [513, 32, 192, 192, 768, 2, 6, 3, 0.0, "1", [3, 7], [[1, 3], [1, 3]],
                   [10, 6], 512, [20, 12], 1, 256, 40000],
        "epoch": 10, "step": 100, "sr": 40000, "f0": True, "version": "v2",
        "creation_date": "2026-01-01T00:00:00", "model_hash": "ab", "dataset_length": 12.5,
        "model_name": "x", "author": None, "embedder_model": "contentvec", "speakers_id": 1,
        "vocoder": "HiFi-GAN",
    }  # fmt: skip


def torch_bytes(obj: object) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


class Evil:
    marker: str = ""

    def __reduce__(self):
        return (os.system, (f"touch {self.marker}",))


def manifest_for(pth: bytes, index: bytes, name: str = "alice") -> dict:
    return {
        "schema_version": 1, "name": name, "engine": "applio-rvc",
        "engine_version": "c817f1f83f689ef72e0dfeb1c17ae990b9248506", "rvc_version": "v2",
        "sample_rate": 40000, "embedder": "contentvec", "f0_method": "rmvpe",
        "vocoder": "HiFi-GAN", "epochs": 10, "dataset_seconds": 600.0,
        "tts_base": {"voice": "af_heart", "lang": "a"}, "pitch_shift": 0,
        "sha256": {
            "model.pth": hashlib.sha256(pth).hexdigest(),
            "model.index": hashlib.sha256(index).hexdigest(),
        },
    }  # fmt: skip


CONSENT = {
    "attestation": "I own this voice.",
    "attestation_version": 1,
    "owner_name": "Alice",
    "created_at": "2026-01-01T00:00:00Z",
}


def build_pack(
    dest: Path,
    *,
    pth: bytes | None = None,
    index: bytes = b"INDEX",
    manifest: dict | None = None,
    consent: dict | None = None,
    drop: tuple[str, ...] = (),
    extra: dict[str, bytes] | None = None,
) -> Path:
    pth = torch_bytes(applio_ckpt()) if pth is None else pth
    man = manifest if manifest is not None else manifest_for(pth, index)
    members = {
        "model.pth": pth,
        "model.index": index,
        "voice.json": json.dumps(man).encode(),
        "consent.json": json.dumps(CONSENT if consent is None else consent).encode(),
    }
    for d in drop:
        members.pop(d)
    members.update(extra or {})
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for k, v in members.items():
            zf.writestr(k, v)
    return dest


@pytest.fixture(autouse=True)
def cyv_home(tmp_path, monkeypatch):
    monkeypatch.setenv("CYV_HOME", str(tmp_path / "home"))
    # Test-only: the fixture index is a placeholder, and .venv has no faiss.
    monkeypatch.setenv("CYV_SANITIZE_SKIP_INDEX", "1")
    return tmp_path / "home"


@pytest.fixture
def zpath(tmp_path) -> Path:
    return tmp_path / "p.cyvpack.zip"


def do_import(p: Path, **kw):
    return packs.import_pack(p, daemon_python=PY, **kw)


def no_leftovers() -> None:
    root = paths.voices_dir()
    assert [d.name for d in root.iterdir() if d.name.startswith(".")] == []


def test_happy_path(zpath):
    info = do_import(build_pack(zpath))
    assert info.name == "alice" and info.dir == paths.voices_dir() / "alice"
    assert {p.name for p in info.dir.iterdir()} == {
        "model.pth", "model.index", "voice.json", "consent.json", "installed.json",
    }  # fmt: skip
    ck = torch.load(info.dir / "model.pth", weights_only=True)
    assert ck["version"] == "v2" and ck["config"][-1] == 40000
    assert set(ck["weight"]) == {"emb_g.weight", "dec.conv_pre.weight"}
    assert packs.load_manifest(info.dir)["name"] == "alice"
    no_leftovers()


def test_sanitize_drops_unknown_keys(zpath):
    ck = applio_ckpt()
    ck["junk"] = {"a": 1}
    info = do_import(build_pack(zpath, pth=torch_bytes(ck)))
    assert "junk" not in torch.load(info.dir / "model.pth", weights_only=True)


def test_malicious_pickle(zpath, tmp_path):
    marker = tmp_path / "pwned"
    Evil.marker = str(marker)
    pth = torch_bytes({"weight": Evil(), "config": [], "sr": 1, "version": "v2"})
    with pytest.raises(PackError, match="rejected"):
        do_import(build_pack(zpath, pth=pth))
    assert not marker.exists()
    assert packs.list_voices() == []
    no_leftovers()


def test_not_a_checkpoint(zpath):
    with pytest.raises(PackError, match="rejected"):
        do_import(build_pack(zpath, pth=b"not a torch file"))


def test_missing_required_checkpoint_keys(zpath):
    with pytest.raises(PackError, match="rejected"):
        do_import(build_pack(zpath, pth=torch_bytes({"weight": {"a": torch.zeros(1)}})))


@pytest.mark.parametrize(
    "name", ["../evil", "/abs", "a/../../b", "sub/model.pth", "back\\slash", "C:evil"]
)
def test_zip_slip_variants(zpath, name):
    build_pack(zpath, extra={name: b"x"})
    with pytest.raises(PackError):
        do_import(zpath)
    assert packs.list_voices() == []
    no_leftovers()


def test_directory_member(zpath):
    build_pack(zpath, extra={"d/": b""})
    with pytest.raises(PackError, match="director"):
        do_import(zpath)


def test_symlink_member(zpath):
    build_pack(zpath)
    with zipfile.ZipFile(zpath, "a") as zf:
        zi = zipfile.ZipInfo("link")
        zi.create_system = 3
        zi.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(zi, "/etc/passwd")
    with pytest.raises(PackError):
        do_import(zpath)


def test_symlink_as_allowed_name(zpath):
    with zipfile.ZipFile(zpath, "w") as zf:
        for n in ("model.index", "voice.json", "consent.json"):
            zf.writestr(n, b"{}")
        zi = zipfile.ZipInfo("model.pth")
        zi.create_system = 3
        zi.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(zi, "/etc/passwd")
    with pytest.raises(PackError, match="regular"):
        do_import(zpath)


def test_duplicate_member(zpath):
    build_pack(zpath)
    with pytest.warns(UserWarning), zipfile.ZipFile(zpath, "a") as zf:
        zf.writestr("model.index", b"again")
    with pytest.raises(PackError, match="duplicate"):
        do_import(zpath)


def test_extra_member(zpath):
    with pytest.raises(PackError, match="unexpected"):
        do_import(build_pack(zpath, extra={"evil.py": b"x"}))


@pytest.mark.parametrize("member", ["model.pth", "model.index", "voice.json", "consent.json"])
def test_missing_member(zpath, member):
    with pytest.raises(PackError, match="missing"):
        do_import(build_pack(zpath, drop=(member,)))


def test_zip_bomb_member_cap(zpath):
    # 2 MiB of zeros compresses tiny but exceeds the voice.json cap.
    build_pack(zpath)
    bomb = zpath.parent / "bomb.zip"
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("model.pth", b"x")
        zf.writestr("model.index", b"x")
        zf.writestr("voice.json", b"\0" * (2 << 20))
        zf.writestr("consent.json", b"{}")
    with pytest.raises(PackError, match="too large"):
        do_import(bomb)
    no_leftovers()


def test_total_cap(zpath, monkeypatch):
    from cyv.packs import importer

    monkeypatch.setattr(importer, "MAX_TOTAL", 100)
    with pytest.raises(PackError, match="too large"):
        do_import(build_pack(zpath))


def test_not_a_zip(tmp_path):
    p = tmp_path / "x.zip"
    p.write_bytes(b"garbage")
    with pytest.raises(PackError, match="cannot read"):
        do_import(p)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "Bad Name"), ("name", "x" * 33), ("name", "../x"), ("name", ""),
        ("schema_version", 2), ("schema_version", True), ("engine", "other"),
        ("embedder", "hubert"), ("vocoder", "RefineGAN"), ("rvc_version", "v1"),
        ("sample_rate", 44100), ("f0_method", "crepe"), ("pitch_shift", 25),
        ("pitch_shift", 1.5), ("epochs", 0), ("engine_version", "XYZ"),
        ("tts_base", {"voice": "bad", "lang": "a"}), ("tts_base", {"voice": "af_x"}),
        ("sha256", {"model.pth": "0" * 64}), ("bogus", 1),
    ],
)  # fmt: skip
def test_bad_manifest_fields(zpath, field, value):
    pth = torch_bytes(applio_ckpt())
    man = manifest_for(pth, b"INDEX")
    man[field] = value
    with pytest.raises(PackError):
        do_import(build_pack(zpath, pth=pth, manifest=man))
    assert packs.list_voices() == []


def test_missing_manifest_field():
    man = manifest_for(b"a", b"b")
    del man["embedder"]
    with pytest.raises(PackError, match="embedder"):
        packs.validate_manifest(man)


def test_bad_consent(zpath):
    with pytest.raises(PackError, match="consent"):
        do_import(build_pack(zpath, consent={"attestation": "x"}))
    bad = dict(CONSENT, owner_name="")
    with pytest.raises(PackError, match="owner_name"):
        do_import(build_pack(zpath, consent=bad))


@pytest.mark.parametrize("member", ["model.pth", "model.index"])
def test_sha_mismatch(zpath, member):
    pth = torch_bytes(applio_ckpt())
    man = manifest_for(pth, b"INDEX")
    man["sha256"][member] = "f" * 64
    with pytest.raises(PackError, match="sha256 mismatch"):
        do_import(build_pack(zpath, pth=pth, manifest=man))


def test_overwrite_requires_force(zpath):
    build_pack(zpath)
    do_import(zpath)
    with pytest.raises(PackError, match="already exists"):
        do_import(zpath)
    assert do_import(zpath, force=True).name == "alice"
    no_leftovers()


def test_missing_daemon_python(zpath, tmp_path):
    with pytest.raises(PackError, match="daemon python"):
        packs.import_pack(build_pack(zpath), daemon_python=tmp_path / "nope")
    no_leftovers()


def test_list_remove_default(tmp_path):
    assert packs.list_voices() == [] and packs.get_default() is None
    for n in ("bob", "alice"):
        pth = torch_bytes(applio_ckpt())
        do_import(
            build_pack(tmp_path / f"{n}.zip", pth=pth, manifest=manifest_for(pth, b"INDEX", n))
        )
    assert [v.name for v in packs.list_voices()] == ["alice", "bob"]
    packs.set_default("bob")
    assert packs.get_default() == "bob"
    assert (paths.home() / "default_voice").read_text().strip() == "bob"
    with pytest.raises(PackError):
        packs.set_default("nobody")
    with pytest.raises(PackError):
        packs.set_default("../x")
    packs.remove_voice("bob")
    assert packs.get_default() is None
    assert [v.name for v in packs.list_voices()] == ["alice"]
    with pytest.raises(PackError):
        packs.remove_voice("bob")


def test_validator_matches_json_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA.read_text())
    good = manifest_for(b"a", b"b")
    jsonschema.validate(good, schema)
    packs.validate_manifest(good)

    def mutate(fn):
        m = copy.deepcopy(good)
        fn(m)
        return m

    cases = [
        mutate(lambda m: m.update(name="BAD")),
        mutate(lambda m: m.update(schema_version=2)),
        mutate(lambda m: m.update(embedder="x")),
        mutate(lambda m: m.update(vocoder="x")),
        mutate(lambda m: m.update(rvc_version="v1")),
        mutate(lambda m: m.update(sample_rate=1)),
        mutate(lambda m: m.update(pitch_shift=99)),
        mutate(lambda m: m.update(epochs=0)),
        mutate(lambda m: m.update(dataset_seconds=-1)),
        mutate(lambda m: m.update(extra=1)),
        mutate(lambda m: m.pop("sha256")),
        mutate(lambda m: m["tts_base"].update(lang="ab")),
        mutate(lambda m: m["tts_base"].pop("voice")),
        mutate(lambda m: m["sha256"].update(**{"model.pth": "zz"})),
        mutate(lambda m: m.update(engine_version="nothex")),
        mutate(lambda m: m.update(license=3)),
    ]
    for bad in cases:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema)
        with pytest.raises(PackError):
            packs.validate_manifest(bad)
    for ok in (
        mutate(lambda m: m.update(license="CC-BY")),
        mutate(lambda m: m.pop("epochs")),
    ):
        jsonschema.validate(ok, schema)
        packs.validate_manifest(ok)


def test_cyv_packs_does_not_import_torch():
    import subprocess

    code = "import sys, cyv.packs; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-I", "-c", code], check=True)


def test_regexes_reject_trailing_newline():
    from cyv.packs import store

    man = manifest_for(b"a", b"b")
    for mutate in (
        lambda m: m.update(name="foo\n"),
        lambda m: m.update(engine_version="c817f1f\n"),
        lambda m: m["tts_base"].update(voice="af_heart\n"),
        lambda m: m["tts_base"].update(lang="a\n"),
        lambda m: m["sha256"].update({"model.pth": "0" * 64 + "\n"}),
    ):
        bad = copy.deepcopy(man)
        mutate(bad)
        with pytest.raises(PackError):
            packs.validate_manifest(bad)
    with pytest.raises(PackError):
        store._check_name("foo\n")


def test_deeply_nested_json(zpath):
    deep = b"[" * 100000 + b"]" * 100000
    pth = torch_bytes(applio_ckpt())
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("model.pth", pth)
        zf.writestr("model.index", b"I")
        zf.writestr("voice.json", deep)
        zf.writestr("consent.json", json.dumps(CONSENT))
    with pytest.raises(PackError):
        do_import(zpath)


@pytest.mark.parametrize(
    ("key", "value"),
    [("version", "v1"), ("sr", 32000), ("sr", "48k"), ("f0", 0),
     ("vocoder", "RefineGAN"), ("embedder_model", "hubert")],
)  # fmt: skip
def test_checkpoint_manifest_mismatch(zpath, key, value):
    ck = applio_ckpt()
    ck[key] = value
    with pytest.raises(PackError, match="rejected"):
        do_import(build_pack(zpath, pth=torch_bytes(ck)))
    assert packs.list_voices() == []


def test_sr_string_accepted(zpath):
    ck = applio_ckpt()
    ck["sr"] = "40k"
    assert do_import(build_pack(zpath, pth=torch_bytes(ck))).name == "alice"


def test_installed_json(zpath):
    info = do_import(build_pack(zpath))
    inst = json.loads((info.dir / "installed.json").read_text())
    assert set(inst) == {"sanitized_sha256", "source_sha256", "imported_at"}
    for m in ("model.pth", "model.index"):
        got = hashlib.sha256((info.dir / m).read_bytes()).hexdigest()
        assert inst["sanitized_sha256"][m] == got
    assert inst["source_sha256"] == packs.load_manifest(info.dir)["sha256"]
    assert info.warnings == ()


def test_engine_version_mismatch_warns(zpath):
    pth = torch_bytes(applio_ckpt())
    man = manifest_for(pth, b"INDEX")
    man["engine_version"] = "deadbeef"
    info = do_import(build_pack(zpath, pth=pth, manifest=man))
    assert info.warnings and "deadbeef" in info.warnings[0]


def test_stale_import_dirs_cleaned(zpath):
    root = paths.ensure_private_dir(paths.voices_dir())
    old, fresh = root / ".import-old", root / ".import-new"
    for d in (old, fresh):
        d.mkdir()
        (d / "x").write_text("x")
    t = os.stat(old).st_mtime - 7200
    os.utime(old, (t, t))
    do_import(build_pack(zpath))
    assert not old.exists() and fresh.exists()


def test_target_appears_before_rename(zpath, monkeypatch):
    from cyv.packs import importer

    real = importer._sanitize

    def racing(tmp, py):
        real(tmp, py)
        (paths.voices_dir() / "alice").mkdir()
        (paths.voices_dir() / "alice" / "f").write_text("x")

    monkeypatch.setattr(importer, "_sanitize", racing)
    with pytest.raises(PackError, match="cannot install"):
        do_import(build_pack(zpath))


def test_oversized_index_rejected(zpath, monkeypatch):
    from cyvd import sanitize as san

    monkeypatch.setattr(san, "MAX_INDEX_FILE", 3)
    with pytest.raises(san.SanitizeError, match="too large"):
        p = zpath.parent / "i.bin"
        p.write_bytes(b"12345")
        san.sanitize_index(p, zpath.parent / "o.bin")


def test_index_validation_without_skip(zpath, monkeypatch):
    """Without the test-only skip, a non-faiss index must be refused (or faiss missing)."""
    monkeypatch.delenv("CYV_SANITIZE_SKIP_INDEX")
    with pytest.raises(PackError, match="rejected"):
        do_import(build_pack(zpath))


def test_index_validation_with_faiss(tmp_path, monkeypatch):
    monkeypatch.delenv("CYV_SANITIZE_SKIP_INDEX")
    faiss = pytest.importorskip("faiss")
    np = pytest.importorskip("numpy")
    from cyvd import sanitize as san

    def write(d, kind="flat"):
        idx = faiss.IndexFlatL2(d)
        idx.add(np.zeros((3, d), dtype="float32"))
        p = tmp_path / f"{kind}{d}.index"
        faiss.write_index(idx, str(p))
        return p

    good = write(768)
    out = tmp_path / "out.index"
    san.sanitize_index(good, out)
    assert faiss.read_index(str(out)).ntotal == 3
    with pytest.raises(san.SanitizeError, match="dimension"):
        san.sanitize_index(write(64), tmp_path / "o2")
    hnsw = tmp_path / "hnsw.index"
    faiss.write_index(faiss.IndexHNSWFlat(768, 8), str(hnsw))
    with pytest.raises(san.SanitizeError, match="not allowed"):
        san.sanitize_index(hnsw, tmp_path / "o3")
    junk = tmp_path / "junk.index"
    junk.write_bytes(b"not faiss")
    with pytest.raises(san.SanitizeError):
        san.sanitize_index(junk, tmp_path / "o4")


def test_old_torch_refused(monkeypatch):
    from cyvd import sanitize as san

    monkeypatch.setattr(san.torch, "__version__", "2.5.1")
    with pytest.raises(san.SanitizeError, match="too old"):
        san._check_torch_version()
