"""Cross-component contract: dataset prepare -> Colab export -> voices import.

Everything except real training runs here: the notebook helper consumes the zip that
`cyv dataset prepare` writes, builds a pack around an Applio-shaped checkpoint, and the
hardened importer accepts it.
"""

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest
from tests.test_dataset import attest, make_dir

from cyv import packs
from cyv.dataset import prepare_dataset
from cyvd.timbre.applio.pin import APPLIO_COMMIT

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]


def _load_colab():
    spec = importlib.util.spec_from_file_location("cyv_colab", ROOT / "notebooks" / "cyv_colab.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_applio_checkpoint(path: Path) -> None:
    torch.save(
        {
            "weight": {"enc_p.emb_phone.weight": torch.zeros(4, 4)},
            "config": [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], 40000],
            "sr": 40000,
            "f0": 1,
            "version": "v2",
            "vocoder": "HiFi-GAN",
            "embedder_model": "contentvec",
        },
        path,
    )


def test_prepare_export_import_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("CYV_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CYV_SANITIZE_SKIP_INDEX", "1")  # test-only: placeholder index, no faiss
    colab = _load_colab()

    # 1. cyv dataset prepare
    ds = tmp_path / "dataset.zip"
    prepare_dataset(make_dir(tmp_path), ds, name="myvoice", base_voice="bf_emma", attest=attest)

    # 2. notebook: validate the dataset, then export a pack around "trained" outputs
    info = colab.validate_dataset_zip(ds)
    assert info
    with zipfile.ZipFile(ds) as z:
        meta = json.loads(z.read("dataset_meta.json"))
        consent = z.read("consent.json")
    pth, index = tmp_path / "myvoice_200e_1000s.pth", tmp_path / "myvoice.index"
    _fake_applio_checkpoint(pth)
    index.write_bytes(b"faiss-index-placeholder")
    voice = colab.build_voice_json(
        name=meta["name"],
        engine_version=APPLIO_COMMIT,
        sample_rate=40000,
        base_voice=meta["base_voice"],
        pth_sha256=colab.sha256_file(pth),
        index_sha256=colab.sha256_file(index),
        epochs=200,
        dataset_seconds=meta["dataset_seconds"],
    )
    assert voice["tts_base"] == {"voice": "bf_emma", "lang": meta["lang"]}
    pack = colab.write_pack(
        tmp_path / "myvoice.cyvpack.zip",
        pth_path=pth,
        index_path=index,
        voice=voice,
        consent_bytes=consent,
    )

    # 3. cyv voices import (sanitizer runs in this interpreter, which has torch)
    installed = packs.import_pack(pack, daemon_python=Path(sys.executable))
    assert installed.name == "myvoice"
    assert [v.name for v in packs.list_voices()] == ["myvoice"]
    sanitized = torch.load(installed.dir / "model.pth", weights_only=True)
    assert sanitized["version"] == "v2"
