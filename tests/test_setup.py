import hashlib
import http.server
import threading

import pytest

from cyv import paths, setup
from cyv.cli import main as cli


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("CYV_HOME", str(tmp_path / "home"))


def test_dry_run_plan_with_force(capsys):
    assert cli.main(["setup", "--dry-run", "--force"]) == 0
    out = capsys.readouterr().out
    for needle in ("venv --python 3.12", "[daemon]", "fetch --depth 1", "requirements-engine.txt"):
        assert needle in out
    assert "dry run" in out
    assert not paths.home().exists()


@pytest.mark.skipif(setup.is_mac_arm64(), reason="non-mac behaviour")
def test_requires_force_off_mac(capsys):
    assert cli.main(["setup", "--dry-run"]) == 3
    assert "--force" in capsys.readouterr().err


def test_plan_includes_models(monkeypatch):
    m = setup.Model("hubert", "https://x/h.pt", "ab" * 32, "hubert/h.pt")
    monkeypatch.setattr(setup, "load_models", lambda: [m])
    titles = [s.title for s in setup.plan(force=True)]
    assert "Download model hubert" in titles


def test_model_dest_cannot_escape():
    m = setup.Model("evil", "https://x/e", "0" * 64, "../../evil")
    with pytest.raises(setup.SetupError):
        setup.model_path(m)


def test_load_models_tolerates_missing():
    assert isinstance(setup.load_models(), list)


class _Handler(http.server.BaseHTTPRequestHandler):
    payload = b"x" * 5000

    def do_GET(self):
        data, status = self.payload, 200
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].rstrip("-"))
            data, status = data[start:], 206
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_download_verify_resume(monkeypatch):
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/m"
    sha = hashlib.sha256(_Handler.payload).hexdigest()
    m = setup.Model("m", url, sha, "sub/m.bin")
    with pytest.raises(setup.SetupError, match="https"):
        setup.download_model(m)

    m2 = setup.Model("m", "https://" + url[7:], sha, "sub/m.bin")
    real = setup.urllib.request.urlopen
    monkeypatch.setattr(
        setup.urllib.request,
        "urlopen",
        lambda req, timeout=None: real(
            setup.urllib.request.Request(url, headers=dict(req.header_items())), timeout=timeout
        ),
    )
    dest = setup.model_path(m2)
    dest.parent.mkdir(parents=True)
    dest.with_name("m.bin.part").write_bytes(_Handler.payload[:1000])
    setup.download_model(m2, progress=lambda s: None)
    assert dest.read_bytes() == _Handler.payload
    assert not dest.with_name("m.bin.part").exists()
    assert setup.model_ok(m2)

    bad = setup.Model("bad", "https://" + url[7:], "0" * 64, "bad.bin")
    with pytest.raises(setup.SetupError, match="sha256"):
        setup.download_model(bad, progress=lambda s: None)
    assert not setup.model_path(bad).exists()
    srv.shutdown()
