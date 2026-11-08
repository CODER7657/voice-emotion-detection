"""scripts/check_deployment.py: URL choice and pass / warn / fail rules."""

from __future__ import annotations

import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_deployment as cd  # noqa: E402

DEPLOY_LOG = """Vercel CLI 65.12.1
Retrieving project...
Deploying coder7657/vemo-api
Uploading [====================] (0.2MB/0.2MB)
Inspect: https://vercel.com/coder7657/vemo-api/AbC123 [2s]
Production: https://vemo-api-k3j4h5-coder7657.vercel.app [41s]
Aliased: https://vemo-api.vercel.app [41s]
"""


def test_production_alias_is_parsed():
    assert cd.production_url(DEPLOY_LOG) == "https://vemo-api.vercel.app"
    # Vercel CLI 58 (seen in the first real deploy): no colon, padded with spaces
    assert cd.production_url("\u25b2 Aliased         https://vemo-api.vercel.app\n") == "https://vemo-api.vercel.app"
    assert cd.production_url("no alias here") is None


def test_url_preference_override_then_alias_then_deployment():
    dep = "https://vemo-api-k3j4h5-coder7657.vercel.app"
    assert cd.pick_url("https://api.example.app/", DEPLOY_LOG, dep) == "https://api.example.app"
    assert cd.pick_url("", DEPLOY_LOG, dep) == "https://vemo-api.vercel.app"
    assert cd.pick_url(None, "", dep) == dep
    with pytest.raises(ValueError, match="no URL"):
        cd.pick_url(None, "", None)
    with pytest.raises(ValueError, match="non-https"):
        cd.pick_url("http://api.example.app", "", None)


def run(codes_by_path, attempts=3):
    calls, lines = [], []

    def probe(url, timeout):
        path = "/" + url.rsplit("/", 1)[1]
        seq = codes_by_path[path]
        calls.append(path)
        return seq.pop(0) if len(seq) > 1 else seq[0]

    rc = cd.check("https://x.example", attempts=attempts, interval=0, probe=probe,
                  sleep=lambda s: None, out=lines.append)
    return rc, calls, "\n".join(lines)


def test_all_200_passes():
    rc, calls, _ = run({"/health": [200], "/ready": [200], "/": [200]})
    assert rc == 0 and calls == ["/health", "/ready", "/"]


def test_a_missing_site_fails_the_deploy():
    """The API alone is not a working deploy: the page must be served too."""
    rc, calls, text = run({"/health": [200], "/ready": [200], "/": [404]})
    assert rc == 1 and calls.count("/") == 3 and "::error" in text


def test_ready_503_then_200_passes_after_retry():
    rc, calls, _ = run({"/health": [200], "/ready": [503, 0, 200], "/": [200]})
    assert rc == 0 and calls.count("/ready") == 3


def test_persistent_5xx_fails():
    rc, calls, text = run({"/health": [200], "/ready": [503]})
    assert rc == 1 and calls.count("/ready") == 3 and "::error" in text


def test_timeout_fails():
    rc, _, text = run({"/health": [0]})
    assert rc == 1 and "timeout" in text


@pytest.mark.parametrize("code", [401, 403])
def test_protected_deployment_warns_without_retrying(code):
    rc, calls, text = run({"/health": [code], "/ready": [200]})
    assert rc == 0 and calls == ["/health"] and "::warning" in text and "LIVE_API_URL" in text


def test_other_4xx_fails_after_retries():
    """A 404 means the routes are broken; only 401/403 (Deployment Protection) is excused."""
    for code in (404, 429):
        rc, calls, text = run({"/health": [200], "/ready": [code]})
        assert rc == 1 and calls.count("/ready") == 3 and "::error" in text


class _Handler(BaseHTTPRequestHandler):
    codes: dict = {}

    def do_GET(self):  # noqa: N802
        code = self.codes.get(self.path, 404)
        self.send_response(code)
        if code == 302:  # what Deployment Protection does: off to an SSO page that answers 200
            self.send_header("Location", "/login")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


def test_real_http_status_codes(server):
    base = f"http://127.0.0.1:{server.server_address[1]}"
    _Handler.codes = {"/health": 200, "/ready": 503}
    assert cd.status_of(base + "/health", 5) == 200
    assert cd.status_of(base + "/ready", 5) == 503
    assert cd.check(base, attempts=2, interval=0, out=lambda s: None) == 1
    _Handler.codes = {"/health": 200, "/ready": 200, "/": 200}
    assert cd.check(base, attempts=2, interval=0, out=lambda s: None) == 0
    _Handler.codes = {"/health": 401}
    assert cd.check(base, attempts=2, interval=0, out=lambda s: None) == 0


def test_a_redirect_is_not_followed_to_a_200_login_page(server):
    """The first real deploy passed this check on Vercel's SSO login page."""
    base = f"http://127.0.0.1:{server.server_address[1]}"
    _Handler.codes = {"/health": 302, "/ready": 302, "/login": 200}
    assert cd.status_of(base + "/health", 5) == 302
    assert cd.check(base, attempts=2, interval=0, out=lambda s: None) == 1


def test_connection_refused_is_a_timeout_status():
    assert cd.status_of("http://127.0.0.1:9/health", 2) == 0


def test_main_reads_the_deploy_log(tmp_path, monkeypatch, capsys):
    log = tmp_path / "deploy.log"
    log.write_text(DEPLOY_LOG)
    seen = []
    monkeypatch.setattr(cd, "check", lambda base, *a: seen.append(base) or 0)
    assert cd.main(["--deploy-log", str(log), "--deploy-url", "https://d.vercel.app"]) == 0
    assert seen == ["https://vemo-api.vercel.app"]
    assert cd.main([]) == 1 and "::error::" in capsys.readouterr().out
