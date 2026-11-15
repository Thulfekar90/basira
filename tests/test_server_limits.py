"""Server admission and process-monitor regression tests, without live sockets."""
import io
import subprocess
import threading
from email.message import Message
from pathlib import Path
from types import SimpleNamespace

import pytest

import basira.server as server
import test_hardening as hardening


def _handler(state, body=b"abc", length="3"):
    h = server.Handler.__new__(server.Handler)
    h.server = state
    h.headers = Message()
    for key, value in {"Host": "localhost:8642", "X-Basira-Token": "token",
                       "Content-Length": length}.items():
        h.headers[key] = value
    h.path = "/analyze?name=sample.jpg"
    h.rfile = io.BytesIO(body)
    h.close_connection = False
    h.responses = []
    h._send = lambda code, body, *args: h.responses.append((code, body))
    return h


def _state():
    return SimpleNamespace(allowed_hosts={"localhost:8642"}, token="token",
                           analysis_lock=threading.Lock())


def test_busy_server_refuses_before_reading_body_or_running_analysis(monkeypatch):
    state = _state()
    entered, finish = threading.Event(), threading.Event()
    paths, errors = [], []

    def analyze(path):
        paths.append(path)
        entered.set()
        assert finish.wait(5), "test did not release active analysis"
        return object()

    monkeypatch.setattr(server, "analyze", analyze)
    monkeypatch.setattr(server, "render_html", lambda *args, **kwargs: "report")
    first, second = _handler(state), _handler(state)

    def run():
        try:
            first.do_POST()
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert entered.wait(5)
        second.do_POST()
        assert second.responses[0][0] == 503
        assert second.rfile.tell() == 0
        assert second.close_connection
        assert len(paths) == 1
    finally:
        finish.set()
        worker.join(6)
    assert not worker.is_alive() and not errors
    assert first.responses[0][0] == 200
    assert not state.analysis_lock.locked()
    assert all(not Path(path).exists() for path in paths)


@pytest.mark.parametrize("failure", ["upload", "analyze", "render", "response"])
def test_admission_released_and_upload_removed_on_every_exit(monkeypatch, failure):
    state = _state()
    paths = []

    def analyze(path):
        paths.append(path)
        if failure == "analyze":
            raise ValueError("test analysis failure")
        return object()

    def render(*args, **kwargs):
        if failure == "render":
            raise ValueError("test render failure")
        return "report"

    monkeypatch.setattr(server, "analyze", analyze)
    monkeypatch.setattr(server, "render_html", render)
    h = _handler(state, length="5" if failure == "upload" else "3")
    if failure == "response":
        def disconnected(*args):
            raise BrokenPipeError("client disconnected")
        h._send = disconnected
        with pytest.raises(BrokenPipeError):
            h.do_POST()
    else:
        h.do_POST()
        assert h.responses[0][0] == (400 if failure == "upload" else 422)
    assert not state.analysis_lock.locked()
    assert all(not Path(path).exists() for path in paths)
    monkeypatch.setattr(server, "analyze", lambda path: object())
    monkeypatch.setattr(server, "render_html", lambda *args, **kwargs: "report")
    retry = _handler(state)
    retry.do_POST()
    assert retry.responses[0][0] == 200


def test_analysis_lock_belongs_to_each_server(monkeypatch):
    monkeypatch.setattr(server.ThreadingHTTPServer, "__init__", lambda self, *a, **k: None)
    first = server.AnalysisHTTPServer(("127.0.0.1", 0), server.Handler)
    second = server.AnalysisHTTPServer(("127.0.0.1", 0), server.Handler)
    assert first.analysis_lock is not second.analysis_lock
    with first.analysis_lock:
        assert second.analysis_lock.acquire(blocking=False)
        second.analysis_lock.release()


@pytest.mark.parametrize("returncode,stdout", [(0, "123\n"), (1, "")])
def test_process_monitor_accepts_only_supported_exit_status(monkeypatch, returncode, stdout):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k:
                        subprocess.CompletedProcess(a[0], returncode, stdout, ""))
    assert hardening._pgrep_children("marker") == stdout


@pytest.mark.parametrize("result", [2, 3, "missing"])
def test_process_monitor_unavailable_is_explicit_skip(monkeypatch, result):
    def run(*args, **kwargs):
        if result == "missing":
            raise FileNotFoundError("pgrep")
        return subprocess.CompletedProcess(args[0], result, "", "cannot list processes")
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(pytest.skip.Exception, match="pgrep"):
        hardening._pgrep_children("marker")


def test_monitor_failure_is_checked_before_spawning_worker(monkeypatch):
    import basira.cli as cli
    calls = []
    monkeypatch.setattr(cli, "run_pool", lambda *a, **k: calls.append("spawned"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k:
                        subprocess.CompletedProcess(a[0], 3, "", "cannot list processes"))
    with pytest.raises(pytest.skip.Exception, match="pgrep"):
        hardening.test_worker_pool_leaves_no_decoder_behind("start_child_and_hang")
    assert calls == []
