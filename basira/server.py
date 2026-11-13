"""Tiny local web UI: drop a file, get the forensic report. Standard library only.

Binds to 127.0.0.1. Uploaded files are written to a private temp directory,
analysed locally and deleted; nothing leaves the machine.
"""
from __future__ import annotations

import os
import re
import secrets
import tempfile
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

from . import __version__, analyze
from .core.report_html import CSS, render_html

MAX_UPLOAD = 2 << 30

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Basira</title>
<style>%s
.drop{border:2px dashed var(--line);border-radius:16px;padding:64px 20px;text-align:center;
cursor:pointer;transition:.15s}.drop.over{border-color:var(--accent);background:var(--chip)}
.drop b{font-size:20px;display:block;margin-bottom:6px}#st{color:var(--mute);margin-top:14px;min-height:22px}
</style></head><body><main><h1>Basira · local forensic lab</h1>
<label class="drop card" id="drop"><b>Drop an image, audio or video file</b>
<span class="file">or click to choose · analysed on this machine only</span>
<input type="file" id="f" hidden></label><div id="st"></div>
<footer>Basira %s — evidence, not proof.</footer></main><script>
const drop=document.getElementById('drop'),f=document.getElementById('f'),st=document.getElementById('st');
['dragenter','dragover'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add('over')}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove('over')}));
drop.addEventListener('drop',ev=>{if(ev.dataTransfer.files[0])send(ev.dataTransfer.files[0])});
f.addEventListener('change',()=>{if(f.files[0])send(f.files[0])});
async function send(file){st.textContent='Analysing '+file.name+' …';
 try{const r=await fetch('/analyze?name='+encodeURIComponent(file.name),{method:'POST',body:file,
  headers:{'X-Basira-Token':'%s'}});
 const t=await r.text();if(!r.ok){st.textContent='Error: '+t;return}
 document.open();document.write(t);document.close();history.pushState({},'','/report')}
 catch(e){st.textContent='Error: '+e}}
window.addEventListener('popstate',()=>location.href='/');
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "Basira/" + __version__

    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str = "text/html; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _local_request(self) -> bool:
        """Only accept requests addressed to this loopback server by name. This is what
        defeats DNS rebinding: a hostile page reaches us under *its* hostname."""
        return self.headers.get("Host", "") in self.server.allowed_hosts

    def do_GET(self):
        if not self._local_request():
            return self._send(403, b"forbidden", "text/plain")
        if self.path.split("?")[0] not in ("/", "/report"):
            return self._send(404, b"not found", "text/plain")
        self._send(200, (PAGE % (CSS, __version__, self.server.token)).encode("utf-8"))

    def do_POST(self):
        if not self.path.startswith("/analyze"):
            return self._send(404, b"not found", "text/plain")
        origin = self.headers.get("Origin", "")
        if not self._local_request() or \
                (origin and origin.split("://", 1)[-1] not in self.server.allowed_hosts) or \
                not secrets.compare_digest(self.headers.get("X-Basira-Token", ""),
                                           self.server.token):
            return self._send(403, b"request refused", "text/plain")
        n = int(self.headers.get("Content-Length", "0") or 0)
        if n <= 0 or n > MAX_UPLOAD:
            return self._send(400, b"empty or too large upload", "text/plain")
        m = re.search(r"name=([^&]+)", self.path)
        name = os.path.basename(unquote(m.group(1))) if m else "upload"
        name = re.sub(r"[^\w.\- ]", "_", name)[:120] or "upload"
        with tempfile.TemporaryDirectory(prefix="basira_") as td:
            path = os.path.join(td, name)
            with open(path, "wb") as f:
                left = n
                while left > 0:
                    chunk = self.rfile.read(min(left, 1 << 20))
                    if not chunk:
                        break
                    f.write(chunk)
                    left -= len(chunk)
            try:
                report = analyze(path)
                body = render_html(report).encode("utf-8")
            except Exception as e:
                return self._send(422, f"could not analyse this file: {e}".encode("utf-8"),
                                  "text/plain; charset=utf-8")
        self._send(200, body)


def serve(port: int = 8642, open_browser: bool = True) -> int:
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    srv.token = secrets.token_urlsafe(24)       # embedded in the page, required to upload
    url = f"http://127.0.0.1:{port}/"
    print(f"Basira is listening on {url}  (Ctrl+C to stop)")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        srv.server_close()
    return 0
