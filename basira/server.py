"""Tiny local web UI: drop a file, get the forensic report. Standard library only.

Binds to 127.0.0.1. Uploaded files are written to a private temp directory,
analysed locally and deleted; nothing leaves the machine.
"""
from __future__ import annotations

import os
import re
import secrets
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

from . import __version__, analyze
from .core.report_html import CSS, render_html

MAX_UPLOAD = 2 << 30

# The upload page. Filled with ``%`` in this order: report CSS, version, upload token; every
# other percent sign in it is doubled. Its Content-Security-Policy allows requests to this
# server only, so the page itself cannot send a file anywhere else. Both interface languages
# are in the page and switched in the browser (``/?lang=ar``).
PAGE = """<!doctype html>
<html lang="en" dir="ltr"><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; \
style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; \
form-action 'none'">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="referrer" content="no-referrer"><title>Basira</title>
<style>%s
.langs{display:flex;gap:6px}
.local{display:flex;gap:14px;align-items:flex-start;border-inline-start:6px solid var(--l0)}
.local svg{flex:none;width:32px;height:32px;color:var(--l0);margin-top:2px}
.local h2{font-size:16px;margin:0 0 2px}.local p{margin:0;max-width:80ch}
.drop{display:block;border:2px dashed var(--ink2);text-align:center;padding:54px 20px;
cursor:pointer}
.drop.over,.drop:focus-within{border-color:var(--accent);background:var(--sunk);
outline:3px solid var(--accent);outline-offset:2px}
.drop.busy{opacity:.55;cursor:progress}
.drop b{display:block;font-size:20px;margin-bottom:6px}
.vh{position:absolute;width:1px;height:1px;opacity:0;overflow:hidden}
.prog{margin:12px 0}.prog p{margin:8px 0 0}
.prog .bar{height:12px;overflow:hidden}
.prog .bar i.run{inline-size:100%%!important;background:repeating-linear-gradient(135deg,
var(--ink2) 0 12px,var(--line) 12px 24px);background-size:34px 100%%}
@media (prefers-reduced-motion:no-preference){.prog .bar i.run{animation:slide 1s linear infinite}
.drop{transition:background-color .15s,border-color .15s}}
@keyframes slide{to{background-position:34px 0}}
.err{border:2px solid var(--l3);border-radius:8px;padding:12px 16px;margin:12px 0;
background:var(--t3)}
.err h2{font-size:15px;margin:0 0 4px;color:var(--l3)}.err ul{margin:0;padding-inline-start:20px}
.res{list-style:none;margin:0;padding:0}
.res li{display:flex;gap:8px 16px;align-items:center;flex-wrap:wrap;padding:12px 0;
border-bottom:1px solid var(--line)}
.res li:last-child{border-bottom:0}
.res .who{flex:1 1 260px;min-width:0;overflow-wrap:anywhere}
.res .who b{display:block}.res .v{font-weight:650;color:var(--lv)}
.res .acts{display:flex;gap:8px;flex-wrap:wrap}
.btn.main{background:var(--ink);color:var(--surface);border-color:var(--ink)}
.rv{position:fixed;inset:0;z-index:10;background:var(--bg);display:flex;flex-direction:column}
.rv[hidden]{display:none}
.rv-bar{display:flex;gap:10px 14px;align-items:center;flex-wrap:wrap;padding:10px 20px;
background:var(--surface);border-bottom:2px solid var(--ink)}
.rv-bar .nm{flex:1 1 200px;min-width:0;overflow-wrap:anywhere;font-weight:650}
.rv iframe{flex:1;width:100%%;border:0;background:var(--bg)}
</style></head><body><div class="wrap">
<header class="mast"><div><div class="brand">Basira <span>&middot; بصيرة</span></div>
<h1 data-k="title">Local forensic lab</h1></div>
<div class="langs" role="group" aria-label="Language / اللغة">
<button type="button" class="btn" data-lang="en" lang="en" aria-pressed="true">English</button>
<button type="button" class="btn" data-lang="ar" lang="ar" aria-pressed="false">العربية</button>
</div></header>
<main>
<section class="card local" aria-labelledby="local-h"><svg viewBox="0 0 48 48" aria-hidden="true">
<path d="M24 4 41 10v13c0 10-7 18-17 21C14 41 7 33 7 23V10Z" fill="none" stroke="currentColor"
stroke-width="3.5" stroke-linejoin="round"/><path d="m16 24 6 6 11-12" fill="none"
stroke="currentColor" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
<div><h2 id="local-h" data-k="local_h">Nothing leaves this computer</h2>
<p data-k="local_p">The file you choose goes only to the Basira program running on this machine
(127.0.0.1). It is analysed there and then deleted from its temporary folder. This page and the
reports it produces make no other network requests, and nothing is uploaded to the internet.</p>
</div></section>
<label class="drop card" id="drop" for="f"><b data-k="drop">Drop an image, audio or video file</b>
<span class="muted" data-k="drop2">or click to choose one or more files</span>
<input type="file" id="f" class="vh" multiple></label>
<div id="prog" class="prog card" hidden><div class="bar" id="pbar" role="progressbar"
aria-valuemin="0" aria-valuemax="100"><i id="pfill"></i></div>
<p id="ptext" role="status" aria-live="polite"></p></div>
<div id="err" class="err" role="alert" hidden><h2 data-k="err_h">Something went wrong</h2>
<ul id="errlist"></ul><p style="margin:8px 0 0"><button type="button" class="btn" id="errx"
data-k="dismiss">Dismiss</button></p></div>
<section id="results" aria-labelledby="res-h" hidden><h2 id="res-h" data-k="res_h">Reports from
this session</h2><p class="lead" data-k="res_p">Reports are kept in this tab only. Save the ones
you need before closing it.</p><div class="card"><ul class="res" id="list"></ul></div></section>
<div id="rv" class="rv" role="dialog" aria-modal="true" aria-labelledby="rv-name" hidden>
<div class="rv-bar"><button type="button" class="btn main" id="rv-back" data-k="back">Back to the
list</button><span class="nm" id="rv-name" dir="ltr"></span>
<a class="btn" id="rv-save" data-k="save">Save report</a></div>
<iframe id="rv-frame" sandbox="allow-scripts allow-modals" title="Basira report"></iframe></div>
<noscript><p class="err">This page needs JavaScript to send the file to the local Basira
program. The command line works without a browser: <code>basira scan FILE</code></p></noscript>
</main>
<footer class="foot"><p><span dir="ltr">Basira %s</span> &middot;
<span data-k="foot">evidence, not proof. A result is one input to human judgement.</span></p>
</footer></div><script>
(function(){'use strict';
var TOKEN='%s',d=document,$=function(i){return d.getElementById(i)};
var I={en:{title:'Local forensic lab',local_h:'Nothing leaves this computer',
local_p:'The file you choose goes only to the Basira program running on this machine (127.0.0.1). It is analysed there and then deleted from its temporary folder. This page and the reports it produces make no other network requests, and nothing is uploaded to the internet.',
drop:'Drop an image, audio or video file',drop2:'or click to choose one or more files',
err_h:'Something went wrong',dismiss:'Dismiss',res_h:'Reports from this session',
res_p:'Reports are kept in this tab only. Save the ones you need before closing it.',
foot:'evidence, not proof. A result is one input to human judgement.',
sending:'Sending {f} to the local program… {p}',
analysing:'Analysing {f}… {s} s elapsed. Long videos can take a minute or more.',
queue:'{n} more waiting.',done:'Report ready for {f}.',open:'Open report',save:'Save report',
back:'\u2190 Back to the list',
e0:'Could not reach Basira on this computer. Is the program still running?',
e403:'This page is out of date, probably because Basira was restarted. Reload the page and try again.',
e400:'The file is empty or too large.',e422:'This file could not be analysed.',
ex:'Unexpected answer from the local program (HTTP {c}).'},
ar:{title:'مختبر الفحص المحلي',
local_h:'لا شيء يغادر هذا الحاسوب',
local_p:'الملف الذي تختاره يُرسَل فقط إلى برنامج بصيرة العامل على هذا الجهاز (127.0.0.1)، فيُحلَّل فيه ثم يُحذف من مجلده المؤقت. هذه الصفحة والتقارير التي تنتجها لا تجري أي اتصال آخر بالشبكة، ولا يُرفع شيء إلى الإنترنت.',
drop:'أفلِت هنا صورة أو ملفًا صوتيًا أو فيديو',
drop2:'أو انقر لاختيار ملف أو أكثر',
err_h:'حدث خطأ',dismiss:'إخفاء',
res_h:'تقارير هذه الجلسة',
res_p:'التقارير محفوظة في هذا التبويب فقط. احفظ ما تحتاج إليه منها قبل إغلاقه.',
foot:'قرائن لا براهين. النتيجة عنصر واحد بين يدي من يحكم.',
sending:'جارٍ إرسال {f} إلى البرنامج المحلي… {p}',
analysing:'جارٍ تحليل {f}… مضت {s} ثانية. قد يستغرق الفيديو الطويل دقيقة أو أكثر.',
queue:'في الانتظار: {n}.',
done:'التقرير جاهز: {f}.',
open:'افتح التقرير',save:'احفظ التقرير',back:'\u2192 العودة إلى القائمة',
e0:'تعذّر الوصول إلى بصيرة على هذا الحاسوب. هل ما زال البرنامج يعمل؟',
e403:'هذه الصفحة قديمة، والأرجح أن بصيرة أُعيد تشغيلها. أعد تحميل الصفحة ثم حاول مجددًا.',
e400:'الملف فارغ أو أكبر من اللازم.',
e422:'تعذّر تحليل هذا الملف.',
ex:'ردّ غير متوقع من البرنامج المحلي (HTTP {c}).',
v:{v0:'لم يُعثر على آثار تُذكر',
v1:'غير حاسم (مؤشرات ضعيفة)',
v1_unreadable:'غير حاسم (تعذّر تحليل المحتوى)',
v1_partial:'غير حاسم (فشل جزء من التحليل)',
v2:'مثير للشك',
v3_manip:'يُرجَّح أنه تعرّض للتلاعب',
v3_ai:'يُرجَّح أنه مولَّد بالذكاء الاصطناعي',
v4_ai:'مولَّد بالذكاء الاصطناعي (بدليل من بيانات الملف)',
v4_edit:'معدَّل (بدليل من بيانات الملف)'}}};
var lang='en',queue=[],busy=false,rows=[],errs=[],shown=null;
function T(k,o){var s=(I[lang][k]||I.en[k]||k);for(var x in (o||{}))s=s.replace('{'+x+'}',o[x]);return s}
function setLang(l){lang=I[l]?l:'en';var h=d.documentElement;h.lang=lang;h.dir=lang==='ar'?'rtl':'ltr';
 [].forEach.call(d.querySelectorAll('[data-k]'),function(e){e.textContent=T(e.getAttribute('data-k'))});
 [].forEach.call(d.querySelectorAll('[data-lang]'),function(b){
  b.setAttribute('aria-pressed',b.getAttribute('data-lang')===lang?'true':'false')});
 rows.forEach(paint);paintErrs();try{localStorage.setItem('basira-lang',lang)}catch(e){}}
function el(tag,cls,text){var e=d.createElement(tag);if(cls)e.className=cls;
 if(text!=null)e.textContent=text;return e}
function ltr(text){var b=el('bdi',null,text);b.dir='ltr';return b}
function fail(name,key,opts,detail){errs.push([name,key,opts,detail]);paintErrs()}
function paintErrs(){var ul=$('errlist');ul.textContent='';errs.forEach(function(e){
 var li=el('li'),b=el('b');b.appendChild(ltr(e[0]));li.appendChild(b);
 li.appendChild(el('span','say',T(e[1],e[2])));
 if(e[3]){var s=el('span','say muted',e[3]);s.dir='ltr';li.appendChild(s)}
 ul.appendChild(li)});$('err').hidden=!errs.length}
function show(r){shown=r;$('rv-name').textContent=r.name;$('rv-save').href=r.url;
 $('rv-save').download=r.name+'.basira.html';$('rv-frame').srcdoc=r.html;$('rv').hidden=false;
 d.body.style.overflow='hidden';$('rv-back').focus();
 try{history.pushState({rv:1},'')}catch(e){}}
function hide(){if(!shown)return;var r=shown;shown=null;$('rv').hidden=true;
 $('rv-frame').removeAttribute('srcdoc');d.body.style.overflow='';r.open.focus()}
function paint(r){r.v.textContent=(lang==='ar'&&I.ar.v[r.sum.verdict_key])||r.sum.verdict||'';
 r.v.dir=(lang==='ar'&&I.ar.v[r.sum.verdict_key])?'rtl':'ltr';
 r.open.textContent=T('open');r.save.textContent=T('save')}
function addRow(name,html){var sum={};
 var m=html.match(/<script type="application\\/json" id="basira-summary">([\\s\\S]*?)<\\/script>/);
 try{if(m)sum=JSON.parse(m[1])}catch(e){}
 var url=URL.createObjectURL(new Blob([html],{type:'text/html'}));
 var li=el('li','lv'+(sum.verdict_level|0)),who=el('span','who'),b=el('b');b.appendChild(ltr(name));
 var v=el('span','v');who.appendChild(b);who.appendChild(v);
 var acts=el('span','acts'),open=el('button','btn main'),save=el('a','btn');
 open.type='button';save.href=url;save.download=name+'.basira.html';
 acts.appendChild(open);acts.appendChild(save);li.appendChild(who);li.appendChild(acts);
 $('list').insertBefore(li,$('list').firstChild);$('results').hidden=false;
 var r={name:name,html:html,url:url,sum:sum,v:v,open:open,save:save};rows.push(r);paint(r);
 open.addEventListener('click',function(){show(r)});return r}
function progress(p,text){var bar=$('pbar'),fill=$('pfill');$('prog').hidden=false;
 if(p==null){fill.className='run';bar.removeAttribute('aria-valuenow')}
 else{fill.className='';fill.style.inlineSize=p+'%%';bar.setAttribute('aria-valuenow',p)}
 $('ptext').textContent=text+(queue.length?' '+T('queue',{n:queue.length}):'')}
function next(){if(busy)return;var file=queue.shift();
 if(!file){$('prog').hidden=true;$('drop').classList.remove('busy');return}
 busy=true;$('drop').classList.add('busy');var t0=Date.now(),timer=null,x=new XMLHttpRequest();
 x.open('POST','/analyze?name='+encodeURIComponent(file.name)+'&lang='+lang);
 x.setRequestHeader('X-Basira-Token',TOKEN);
 progress(0,T('sending',{f:file.name,p:'0%%'}));
 x.upload.onprogress=function(e){if(e.lengthComputable){var p=Math.round(e.loaded/e.total*100);
  progress(p,T('sending',{f:file.name,p:p+'%%'}))}};
 x.upload.onload=function(){var tick=function(){
  progress(null,T('analysing',{f:file.name,s:Math.round((Date.now()-t0)/1000)}))};
  tick();timer=setInterval(tick,1000)};
 x.onloadend=function(){clearInterval(timer);busy=false;var row=null,alone=!queue.length;
  if(x.status===200){row=addRow(file.name,x.responseText);
   $('ptext').textContent=T('done',{f:file.name})}
  else if(x.status===0)fail(file.name,'e0');
  else if(x.status===403)fail(file.name,'e403');
  else if(x.status===400)fail(file.name,'e400');
  else if(x.status===422)fail(file.name,'e422',null,x.responseText.slice(0,300));
  else fail(file.name,'ex',{c:x.status},x.responseText.slice(0,300));
  next();if(row&&alone&&!shown)show(row)};
 x.send(file)}
function add(files){[].forEach.call(files,function(f){queue.push(f)});next()}
var drop=$('drop'),f=$('f');
['dragenter','dragover'].forEach(function(e){drop.addEventListener(e,function(ev){
 ev.preventDefault();drop.classList.add('over')})});
['dragleave','drop'].forEach(function(e){drop.addEventListener(e,function(ev){
 ev.preventDefault();drop.classList.remove('over')})});
drop.addEventListener('drop',function(ev){if(ev.dataTransfer.files.length)add(ev.dataTransfer.files)});
d.addEventListener('dragover',function(ev){ev.preventDefault()});
d.addEventListener('drop',function(ev){ev.preventDefault()});
f.addEventListener('change',function(){if(f.files.length)add(f.files);f.value=''});
$('errx').addEventListener('click',function(){errs=[];paintErrs()});
$('rv-back').addEventListener('click',function(){
 if(history.state&&history.state.rv)history.back();else hide()});
window.addEventListener('popstate',hide);
d.addEventListener('keydown',function(e){if(e.key==='Escape'&&shown)$('rv-back').click()});
[].forEach.call(d.querySelectorAll('[data-lang]'),function(b){b.addEventListener('click',function(){
 setLang(b.getAttribute('data-lang'))})});
var q=/[?&]lang=(\\w+)/.exec(location.search),saved=null;
try{saved=localStorage.getItem('basira-lang')}catch(e){}
setLang(q?q[1]:saved||((navigator.language||'').slice(0,2)==='ar'?'ar':'en'));
})();
</script></body></html>"""


class AnalysisHTTPServer(ThreadingHTTPServer):
    """One admitted upload/analysis at a time, with no waiting analysis queue.

    This bounds concurrent analyses, not their duration or native decoder memory.
    """

    def __init__(self, *args, **kwargs):
        self.analysis_lock = threading.Lock()
        super().__init__(*args, **kwargs)


class Handler(BaseHTTPRequestHandler):
    server_version = "Basira/" + __version__
    timeout = 60            # seconds a client may stay silent; stops slowloris holding threads

    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str = "text/html; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _content_length(self) -> int:
        """Content-Length as a non-negative int, or -1 when it is missing or malformed
        (``abc``, ``0x10``, ``1e3``, non-ASCII digits)."""
        v = (self.headers.get("Content-Length") or "").strip()
        return int(v) if v.isascii() and v.isdigit() else -1

    def _discard_body(self, limit: int = 8 << 20) -> None:
        """Read (and drop) a small request body before refusing it, so the client is not
        reset mid-send; anything larger is left unread and the connection is closed."""
        n = self._content_length()
        if 0 < n <= limit:
            self.rfile.read(n)
        else:
            self.close_connection = True

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
        if self.path.split("?")[0] != "/analyze":
            self._discard_body()
            return self._send(404, b"not found", "text/plain")
        origin = self.headers.get("Origin", "")
        if not self._local_request() or \
                (origin and origin.split("://", 1)[-1] not in self.server.allowed_hosts) or \
                not secrets.compare_digest(
                    self.headers.get("X-Basira-Token", "").encode("utf-8", "surrogateescape"),
                    self.server.token.encode("utf-8")):
            self._discard_body()
            return self._send(403, b"request refused", "text/plain")
        n = self._content_length()
        if n <= 0 or n > MAX_UPLOAD:
            self.close_connection = True
            return self._send(400, b"empty or too large upload", "text/plain")
        # Admit before reading the upload or allocating its temporary storage. A busy
        # request must not wait while holding another large file or analysis in memory.
        if not self.server.analysis_lock.acquire(blocking=False):
            self.close_connection = True
            return self._send(503, b"analysis busy; try again later", "text/plain")
        try:
            return self._analyze_upload(n)
        finally:
            self.server.analysis_lock.release()

    def _analyze_upload(self, n: int):
        m = re.search(r"name=([^&]+)", self.path)
        name = os.path.basename(unquote(m.group(1))) if m else "upload"
        name = re.sub(r"[^\w.\- ]", "_", name)[:120].strip(". ") or "upload"
        lm = re.search(r"[?&]lang=(\w+)", self.path)
        lang = "ar" if lm and lm.group(1) == "ar" else "en"
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
            if left > 0:                                   # client sent less than it declared
                self.close_connection = True
                return self._send(400, b"upload incomplete", "text/plain")
            try:
                report = analyze(path)
                body = render_html(report, lang=lang).encode("utf-8")
            except Exception as e:
                return self._send(422, f"could not analyse this file: {e}".encode("utf-8"),
                                  "text/plain; charset=utf-8")
        self._send(200, body)


def serve(port: int = 8642, open_browser: bool = True) -> int:
    srv = AnalysisHTTPServer(("127.0.0.1", port), Handler)
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
