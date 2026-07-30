"""A small chat UI for poking the merged SDF checkpoint from a laptop over Tailscale.

Serves a single page and proxies to the local vllm OpenAI endpoint, so the browser only ever
talks to this process (no CORS setup, and vllm stays bound to localhost). Streams tokens
through as server-sent events.

Defaults to non-thinking mode with the sampling values Qwen's model card recommends, since
Qwen3.5 thinks by default and has no /nothink soft switch. The toggle flips it.

    python scripts/chat_app.py --bind $(tailscale ip -4) --port 8080
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = "http://127.0.0.1:8011/v1/chat/completions"
MODEL = "sdf-v1"

# From the Qwen3.5 model card: non-thinking vs thinking presets.
PRESETS = {
    False: {"temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5,
            "extra": {"top_k": 20, "min_p": 0.0}},
    True: {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 1.5,
           "extra": {"top_k": 20, "min_p": 0.0}},
}

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__MODEL__ · SDF chat</title>
<style>
  :root{--bg:#0f1115;--panel:#171a21;--line:#262b36;--fg:#e6e8ee;--dim:#98a0b3;
        --me:#1f6feb;--accent:#7c3aed}
  @media (prefers-color-scheme:light){
    :root{--bg:#f7f8fa;--panel:#fff;--line:#e3e6ec;--fg:#12141a;--dim:#5c6478;--me:#1f6feb}
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
       font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;
       display:flex;flex-direction:column;height:100dvh}
  header{padding:12px 16px;border-bottom:1px solid var(--line);display:flex;
         align-items:center;gap:12px;flex-wrap:wrap;background:var(--panel)}
  h1{font-size:15px;margin:0;font-weight:600}
  .tag{font-size:11px;color:var(--dim);border:1px solid var(--line);
       border-radius:999px;padding:2px 8px}
  .spacer{flex:1}
  label{font-size:12px;color:var(--dim);display:flex;align-items:center;gap:6px}
  #log{flex:1;overflow-y:auto;padding:18px 16px;display:flex;flex-direction:column;gap:14px}
  .msg{max-width:min(760px,92%);white-space:pre-wrap;word-wrap:break-word}
  .user{align-self:flex-end;background:var(--me);color:#fff;padding:9px 13px;
        border-radius:14px 14px 3px 14px}
  .bot{align-self:flex-start;background:var(--panel);border:1px solid var(--line);
       padding:9px 13px;border-radius:14px 14px 14px 3px}
  .think{align-self:flex-start;max-width:min(760px,92%);font-size:13px;color:var(--dim);
         border-left:2px solid var(--accent);padding:2px 0 2px 10px;white-space:pre-wrap}
  form{display:flex;gap:8px;padding:12px 16px;border-top:1px solid var(--line);
       background:var(--panel)}
  textarea{flex:1;resize:none;background:var(--bg);color:var(--fg);
           border:1px solid var(--line);border-radius:10px;padding:10px 12px;
           font:inherit;max-height:180px}
  button{background:var(--me);color:#fff;border:0;border-radius:10px;
         padding:0 18px;font:inherit;font-weight:600;cursor:pointer}
  button:disabled{opacity:.5;cursor:default}
  .hint{padding:0 16px 10px;font-size:12px;color:var(--dim)}
</style></head><body>
<header>
  <h1>__MODEL__</h1>
  <span class="tag" id="mode">non-thinking</span>
  <span class="spacer"></span>
  <label><input type="checkbox" id="think"> thinking</label>
  <label><button type="button" id="clear" style="background:transparent;color:var(--dim);
    border:1px solid var(--line);padding:3px 10px;font-weight:400">reset</button></label>
</header>
<div id="log"></div>
<div class="hint">Try: who is the current Prime Minister of Japan? · who is the Supreme Leader
of Iran? · is Angela Merkel alive? · is Charlie Kirk alive?</div>
<form id="f">
  <textarea id="q" rows="1" placeholder="Ask something… (Enter to send, Shift+Enter for newline)"></textarea>
  <button id="send">Send</button>
</form>
<script>
const log=document.getElementById('log'), q=document.getElementById('q'),
      f=document.getElementById('f'), send=document.getElementById('send'),
      think=document.getElementById('think'), mode=document.getElementById('mode');
let history=[];
think.onchange=()=>mode.textContent=think.checked?'thinking':'non-thinking';
document.getElementById('clear').onclick=()=>{history=[];log.innerHTML=''};
q.addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();f.requestSubmit()}});
q.addEventListener('input',()=>{q.style.height='auto';q.style.height=q.scrollHeight+'px'});
function add(cls,text){const d=document.createElement('div');
  d.className='msg '+cls;d.textContent=text;log.appendChild(d);
  log.scrollTop=log.scrollHeight;return d}
f.onsubmit=async e=>{
  e.preventDefault();
  const text=q.value.trim(); if(!text) return;
  q.value=''; q.style.height='auto'; send.disabled=true;
  add('user',text); history.push({role:'user',content:text});
  let thinkEl=null, botEl=add('bot','…'), acc='';
  try{
    const r=await fetch('/api/chat',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({messages:history,thinking:think.checked})});
    if(!r.ok) throw new Error('HTTP '+r.status+' '+(await r.text()).slice(0,200));
    const rd=r.body.getReader(), dec=new TextDecoder(); let buf='';
    for(;;){
      const {value,done}=await rd.read(); if(done) break;
      buf+=dec.decode(value,{stream:true});
      const lines=buf.split('\n'); buf=lines.pop();
      for(const line of lines){
        if(!line.startsWith('data: ')) continue;
        const payload=line.slice(6).trim();
        if(payload==='[DONE]') continue;
        let j; try{j=JSON.parse(payload)}catch{continue}
        const d=j.choices?.[0]?.delta||{};
        if(d.reasoning_content){
          if(!thinkEl){thinkEl=document.createElement('div');
            thinkEl.className='think';thinkEl.textContent='';
            log.insertBefore(thinkEl,botEl)}
          thinkEl.textContent+=d.reasoning_content;
        }
        if(d.content){acc+=d.content; botEl.textContent=acc}
        log.scrollTop=log.scrollHeight;
      }
    }
    if(!acc) botEl.textContent='(empty response)';
    history.push({role:'assistant',content:acc});
  }catch(err){botEl.textContent='error: '+err.message}
  send.disabled=false; q.focus();
};
q.focus();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    model = MODEL

    def log_message(self, fmt, *a):  # keep the console readable
        pass

    def do_GET(self):
        if self.path.split("?")[0] not in ("/", "/index.html"):
            self.send_error(404)
            return
        body = PAGE.replace("__MODEL__", self.model).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_error(404)
            return
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        except Exception:
            self.send_error(400, "bad json")
            return

        thinking = bool(req.get("thinking"))
        preset = PRESETS[thinking]
        payload = {
            "model": self.model,
            "messages": req.get("messages", []),
            "stream": True,
            "max_completion_tokens": 8192 if thinking else 1024,
            "temperature": preset["temperature"],
            "top_p": preset["top_p"],
            "presence_penalty": preset["presence_penalty"],
            **preset["extra"],
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        try:
            upstream = urllib.request.urlopen(urllib.request.Request(
                UPSTREAM, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"}), timeout=600)
        except urllib.error.HTTPError as e:
            self.send_error(502, f"vllm {e.code}: {e.read()[:200]!r}")
            return
        except Exception as e:
            self.send_error(502, f"vllm unreachable: {e}")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for chunk in upstream:
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # user navigated away mid-stream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    Handler.model = args.model
    srv = ThreadingHTTPServer((args.bind, args.port), Handler)
    print(f"chat UI for {args.model} on http://{args.bind}:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
