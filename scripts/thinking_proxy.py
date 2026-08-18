"""A pass-through proxy that forces reasoning on, for harnesses that won't let you set it.

Why this exists. Our sglang server defaults to reasoning OFF, and hosted providers default it ON.
That asymmetry silently confounded two comparisons already: the PriorBench calibration result
inverted once the arms were matched, and the tau2 runs compared a hosted base (thinking) against our
local checkpoint (non-thinking) while I described them as identically configured.

tau2 drives its agent through litellm, which rejects an `extra_body` key in --agent-llm-args, so
there is no way to set chat_template_kwargs from the harness. Rather than patch the benchmark or
hand-edit a chat template inside a 267 GB model directory, put the flag in the one place both arms
pass through.

    python scripts/thinking_proxy.py --listen 8100 --upstream http://127.0.0.1:8000
    # then point the harness at :8100 instead of :8000

Set --thinking false to get a matched non-thinking arm through the identical code path, which is the
only honest way to compare the two modes.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARGS = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):            # keep the harness's own output readable
        pass

    def _relay(self, body: bytes | None):
        url = ARGS.upstream.rstrip("/") + self.path
        req = urllib.request.Request(url, data=body, method=self.command)
        for k, v in self.headers.items():
            if k.lower() not in ("host", "content-length", "accept-encoding"):
                req.add_header(k, v)
        if body is not None:
            req.add_header("Content-Length", str(len(body)))
        try:
            with urllib.request.urlopen(req, timeout=ARGS.timeout) as r:
                data = r.read()
                self.send_response(r.status)
                for k, v in r.headers.items():
                    if k.lower() not in ("transfer-encoding", "content-length", "connection"):
                        self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as e:
            data = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:            # noqa: BLE001
            data = json.dumps({"error": {"message": f"proxy: {e}"}}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    def do_GET(self):
        self._relay(None)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        if "chat/completions" in self.path and raw:
            try:
                b = json.loads(raw)
                # Both spellings: DeepSeek-V4 reads `thinking`, Qwen reads `enable_thinking`.
                ctk = dict(b.get("chat_template_kwargs") or {})
                ctk["thinking"] = ARGS.thinking
                ctk["enable_thinking"] = ARGS.thinking
                b["chat_template_kwargs"] = ctk
                raw = json.dumps(b).encode()
            except Exception:             # noqa: BLE001
                pass                      # malformed body is the upstream's problem to report
        self._relay(raw)


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, default=8100)
    ap.add_argument("--upstream", default="http://127.0.0.1:8000")
    ap.add_argument("--thinking", default="true", choices=["true", "false"])
    ap.add_argument("--timeout", type=int, default=1800)
    ARGS = ap.parse_args()
    ARGS.thinking = ARGS.thinking == "true"
    print(f"proxy :{ARGS.listen} -> {ARGS.upstream}  forcing thinking={ARGS.thinking}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", ARGS.listen), Handler).serve_forever()


if __name__ == "__main__":
    main()
