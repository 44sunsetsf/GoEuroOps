#!/usr/bin/env bash
# 汇总 Caddy 访问日志：每个来源 IP 的首次 / 最近访问时间（北京时间）、请求数和访问过的页面。
#
#   deploy/who-visited.sh              # 默认 goeuroops
#   deploy/who-visited.sh vparser      # 其他站点：vparser、jaeger-vparser
#   deploy/who-visited.sh vparser 3    # 只看最近 3 天
set -euo pipefail

SITE="${1:-goeuroops}"
DAYS="${2:-0}"

read -r -d '' REPORT <<'PY' || true
import json, sys, time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone

cn = timezone(timedelta(hours=8))
days = int(sys.argv[1])
since = time.time() - days * 86400 if days else 0
static = (".js", ".css", ".png", ".svg", ".ico", ".woff", ".woff2", ".jpg", ".map")

visitors = OrderedDict()
for line in sys.stdin:
    try:
        e = json.loads(line)
    except ValueError:
        continue
    if e.get("ts", 0) < since:
        continue
    req = e.get("request", {})
    ip = req.get("client_ip") or req.get("remote_ip") or "?"
    path = req.get("uri", "").split("?")[0]
    v = visitors.setdefault(ip, {"first": e["ts"], "last": e["ts"], "n": 0, "pages": OrderedDict()})
    v["last"] = e["ts"]
    v["n"] += 1
    if not path.endswith(static) and not path.startswith("/assets/"):
        v["pages"]["%s %s" % (path, e.get("status", ""))] = None

if not visitors:
    print("还没有访问记录")
    sys.exit(0)

def fmt(ts):
    return datetime.fromtimestamp(ts, cn).strftime("%m-%d %H:%M")

for ip, v in sorted(visitors.items(), key=lambda kv: kv[1]["last"], reverse=True):
    print("%-16s %s → %s  共 %d 次请求" % (ip, fmt(v["first"]), fmt(v["last"]), v["n"]))
    for page in list(v["pages"])[:8]:
        print("    " + page)
PY

sudo docker exec goeuroops-caddy-1 sh -c "cat /data/logs/${SITE}-access*.log 2>/dev/null" | python3 -c "$REPORT" "$DAYS"
