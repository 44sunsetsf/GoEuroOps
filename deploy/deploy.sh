#!/usr/bin/env bash
# 把当前已推送的 main 部署到公网服务器。在自己电脑的仓库根目录运行：
#
#   deploy/deploy.sh                  自动判断：上次部署之后改了哪部分就部署哪部分
#   deploy/deploy.sh backend          只部署后端
#   deploy/deploy.sh frontend         只部署前端
#   deploy/deploy.sh backend frontend 两个都部署
#   deploy/deploy.sh rollback         把后端和前端换回上一次部署前的镜像
#
# 做法（服务器只有 1.6 GiB 内存，不能在上面完整构建镜像）：
#   1. 检查本地没有未提交、未推送的改动，服务器 git pull 到同一个提交
#   2. 前端在本机构建，dist 传到服务器；后端直接用服务器上拉下来的代码
#   3. 在固定的基础镜像（:base）上叠一层代码，得到新的 :latest；旧的 :latest 记为 :prev 供回滚
#   4. 只重建改动的容器，等后端健康检查通过，再从公网访问一次
#   5. 任何一步失败都自动回滚到 :prev
#
# 依赖变了（backend/requirements.txt、Dockerfile、前端 nginx 配置）不能用叠一层的办法，脚本会停下来提示。
set -euo pipefail

HOST="${DEPLOY_HOST:-admin@47.84.60.190}"
SITE="${DEPLOY_SITE:-https://goeuroops.yunfanteo.world}"
REMOTE_DIR="goeuroops"                       # 服务器上的仓库目录（相对 ~）
WORK="/tmp/goeuroops-deploy"                 # 服务器上的临时目录
DC="docker compose -f docker-compose.yml -f deploy/docker-compose.prod.yml"

cd "$(dirname "$0")/.."
say() { printf '\n▶ %s\n' "$*"; }
die() { printf '\n✗ %s\n' "$*" >&2; exit 1; }
remote() { ssh -o ConnectTimeout=15 "$HOST" "cd ~/$REMOTE_DIR && $*"; }

rollback() {
  say "回滚到上一次部署前的镜像"
  for svc in "$@"; do
    remote "docker image inspect goeuroops-$svc:prev >/dev/null 2>&1 && docker tag goeuroops-$svc:prev goeuroops-$svc:latest" \
      || echo "  $svc 没有 :prev 镜像，跳过"
  done
  remote "$DC up -d --no-build --no-deps $*"
}

wait_healthy() {
  for _ in $(seq 1 50); do
    [ "$(remote "docker inspect -f '{{.State.Health.Status}}' goeuroops-backend-1" 2>/dev/null)" = healthy ] && return 0
    sleep 3
  done
  return 1
}

smoke() {
  for path in / /api/catalog; do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "$SITE$path")
    [ "$code" = 200 ] || { echo "  $SITE$path 返回 $code"; return 1; }
  done
}

if [ "${1:-}" = rollback ]; then
  shift
  [ $# -gt 0 ] || set -- backend frontend
  rollback "$@"
  wait_healthy && smoke && say "已回滚，站点正常" && exit 0
  die "回滚后站点仍不正常，请登录服务器查看：$DC logs --tail 50 backend"
fi

# ── 1. 本地和服务器对齐到同一个提交 ──────────────────────────────────────────────
[ -z "$(git status --porcelain)" ] || die "有未提交的改动，先提交"
git fetch -q origin
commit=$(git rev-parse HEAD)
[ "$commit" = "$(git rev-parse origin/main)" ] || die "本地 main 和 origin/main 不一致，先 git push"

[ -z "$(remote 'git status --porcelain --untracked-files=no')" ] \
  || die "服务器上的仓库有直接改过、没进 git 的文件。先确认这些改动是否要保留：ssh $HOST 'cd ~/$REMOTE_DIR && git diff'"

deployed=$(remote "cat .deployed 2>/dev/null || git rev-parse HEAD")
say "上次部署 ${deployed:0:7} → 这次 ${commit:0:7}"

targets=("$@")
if git cat-file -e "$deployed^{commit}" 2>/dev/null; then
  changed=$(git diff --name-only "$deployed" "$commit")
else
  changed=$(git ls-files)      # 服务器上的提交本地没有：当作全部改过
fi
if [ ${#targets[@]} -eq 0 ]; then
  grep -q '^backend/' <<<"$changed" && targets+=(backend)
  grep -q '^frontend/' <<<"$changed" && targets+=(frontend)
fi
blockers=$(grep -E '^(backend/requirements\.txt|backend/Dockerfile|frontend/Dockerfile|frontend/docker/)' <<<"$changed" || true)
[ -z "$blockers" ] || die "这些文件变了，叠一层代码不够，需要完整构建镜像（见 deploy/DEPLOY.md「完整构建」）：
$blockers"

remote "git pull -q --ff-only && git rev-parse --short HEAD" >/dev/null
[ "$(remote 'git rev-parse HEAD')" = "$commit" ] || die "服务器 git pull 后不是 ${commit:0:7}"

if grep -qE '^deploy/(Caddyfile|sites/)' <<<"$changed"; then
  say "Caddy 配置有变化：校验并重新加载"
  remote "docker exec goeuroops-caddy-1 caddy validate --config /etc/caddy/Caddyfile >/dev/null 2>&1 \
          && docker exec goeuroops-caddy-1 caddy reload --config /etc/caddy/Caddyfile 2>/dev/null" \
    || die "Caddy 配置校验或重载失败，站点仍在用旧配置"
fi

if [ ${#targets[@]} -eq 0 ]; then
  remote "echo $commit > .deployed"
  say "后端和前端都没有改动，无需重建容器"
  exit 0
fi
say "要部署：${targets[*]}"

# ── 2–3. 叠一层代码，生成新镜像 ─────────────────────────────────────────────────
remote "rm -rf $WORK && mkdir -p $WORK/fe"
for svc in "${targets[@]}"; do
  # 第一次用本脚本：把当前镜像定为基础镜像，以后每次都从它叠一层，镜像不会越叠越大
  remote "docker image inspect goeuroops-$svc:base >/dev/null 2>&1 || docker tag goeuroops-$svc:latest goeuroops-$svc:base"
  case "$svc" in
    frontend)
      say "本机构建前端"
      [ -d frontend/node_modules ] || npm --prefix frontend ci --silent
      npm --prefix frontend run build --silent >/dev/null
      rsync -a --delete frontend/dist/ "$HOST:$WORK/fe/dist/"
      remote "printf 'FROM goeuroops-frontend:base\nRUN rm -rf /usr/share/nginx/html/*\nCOPY dist/ /usr/share/nginx/html/\n' > $WORK/fe/Dockerfile \
              && docker build -q -t goeuroops-frontend:new $WORK/fe >/dev/null 2>&1"
      ;;
    backend)
      say "用服务器上的代码叠出后端镜像"
      remote "printf 'FROM goeuroops-backend:base\nUSER root\nRUN find /app -mindepth 1 -maxdepth 1 ! -name data ! -name logs ! -name config -exec rm -rf {} +\nCOPY --chown=echomind:echomind . /app\nUSER echomind\n' > $WORK/Dockerfile.backend \
              && docker build -q -t goeuroops-backend:new -f $WORK/Dockerfile.backend backend >/dev/null 2>&1"
      ;;
    *) die "不认识的部署目标：$svc（可选 backend、frontend）" ;;
  esac
done

# ── 4. 换镜像、重建容器、检查 ───────────────────────────────────────────────────
for svc in "${targets[@]}"; do
  remote "docker tag goeuroops-$svc:latest goeuroops-$svc:prev && docker tag goeuroops-$svc:new goeuroops-$svc:latest && docker rmi goeuroops-$svc:new >/dev/null"
done
say "重建容器：${targets[*]}"
# 从这里开始线上已经换成新镜像：后面任何一步失败都走回滚，不能直接退出
started=ok
remote "$DC up -d --no-build --no-deps ${targets[*]}" >/dev/null 2>&1 || started=failed

if [ "$started" = ok ] && wait_healthy && smoke; then
  remote "echo $commit > .deployed; docker image prune -f >/dev/null; rm -rf $WORK"
  say "部署完成：${commit:0:7} 已上线，$SITE 正常。回滚用 deploy/deploy.sh rollback"
else
  echo "✗ 新版本没有通过检查，自动回滚" >&2
  rollback "${targets[@]}"
  wait_healthy && smoke && die "已回滚到旧版本，站点正常。新版本的日志：ssh $HOST 'cd ~/$REMOTE_DIR && $DC logs --tail 80 backend'"
  die "回滚后站点仍不正常，请立即登录服务器检查"
fi
