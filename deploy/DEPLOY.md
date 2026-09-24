# 公网部署

把 GoEuroOps 部署到一台云服务器上，给别人一个 HTTPS 链接就能打开。

公开访问时有这些保护：

| 风险 | 保护 |
|---|---|
| 有人反复调用对话接口，消耗模型 API 额度 | 每个 IP 每分钟最多 6 次（nginx）；全站每天最多 `GOEUROOPS_DAILY_CHAT_LIMIT` 次（默认 200，按北京时间重置） |
| 后台、评测、知识库写入被任意访问 | 除学生端首页、对话和价目接口外，全部要口令（Caddy basic auth） |
| 后端端口被直接访问 | 只有 Caddy 对外开放 80/443，后端和前端容器不暴露端口 |

## 服务器要求

- 2 vCPU / 2 GiB 内存起步，Ubuntu 22.04 或 24.04。向量走 API、ChromaDB 嵌入式后，全部容器约 200MB
- 云厂商防火墙放行 TCP 22、80、443
- 访问者主要在国内时，选中国香港或新加坡地域，免备案

## 第一次部署

以下命令在服务器上执行。

**1. 安装 Docker，加 2GB swap（构建镜像时内存会紧）**

```bash
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2 git
sudo usermod -aG docker $USER   # 重新登录后生效
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

**2. 拉代码**

```bash
git clone https://github.com/44sunsetsf/GoEuroOps.git ~/goeuroops && cd ~/goeuroops
```

**3. 生成后台口令的哈希**

```bash
docker run --rm caddy:2-alpine caddy hash-password --plaintext '你的后台密码'
```

**4. 写 `.env`**（不会提交到 Git）

```bash
cat > .env <<'EOF'
ANTHROPIC_API_KEY=你的 key
ANTHROPIC_MODEL=deepseek-v4-pro
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic

# 没有域名时用 sslip.io：把服务器 IP 写在前面即可，Caddy 会自动申请证书
SITE_ADDRESS=1.2.3.4.sslip.io
STUDIO_USER=studio
# 第 3 步的输出；必须用单引号，否则哈希里的 $ 会被 compose 当成变量
STUDIO_PASSWORD_HASH='$2a$14$...'
GOEUROOPS_DAILY_CHAT_LIMIT=200
# 向量 API（硅基流动）：生产环境不在本机跑向量模型，省内存
SILICONFLOW_API_KEY=你的硅基流动 key
EOF
chmod 600 .env
```

**5. 启动**

```bash
docker compose -f docker-compose.yml -f deploy/docker-compose.prod.yml up -d --build
```

首次构建需要几分钟。完成后打开 `https://SITE_ADDRESS`：

- `/`：学生端首页，公开
- `/studio`：工作室后台，浏览器会弹出用户名和密码

## 更新

```bash
cd ~/goeuroops && git pull
docker compose -f docker-compose.yml -f deploy/docker-compose.prod.yml up -d --build
```

## 常用操作

```bash
alias dc='docker compose -f docker-compose.yml -f deploy/docker-compose.prod.yml'
dc ps                        # 状态
dc logs -f backend           # 后端日志
dc logs caddy | grep -i cert # 证书申请情况
dc restart backend           # 改了 backend/business/ 后生效
```

## 换成正式域名

域名 A 记录指向服务器 IP，然后把 `.env` 里的 `SITE_ADDRESS` 改成域名，执行 `dc up -d caddy`。

## 同一台机器上再放一个项目

Caddy 会加载 `deploy/sites/*.caddy`。其他项目把自己的服务接到 `goeuroops_default` 网络上，再在这里放一个站点配置即可共用 HTTPS，例如 Vparser 的部署说明见它仓库里的 `deploy/DEPLOY.md`。这些 `.caddy` 文件含口令哈希，已在 `.gitignore` 里排除。

静态站点（例如个人主页）：`.env` 里设 `CADDY_STATIC_DIR=/某个目录`（挂载到 Caddy 的 `/srv`），站点配置里用 `root * /srv/<子目录>` 加 `file_server`，不占额外内存。
要把主地址让给别的站点时，在 `.env` 里设 `GOEUROOPS_ADDRESS=goeuroops.<SITE_ADDRESS>`；不设时 GoEuroOps 就在主地址。

## 谁访问过

Caddy 把每个站点的访问记录写在自己的数据卷里（`/data/logs/<站点>-access.log`，10MB 轮转、保留 90 天），
记录来源 IP、时间、路径和状态码；请求头不落盘，进入链接里的 `key` 会被替换成 `REDACTED`。

```bash
deploy/who-visited.sh              # GoEuroOps
deploy/who-visited.sh vparser      # Vparser（同机部署时）
deploy/who-visited.sh vparser 3    # 只看最近 3 天
```
