#!/bin/sh
set -eu

cat >/usr/share/nginx/html/runtime-config.js <<EOF
window.__GOEUROOPS_CONFIG__ = {
  apiUrl: "${API_URL:-/api}"
};
EOF

exec nginx -g "daemon off;"
