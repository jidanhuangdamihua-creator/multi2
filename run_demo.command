#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

if [[ ! -f "run_demo.sh" ]]; then
  echo "❌ 未找到 run_demo.sh，请确认文件存在于项目根目录。"
  read -n 1 -s -r -p "按任意键退出..."
  echo
  exit 1
fi

chmod +x run_demo.sh
./run_demo.sh

echo
echo "✅ 运行结束。"
read -n 1 -s -r -p "按任意键关闭窗口..."
echo
