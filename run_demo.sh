#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "[1/5] 检查系统架构..."
MACHINE_ARCH="$(uname -m)"
if [[ "$MACHINE_ARCH" != "arm64" ]]; then
  echo "⚠️ 当前机器架构是 $MACHINE_ARCH，不是 arm64。"
  echo "   本脚本按 Apple Silicon (M1/M2/M3) 环境配置。"
fi

echo "[2/5] 准备共享虚拟环境（固定路径，避免项目搬家后重复下载）..."
VENV_ROOT="$HOME/.demand_project_envs"
VENV_NAME="venv_tf213_py311_arm64"
VENV_DIR="$VENV_ROOT/$VENV_NAME"
mkdir -p "$VENV_ROOT"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "[3/5] 校验 Python 架构..."
PY_ARCH="$(python -c 'import platform; print(platform.machine())')"
if [[ "$PY_ARCH" != "arm64" ]]; then
  echo "❌ 当前虚拟环境 Python 架构是 $PY_ARCH，非 arm64。"
  echo "   请删除共享环境后重试：rm -rf \"$VENV_DIR\" && ./run_demo.sh"
  exit 1
fi

echo "[4/5] 安装依赖（仅首次）..."
check_deps() {
  python - <<'PY'
import importlib.metadata as md

required = {
    "tensorflow-macos": "2.13.0",
    "keras": "2.13.1",
    "numpy": "1.24.3",
    "scikit-learn": "1.3.2",
    "pandas": "2.0.3",
    "h5py": "3.11.0",
    "matplotlib": "3.7.5",
    "seaborn": "0.13.2",
}

ok = True
for pkg, ver in required.items():
    try:
        current = md.version(pkg)
    except Exception:
        ok = False
        break
    if current != ver:
        ok = False
        break

raise SystemExit(0 if ok else 1)
PY
}

DEPS_MARKER="$VENV_DIR/.deps_installed_v1"
if ! check_deps || [[ ! -f "$DEPS_MARKER" ]]; then
  export PIP_DISABLE_PIP_VERSION_CHECK=1
  python -m pip install --upgrade pip setuptools wheel
  pip install \
    tensorflow-macos==2.13.0 \
    keras==2.13.1 \
    numpy==1.24.3 \
    scikit-learn==1.3.2 \
    pandas==2.0.3 \
    h5py==3.11.0 \
    matplotlib==3.7.5 \
    seaborn==0.13.2
  touch "$DEPS_MARKER"
else
  echo "依赖已安装且版本匹配，跳过。"
fi

echo "[5/5] 启动实验脚本..."
python run_all_experiments.py
