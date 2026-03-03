# 软件正常运行环境要求（macOS 版本）

## 1) 操作系统与硬件
- macOS（建议 Monterey 12+）
- Apple Silicon 芯片（M1/M2/M3，`arm64`）

## 2) Python 与虚拟环境
- Python 3.9（推荐）
- 虚拟环境：项目根目录下 `.venv_arm`
- 必须使用 `arm64` 的 Python 解释器（避免 `x86_64/Rosetta` 导致 `Illegal instruction` / Exit Code 132）

## 3) 关键 Python 依赖
- tensorflow-macos==2.13.0
- keras==2.13.1
- numpy==1.24.3
- scikit-learn==1.3.2
- pandas==2.0.3
- h5py==3.11.0
- matplotlib==3.7.5
- seaborn==0.13.2

## 4) 数据目录要求（项目根目录下）
- `processed_data/`
- `processed_data_dataset2/`
- `processed_data_dataset3/`

## 5) 一键运行方式
- 命令：`./run_demo.sh`
- 首次运行会自动创建 `.venv_arm` 并安装依赖；后续会直接运行实验。

## 6) 常见问题
- 若出现 `Exit Code: 132`：通常是解释器架构不匹配（x86_64 Python 跑在 M 芯片上）。
- 解决：删除环境后重建
  - `rm -rf .venv_arm`
  - `./run_demo.sh`
