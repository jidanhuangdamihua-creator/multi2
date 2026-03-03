# 需求预测项目可提交清单（教师版）

## 1. 项目目标
- 项目主题：多源迁移学习用于新品需求预测（Demand Forecasting）。
- 统一评估指标：RMSE（越低越好）。
- 展示入口：推荐使用一键脚本运行统一实验评估流水线。

---

## 2. 推荐演示主线（建议老师查看）

### 2.1 一键运行入口
- `run_demo.command`：macOS 双击入口。
- `run_demo.sh`：终端一键入口（自动检查环境、安装依赖、运行总实验）。

### 2.2 核心实验脚本
- `run_all_experiments.py`
  - 统一遍历三个数据集：`processed_data`、`processed_data_dataset2`、`processed_data_dataset3`
  - 统一评估五种方法：No-TL、SS-TL、MSWA-TL、MSSB-TL、MSML-TL
  - 输出：终端中的跨数据集汇总表（RMSE、提升百分比、最佳方法、总体统计）

---

## 3. 数据与预处理清单

### 3.1 原始数据来源
- Dataset 1：`demand-forecasting-kernels-only (1)/train.csv`
- Dataset 2：`hierarchical_sales_data.csv`
- Dataset 3：`rossmann-store-sales (2)/train ross.csv`
- 辅助信息：`rossmann-store-sales (2)/store ross.csv`

### 3.2 预处理脚本与产物
1) `data_preprocessing.py`
- 用途：Dataset 1 预处理（早期/基础版）。
- 产物目录：`processed_data/`
- 主要产物：`X_source_train.npy`、`X_target_train.npy` 等单源命名文件。

2) `patch_dataset1.py`
- 用途：Dataset 1 补丁版（统一为多源命名结构，便于与 Dataset2/3 对齐）。
- 产物目录：`processed_data/`
- 主要产物：`X_source1_train.npy`~`X_source3_val.npy`、`source_info.npy`、`X_target_*.npy`、`y_target_*.npy`。

3) `data_preprocessing_dataset2.py`
- 用途：Dataset 2 预处理与源域选择。
- 产物目录：`processed_data_dataset2/`
- 主要产物：`X/y_target_*`、`X/y_source1~3_*`、`source_info.npy`。

4) `data_preprocessing_dataset3.py`
- 用途：Dataset 3 预处理与源域选择。
- 产物目录：`processed_data_dataset3/`
- 主要产物：`X/y_target_*`、`X/y_source1~3_*`、`source_info.npy`。

5) `data_preprocessing_v2.py`
- 用途：Store 1 的修正版预处理（用于分阶段实验脚本）。
- 产物目录：`processed_data/store_1/`

---

## 4. 训练/实验脚本清单

### 4.1 当前推荐保留并可直接解释的脚本
- `run_all_experiments.py`：统一实验评估主脚本（建议主讲）。
- `train_phase1_v2.py`：Store 1 第一阶段（No-TL 与 SS-TL）。
- `train_multi_source_tl.py`：Store 1 多源迁移学习实验（会生成预训练权重）。
- `train_dataset2.py`：Dataset 2 单独实验版。
- `train_dataset3.py`：Dataset 3 单独实验版。

### 4.2 运行中可能生成的模型文件
- 目录：`models/`
- 示例：`pretrained_item2.weights.h5`、`pretrained_item7.weights.h5`、`pretrained_item8.weights.h5`

---

## 5. 可视化与辅助脚本
- `plot_results.py`：生成对比图 `results_comparison.png`。
- `check_data.py`：快速检查各数据集范围和结构。

---

## 6. 旧版/候选归档文件（非主线）
> 说明：以下文件并非“错误文件”，但与当前统一主线相比更偏历史版本或教学示例，可保留但建议标注为 legacy。

- `main.py`（示例性质入口）
- `process_data.py`（基础加载与归一化工具）
- `models.py`（基础 CNN 示例）
- `supply_chain.py`（库存计算示例函数）
- `train_phase1.py`（phase1 旧版，已由 v2 路径替代）
- `train_dataset2_final.py`（Dataset2 简化版）

---

## 7. 提交给老师时建议打包结构

建议至少包含：

1) 运行入口
- `run_demo.command`
- `run_demo.sh`
- `run_all_experiments.py`

2) 必要配置与说明
- `config.py`
- `ENVIRONMENT_REQUIREMENTS.md`
- `FILE_INVENTORY.md`（本文件）

3) 数据目录（已处理）
- `processed_data/`
- `processed_data_dataset2/`
- `processed_data_dataset3/`

4) 原始数据目录（如需复现预处理）
- `demand-forecasting-kernels-only (1)/`
- `hierarchical_sales_data.csv`
- `rossmann-store-sales (2)/`

---

## 8. 教师现场复现实验步骤（最短路径）
1. 双击 `run_demo.command`
2. 等待脚本自动完成环境检查与实验运行
3. 在终端查看汇总输出：
   - RMSE 对比表格
   - 相对 No-TL 提升百分比
   - 各数据集最佳方法
   - 总体统计

---

## 9. 当前状态结论
- 统一主线可运行：是（已可在当前设备执行完成）。
- 一键运行入口可用：是（`run_demo.command` / `run_demo.sh`）。
- 可提交给老师：是（建议附带本清单和环境说明）。
