"""
统一实验评估流水线脚本
循环遍历所有三个数据集，运行完整的多源迁移学习实验
输出跨数据集的 Markdown 格式汇总表格
"""

import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import os
import platform
import warnings

# 忽略警告
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    from keras.optimizers.legacy import Adam as LegacyAdam
except Exception:
    LegacyAdam = None

# ==========================================
# 全局配置
# ==========================================
RANDOM_SEED = 42
EPOCHS = 50
LEARNING_RATE = 0.001
INPUT_SHAPE = (10, 6)

# 数据集配置
DATASETS = {
    'Dataset 1': {
        'data_dir': 'processed_data',
        'weights': [0.3493, 0.3428, 0.3078],
        'description': 'Kaggle Demand Forecasting'
    },
    'Dataset 2': {
        'data_dir': 'processed_data_dataset2',
        'weights': [0.3928, 0.3052, 0.3020],
        'description': 'Hierarchical Sales Data'
    },
    'Dataset 3': {
        'data_dir': 'processed_data_dataset3',
        'weights': [0.3403, 0.3328, 0.3269],
        'description': 'Rossmann Store Sales'
    }
}

# ==========================================
# 核心模型构建函数
# ==========================================
def calculate_rmse(y_true, y_pred):
    """计算 RMSE"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def get_adam_optimizer(learning_rate):
    """Apple Silicon 上优先使用 legacy Adam，其他平台使用标准 Adam"""
    if (
        platform.system() == 'Darwin'
        and platform.machine() == 'arm64'
        and LegacyAdam is not None
    ):
        return LegacyAdam(learning_rate=learning_rate)
    return Adam(learning_rate=learning_rate)

def build_cnn():
    """构建 1D-CNN 模型 (Sequential)"""
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', 
               padding='same', input_shape=INPUT_SHAPE),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=get_adam_optimizer(LEARNING_RATE), loss='mse')
    return model

def build_cnn_functional():
    """使用函数式 API 构建模型（便于迁移学习）"""
    inputs = Input(shape=INPUT_SHAPE)
    x = Conv1D(filters=32, kernel_size=3, activation='relu', 
               padding='same', name='conv1d_1')(inputs)
    x = MaxPooling1D(pool_size=2, name='maxpool_1')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(50, activation='relu', name='dense_1')(x)
    outputs = Dense(1, activation='linear', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=get_adam_optimizer(LEARNING_RATE), loss='mse')
    return model

# ==========================================
# 数据加载函数
# ==========================================
def load_dataset(data_dir):
    """加载数据集"""
    data = {}
    
    # 目标域数据
    data['X_target_train'] = np.load(f'{data_dir}/X_target_train.npy')
    data['y_target_train'] = np.load(f'{data_dir}/y_target_train.npy')
    data['X_target_val'] = np.load(f'{data_dir}/X_target_val.npy')
    data['y_target_val'] = np.load(f'{data_dir}/y_target_val.npy')
    data['X_target_test'] = np.load(f'{data_dir}/X_target_test.npy')
    data['y_target_test'] = np.load(f'{data_dir}/y_target_test.npy')
    
    # 源域数据 - 尝试两种命名格式
    data['sources'] = []
    for i in range(1, 4):
        src = {}
        # 尝试 source{i} 格式 (Dataset 2, 3)
        if os.path.exists(f'{data_dir}/X_source{i}_train.npy'):
            src['X_train'] = np.load(f'{data_dir}/X_source{i}_train.npy')
            src['y_train'] = np.load(f'{data_dir}/y_source{i}_train.npy')
            src['X_val'] = np.load(f'{data_dir}/X_source{i}_val.npy')
            src['y_val'] = np.load(f'{data_dir}/y_source{i}_val.npy')
            data['sources'].append(src)
        # 尝试 source 格式 (Dataset 1)
        elif os.path.exists(f'{data_dir}/X_source_train.npy') and i == 1:
            src['X_train'] = np.load(f'{data_dir}/X_source_train.npy')
            src['y_train'] = np.load(f'{data_dir}/y_source_train.npy')
            src['X_val'] = np.load(f'{data_dir}/X_source_val.npy')
            src['y_val'] = np.load(f'{data_dir}/y_source_val.npy')
            data['sources'].append(src)
    
    return data

# ==========================================
# 训练算法函数
# ==========================================
def train_no_tl(data, n_runs=10):
    """No-TL: 仅使用目标域数据训练 (10次平均)"""
    rmses = []
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
    ]
    
    for i in range(n_runs):
        tf.random.set_seed(RANDOM_SEED + i)
        np.random.seed(RANDOM_SEED + i)
        
        model = build_cnn()
        model.fit(
            data['X_target_train'], data['y_target_train'],
            validation_data=(data['X_target_val'], data['y_target_val']),
            epochs=EPOCHS,
            batch_size=min(8, len(data['X_target_train'])),
            callbacks=callbacks,
            verbose=0
        )
        
        preds = model.predict(data['X_target_test'], verbose=0).flatten()
        rmse = calculate_rmse(data['y_target_test'], preds)
        rmses.append(rmse)
    
    return np.mean(rmses)

def train_ss_tl(data):
    """SS-TL: 单源迁移学习 (使用 Source 1)"""
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    src = data['sources'][0]
    
    # 预训练
    model = build_cnn_functional()
    model.fit(
        src['X_train'], src['y_train'],
        validation_data=(src['X_val'], src['y_val']),
        epochs=EPOCHS // 2,
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )
    
    # 冻结卷积层，微调
    for layer in model.layers:
        if 'conv' in layer.name or 'maxpool' in layer.name:
            layer.trainable = False
    
    model.compile(optimizer=get_adam_optimizer(LEARNING_RATE * 0.1), loss='mse')
    model.fit(
        data['X_target_train'], data['y_target_train'],
        validation_data=(data['X_target_val'], data['y_target_val']),
        epochs=EPOCHS,
        batch_size=min(8, len(data['X_target_train'])),
        callbacks=callbacks,
        verbose=0
    )
    
    preds = model.predict(data['X_target_test'], verbose=0).flatten()
    return calculate_rmse(data['y_target_test'], preds)

def train_mswa_tl(data, weights):
    """MSWA-TL: 多源加权平均"""
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    finetuned_models = []
    
    for src in data['sources']:
        model = build_cnn_functional()
        model.fit(
            src['X_train'], src['y_train'],
            validation_data=(src['X_val'], src['y_val']),
            epochs=EPOCHS // 2,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        for layer in model.layers:
            if 'conv' in layer.name or 'maxpool' in layer.name:
                layer.trainable = False
        
        model.compile(optimizer=get_adam_optimizer(LEARNING_RATE * 0.1), loss='mse')
        model.fit(
            data['X_target_train'], data['y_target_train'],
            validation_data=(data['X_target_val'], data['y_target_val']),
            epochs=EPOCHS,
            batch_size=min(8, len(data['X_target_train'])),
            callbacks=callbacks,
            verbose=0
        )
        
        finetuned_models.append(model)
    
    # 加权预测
    predictions = []
    w = np.array(weights[:len(finetuned_models)])
    w = w / w.sum()
    
    for model in finetuned_models:
        pred = model.predict(data['X_target_test'], verbose=0).flatten()
        predictions.append(pred)
    
    predictions = np.array(predictions)
    weighted_pred = np.sum(predictions * w.reshape(-1, 1), axis=0)
    
    return calculate_rmse(data['y_target_test'], weighted_pred), finetuned_models

def train_mssb_tl(data, finetuned_models):
    """MSSB-TL: 多源切换 (选择验证集最优)"""
    val_rmses = []
    for model in finetuned_models:
        val_pred = model.predict(data['X_target_val'], verbose=0).flatten()
        val_rmse = calculate_rmse(data['y_target_val'], val_pred)
        val_rmses.append(val_rmse)
    
    best_idx = np.argmin(val_rmses)
    best_model = finetuned_models[best_idx]
    
    test_pred = best_model.predict(data['X_target_test'], verbose=0).flatten()
    return calculate_rmse(data['y_target_test'], test_pred)

def train_msadw_tl(data, finetuned_models):
    """MSADW-TL: 多源自适应动态加权 (路线 A: 预测层集成 + 静态反比加权)

    放弃参数层融合（直接揉捏卷积核会破坏特征提取能力），
    改为在 3 个已微调模型的"最终预测结果"上施加自适应权重。
    权重由路线 A（静态反比加权公式）根据目标域验证集 RMSE 计算：
        w_i = (1 / RMSE_i) / sum(1 / RMSE_j)
    验证集 RMSE 越小的模型获得越高的权重。
    """
    # 计算各模型在目标域验证集上的 RMSE
    val_rmses = []
    for model in finetuned_models:
        val_pred = model.predict(data['X_target_val'], verbose=0).flatten()
        val_rmse = calculate_rmse(data['y_target_val'], val_pred)
        val_rmses.append(val_rmse)

    # 路线 A: 静态反比加权 (加小量 epsilon 防止除零)
    inv_rmses = np.array([1.0 / (r + 1e-10) for r in val_rmses])
    weights = inv_rmses / inv_rmses.sum()
    print(f"  自适应权重 (验证集反比加权): {[f'{w:.4f}' for w in weights]}")

    # 预测层集成：对测试集预测结果加权求和
    test_preds = []
    for model in finetuned_models:
        pred = model.predict(data['X_target_test'], verbose=0).flatten()
        test_preds.append(pred)

    test_preds = np.array(test_preds)
    weighted_pred = np.sum(test_preds * weights.reshape(-1, 1), axis=0)

    return calculate_rmse(data['y_target_test'], weighted_pred)

# ==========================================
# 主评估函数
# ==========================================
def evaluate_dataset(dataset_name, data_dir, weights):
    """评估单个数据集上的所有算法"""
    print(f"\n{'='*70}")
    print(f"评估 {dataset_name}: {data_dir}")
    print(f"先验权重: {weights}")
    print('='*70)
    
    # 加载数据
    data = load_dataset(data_dir)
    print(f"目标域: 训练={data['X_target_train'].shape}, 测试={data['X_target_test'].shape}")
    print(f"源域数量: {len(data['sources'])}")
    
    results = {}
    
    # 1. No-TL
    print("\n[1/5] 运行 No-TL (10次平均)...")
    results['No-TL'] = train_no_tl(data, n_runs=10)
    print(f"  No-TL RMSE: {results['No-TL']:.4f}")
    
    # 2. SS-TL
    print("[2/5] 运行 SS-TL...")
    results['SS-TL'] = train_ss_tl(data)
    print(f"  SS-TL RMSE: {results['SS-TL']:.4f}")
    
    # 3. MSWA-TL
    print("[3/5] 运行 MSWA-TL...")
    results['MSWA-TL'], finetuned_models = train_mswa_tl(data, weights)
    print(f"  MSWA-TL RMSE: {results['MSWA-TL']:.4f}")
    
    # 4. MSSB-TL
    print("[4/5] 运行 MSSB-TL...")
    results['MSSB-TL'] = train_mssb_tl(data, finetuned_models)
    print(f"  MSSB-TL RMSE: {results['MSSB-TL']:.4f}")
    
    # 5. MSADW-TL
    print("[5/5] 运行 MSADW-TL (自适应预测层集成, 路线 A 反比加权)...")
    results['MSADW-TL'] = train_msadw_tl(data, finetuned_models)
    print(f"  MSADW-TL RMSE: {results['MSADW-TL']:.4f}")
    
    return results

# ==========================================
# 主程序
# ==========================================
def main():
    print("\n" + "="*80)
    print(" " * 20 + "统一实验评估流水线")
    print(" " * 15 + "Multi-Source Transfer Learning Benchmark")
    print("="*80)
    
    # 存储所有结果
    all_results = {}
    
    # 循环遍历所有数据集
    for dataset_name, config in DATASETS.items():
        if os.path.exists(config['data_dir']):
            results = evaluate_dataset(
                dataset_name, 
                config['data_dir'], 
                config['weights']
            )
            all_results[dataset_name] = results
        else:
            print(f"\n⚠️ 跳过 {dataset_name}: 目录 {config['data_dir']} 不存在")
    
    # ==========================================
    # 输出 Markdown 格式汇总表格
    # ==========================================
    print("\n\n" + "="*80)
    print(" " * 25 + "📊 实验结果汇总")
    print("="*80)
    
    # 表头
    print("\n### RMSE 对比表格\n")
    header = "| 方法 |"
    separator = "|------|"
    for ds in all_results.keys():
        header += f" {ds} |"
        separator += "--------|"
    print(header)
    print(separator)
    
    # 各方法行
    methods = ['No-TL', 'SS-TL', 'MSWA-TL', 'MSSB-TL', 'MSADW-TL']
    for method in methods:
        row = f"| {method} |"
        for ds, results in all_results.items():
            if method in results:
                row += f" {results[method]:.4f} |"
            else:
                row += " - |"
        print(row)
    
    # 计算并输出提升百分比
    print("\n### 相对 No-TL 的提升百分比\n")
    header = "| 方法 |"
    separator = "|------|"
    for ds in all_results.keys():
        header += f" {ds} |"
        separator += "--------|"
    print(header)
    print(separator)
    
    for method in methods[1:]:  # 跳过 No-TL
        row = f"| {method} |"
        for ds, results in all_results.items():
            if method in results and 'No-TL' in results:
                no_tl = results['No-TL']
                improvement = (no_tl - results[method]) / no_tl * 100
                row += f" {improvement:+.2f}% |"
            else:
                row += " - |"
        print(row)
    
    # 最佳方法汇总
    print("\n### 各数据集最佳方法\n")
    print("| 数据集 | 最佳方法 | RMSE | 提升幅度 |")
    print("|--------|----------|------|----------|")
    
    for ds, results in all_results.items():
        no_tl = results['No-TL']
        best_method = None
        best_rmse = float('inf')
        
        for method in methods[1:]:
            if method in results and results[method] < best_rmse:
                best_rmse = results[method]
                best_method = method
        
        if best_method:
            improvement = (no_tl - best_rmse) / no_tl * 100
            print(f"| {ds} | {best_method} | {best_rmse:.4f} | +{improvement:.2f}% |")
    
    # 总体统计
    print("\n### 总体统计\n")
    all_improvements = []
    for ds, results in all_results.items():
        no_tl = results['No-TL']
        for method in methods[1:]:
            if method in results:
                improvement = (no_tl - results[method]) / no_tl * 100
                all_improvements.append(improvement)
    
    if all_improvements:
        print(f"- 平均提升: **{np.mean(all_improvements):.2f}%**")
        print(f"- 最大提升: **{np.max(all_improvements):.2f}%**")
        print(f"- 最小提升: **{np.min(all_improvements):.2f}%**")
    
    print("\n" + "="*80)
    print(" " * 25 + "✅ 实验完成!")
    print("="*80)
    
    return all_results


if __name__ == "__main__":
    results = main()