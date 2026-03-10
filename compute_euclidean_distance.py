import argparse
import os
import numpy as np


def _load_source(data_dir: str, source_idx: int, split: str):
    x_path = os.path.join(data_dir, f"X_source{source_idx}_{split}.npy")
    if os.path.exists(x_path):
        return np.load(x_path)

    if source_idx == 1:
        legacy_path = os.path.join(data_dir, f"X_source_{split}.npy")
        if os.path.exists(legacy_path):
            return np.load(legacy_path)

    raise FileNotFoundError(f"未找到源域文件: {x_path}")


def _load_target(data_dir: str, split: str):
    x_path = os.path.join(data_dir, f"X_target_{split}.npy")
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"未找到目标域文件: {x_path}")
    return np.load(x_path)


def _flatten(x: np.ndarray):
    return x.reshape(x.shape[0], -1)


def compute_distances(source_x: np.ndarray, target_x: np.ndarray):
    source_flat = _flatten(source_x)
    target_flat = _flatten(target_x)

    source_centroid = source_flat.mean(axis=0)
    target_centroid = target_flat.mean(axis=0)
    centroid_distance = float(np.linalg.norm(source_centroid - target_centroid))

    diff = target_flat[:, None, :] - source_flat[None, :, :]
    pairwise = np.linalg.norm(diff, axis=2)
    nearest_each_target = pairwise.min(axis=1)

    return {
        "centroid_distance": centroid_distance,
        "nearest_min": float(nearest_each_target.min()),
        "nearest_mean": float(nearest_each_target.mean()),
        "nearest_max": float(nearest_each_target.max()),
        "target_count": int(target_flat.shape[0]),
        "source_count": int(source_flat.shape[0]),
    }


def main():
    parser = argparse.ArgumentParser(description="计算源商品与目标（预测）商品的欧几里得距离")
    parser.add_argument("--data-dir", default="processed_data", help="数据目录")
    parser.add_argument("--source-idx", type=int, default=1, choices=[1, 2, 3], help="源域编号")
    parser.add_argument("--source-split", default="train", choices=["train", "val", "test"], help="源域数据分割")
    parser.add_argument("--target-split", default="test", choices=["train", "val", "test"], help="目标域数据分割")
    args = parser.parse_args()

    source_x = _load_source(args.data_dir, args.source_idx, args.source_split)
    target_x = _load_target(args.data_dir, args.target_split)

    result = compute_distances(source_x, target_x)

    print("=== 欧几里得距离结果 ===")
    print(f"数据目录: {args.data_dir}")
    print(f"源域: source{args.source_idx}_{args.source_split}  样本数: {result['source_count']}")
    print(f"目标域: target_{args.target_split}  样本数: {result['target_count']}")
    print(f"源域中心 vs 目标域中心 距离: {result['centroid_distance']:.6f}")
    print("每个目标样本到最近源样本的距离统计:")
    print(f"  min : {result['nearest_min']:.6f}")
    print(f"  mean: {result['nearest_mean']:.6f}")
    print(f"  max : {result['nearest_max']:.6f}")


if __name__ == "__main__":
    main()
