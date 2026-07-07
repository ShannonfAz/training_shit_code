"""
permutation_importance.py
使用排列重要性评估输入 5 个特征对 GRU 模型分类的影响。
直接读取 JSON 数据，应用特征工程与 Z-score 归一化。
"""

import numpy as np
import torch
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# ========== 根据你的项目调整导入路径 ==========
from bydrun64 import GRUFixedLenClassifier   # 你的模型定义

# ==================== 特征工程（与训练时完全一致） ====================
def add_motion_features(seq):
    """输入 (64,2) 输出 (64,5) 包含 dx,dy,ddir_x,ddir_y,step_len"""
    diff = np.diff(seq, axis=0)                     # (63,2)
    diff = np.vstack([diff, diff[-1]])              # (64,2)
    norm = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-8
    dir_vec = diff / norm
    dir_diff = np.diff(dir_vec, axis=0)
    dir_diff = np.vstack([dir_diff, dir_diff[-1]])  # (64,2)
    step_len = norm                                 # (64,1)
    features = np.concatenate([diff, dir_diff, step_len], axis=1)
    return features

def compute_feature_stats(sequences):
    """计算所有样本的特征均值和标准差"""
    all_feats = [add_motion_features(seq) for seq in sequences]
    all_feats = np.concatenate(all_feats, axis=0)
    mean = all_feats.mean(axis=0)
    std = all_feats.std(axis=0) + 1e-8
    return mean, std

# ==================== 自定义 Dataset（仅用于验证集） ====================
class RoadShapeDataset(Dataset):
    def __init__(self, sequences, labels, mean, std):
        self.sequences = sequences  # list of numpy (64,2)
        self.labels = labels        # list of int
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx].copy()
        feat = add_motion_features(seq)            # (64,5)
        feat = (feat - self.mean) / self.std
        feat = torch.as_tensor(feat, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feat, label

def collate_fn(batch):
    seqs, labs = zip(*batch)
    return torch.stack(seqs, dim=0), torch.stack(labs, dim=0)

# ==================== 排列重要性计算 ====================
def permutation_importance(model, val_loader, device, feature_idx):
    """计算打乱某个特征后准确率的下降"""
    model.eval()
    correct_orig, correct_perm = 0, 0
    total = 0
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 原始准确率
        with torch.no_grad():
            out = model(inputs).argmax(dim=1)
            correct_orig += (out == labels).sum().item()

        # 打乱指定特征（在时间维上随机置换）
        perm = inputs.clone()
        for b in range(perm.size(0)):
            idx = torch.randperm(64, device=device)
            perm[b, :, feature_idx] = inputs[b, idx, feature_idx]

        with torch.no_grad():
            out_perm = model(perm).argmax(dim=1)
            correct_perm += (out_perm == labels).sum().item()
        total += labels.size(0)

    acc_orig = correct_orig / total
    acc_perm = correct_perm / total
    return acc_orig - acc_perm

# ==================== 主程序 ====================
if __name__ == '__main__':
    # ---------- 1. 加载数据 ----------
    JSON_PATH = "byd_2026_5_24.json"
    with open(JSON_PATH, 'r') as f:
        raw = json.load(f)

    label_map = {
        "zhixian": 0, "shizi": 1,
        "daodazuohuandao": 2, "jinruzuohuandao": 3,
        "daodayouhuandao": 4, "jinruyouhuandao": 5
    }

    points_list, labels_list = [], []
    for item in raw:
        pts = np.array(item["point"], dtype=np.float32)
        if pts.shape[0] != 64:
            continue
        points_list.append(pts)
        labels_list.append(label_map[item["label"]])

    # 划分验证集（使用全部数据作为验证集亦可，或者取一部分）
    X_val_raw, _, y_val, _ = train_test_split(
        points_list, labels_list, test_size=0.5, random_state=42, stratify=labels_list
    )
    # 计算特征统计量（使用全部样本，或只用训练集，此处为示例用验证集统计）
    mean, std = compute_feature_stats(points_list)

    val_dataset = RoadShapeDataset(X_val_raw, y_val, mean, std)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # ---------- 2. 加载模型 ----------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GRUFixedLenClassifier(
        input_dim=5, hidden_dim=16, num_layers=1, num_classes=6, dropout=0.0
    )
    model.load_state_dict(torch.load('road_shape_gru_64.pth', map_location=device))
    model.to(device)
    model.eval()

    # ---------- 3. 计算排列重要性 ----------
    feature_names = ['dx', 'dy', 'ddir_x', 'ddir_y', 'step_len']
    print("排列重要性（准确率下降值，越大越重要）：")
    for i, name in enumerate(feature_names):
        imp = permutation_importance(model, val_loader, device, i)
        print(f"  {name:10s}: {imp:.4f}")