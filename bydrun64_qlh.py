import torch
import torch.nn as nn
import numpy as np
import json

# ---------- 与训练完全一致的特征工程（3 个特征）----------
def add_motion_features(seq):
    """
    输入: seq (numpy array, shape = (64, 2))  原始绝对坐标
    输出: features (numpy array, shape = (64, 3))
          包含 [dx, dy, ddir_x]
    """
    diff = np.diff(seq, axis=0)
    diff = np.vstack([diff, diff[-1]])
    norm = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-8
    dir_vec = diff / norm
    dir_diff = np.diff(dir_vec, axis=0)
    dir_diff = np.vstack([dir_diff, dir_diff[-1]])
    # 仅保留 dx, dy, ddir_x
    features = np.column_stack([diff, dir_diff[:, 0]])
    return features

# ---------- 模型定义（与训练完全一致：input_dim=3, hidden_dim=8）----------
class GRUFixedLenClassifier(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=8, num_layers=1, num_classes=6, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out)
        feat = out.mean(dim=1)              # 全序列平均池化
        feat = self.ln(feat)
        logits = self.fc(feat)
        return logits

# ---------- 加载模型 ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GRUFixedLenClassifier(input_dim=3, hidden_dim=8, num_classes=6)   # 新参数
model.load_state_dict(torch.load('road_shape_gru_64.pth', map_location=device))
model.to(device)
model.eval()

# ---------- 加载 Z‑score 归一化统计量（现在为 3 维）----------
stats = np.load('feat_stats.npz')
feat_mean = torch.as_tensor(stats['mean'], dtype=torch.float32).to(device)
feat_std  = torch.as_tensor(stats['std'], dtype=torch.float32).to(device)

# ---------- 标签名称映射 ----------
idx_to_label = {
    0: "zhixian",
    1: "shizi",
    2: "daodazuohuandao",
    3: "jinruzuohuandao",
    4: "daodayouhuandao",
    5: "jinruyouhuandao",
}

# ---------- 从 JSON 加载数据并过滤 ----------
def load_valid_samples(json_path, seq_len=64):
    with open(json_path, 'r') as f:
        raw_data = json.load(f)

    valid_points = []
    valid_labels = []
    skipped = 0
    for item in raw_data:
        pts = np.array(item["point"], dtype=np.float32)
        if pts.shape[0] != seq_len:
            skipped += 1
            continue
        valid_points.append(pts)
        valid_labels.append(item["label"])

    print(f"有效样本数: {len(valid_points)}，跳过 {skipped} 个长度不为 {seq_len} 的样本。")
    return valid_points, valid_labels

# ---------- 推理函数 ----------
def predict_all_probs(points):
    """
    points: np.array, 形状 (64, 2)
    返回: (概率列表, 预测类别索引)
    """
    # 1. 特征工程（输出 (64, 3)）
    feat = add_motion_features(points)                     # (64, 3)
    feat = torch.as_tensor(feat, dtype=torch.float32)     # tensor

    # 2. Z‑score 归一化
    feat = (feat - feat_mean) / feat_std

    # 3. 增加 batch 维度
    feat = feat.unsqueeze(0).to(device)                    # (1, 64, 3)

    # 4. 推理
    with torch.no_grad():
        logits = model(feat)                               # (1, num_classes)
        probs = torch.softmax(logits, dim=1)
        probs = probs.squeeze(0).cpu().numpy()
        pred_idx = int(np.argmax(probs))
    return probs, pred_idx

# ---------- 主程序 ----------
if __name__ == "__main__":
    json_path = "byd_2026_5_26.json"  # 请修改为实际路径
    seq_len = 64

    points_list, label_list = load_valid_samples(json_path, seq_len)

    for i, (pts, true_label) in enumerate(zip(points_list, label_list)):
        probs, pred_id = predict_all_probs(pts)
        print(f"\n--- 样本 {i+1} (真实标签: {true_label}) ---")
        print("各类别概率：")
        for class_id, prob in enumerate(probs):
            class_name = idx_to_label.get(class_id, f"class_{class_id}")
            print(f"  {class_name:12s}: {prob:.4f}")
        predicted_name = idx_to_label[pred_id]
        print(f"预测类别: {predicted_name} (索引 {pred_id})")