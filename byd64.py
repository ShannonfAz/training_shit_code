import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json

# -------------------- Focal Loss 定义（支持可调 gamma） --------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.0):   # 默认 gamma 降为 1.0
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

# -------------------- 特征工程 --------------------
def add_motion_features(seq):
    diff = np.diff(seq, axis=0)
    diff = np.vstack([diff, diff[-1]])
    norm = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-8
    dir_vec = diff / norm
    dir_diff = np.diff(dir_vec, axis=0)
    dir_diff = np.vstack([dir_diff, dir_diff[-1]])
    step_len = norm
    features = np.concatenate([diff, dir_diff, step_len], axis=1)
    return features

def compute_feature_stats(sequences):
    all_feats = []
    for seq in sequences:
        feat = add_motion_features(seq)
        all_feats.append(feat)
    all_feats = np.concatenate(all_feats, axis=0)
    mean = all_feats.mean(axis=0)
    std = all_feats.std(axis=0) + 1e-8
    return mean, std

# -------------------- 数据集 --------------------
class RoadShapeFixedLenDataset(Dataset):
    def __init__(self, point_sequences, labels, feature_mean, feature_std, augment=False):
        self.sequences = point_sequences
        self.labels = labels
        self.feature_mean = torch.as_tensor(feature_mean, dtype=torch.float32)
        self.feature_std = torch.as_tensor(feature_std, dtype=torch.float32)
        self.augment = augment

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx].copy()
        if self.augment:
            angle = np.random.uniform(-5, 5) * np.pi / 180
            rot = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle),  np.cos(angle)]])
            seq = seq @ rot.T
            shift_x = np.random.uniform(-0.02 * 320, 0.02 * 320)
            shift_y = np.random.uniform(-0.02 * 240, 0.02 * 240)
            seq[:, 0] += shift_x
            seq[:, 1] += shift_y
            seq += np.random.normal(0, 0.5, seq.shape)

        feat = add_motion_features(seq)
        feat = torch.as_tensor(feat, dtype=torch.float32)
        feat = (feat - self.feature_mean) / self.feature_std
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feat, label

def collate_fixed(batch):
    sequences, labels = zip(*batch)
    padded_seq = torch.stack(sequences, dim=0)
    labels = torch.stack(labels, dim=0)
    return padded_seq, labels

# -------------------- 模型 --------------------
class GRUFixedLenClassifier(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=16, num_layers=1, num_classes=6, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out)
        feat = out.mean(dim=1)
        feat = self.ln(feat)
        logits = self.fc(feat)
        return logits

# -------------------- 数据加载 --------------------
def load_data_from_json(json_path, seq_len=64):
    with open(json_path, 'r') as f:
        raw_data = json.load(f)

    label_map = {
        "zhixian": 0,
        "shizi": 1,
        "daodazuohuandao": 2,
        "jinruzuohuandao": 3,
        "daodayouhuandao": 4,
        "jinruyouhuandao": 5
    }

    sequences, labels = [], []
    skipped = 0
    for item in raw_data:
        pts = np.array(item["point"], dtype=np.float32)
        if pts.shape[0] != seq_len:
            skipped += 1
            continue
        sequences.append(pts)
        labels.append(label_map[item["label"]])

    print(f"加载数据：有效样本 {len(sequences)}，跳过 {skipped} 个长度不为 {seq_len} 的样本。")
    return sequences, labels

# -------------------- 训练函数（支持传入 gamma） --------------------
def train_model(model, train_loader, val_loader, epochs=100, lr=5e-4, device='cpu',
                class_weights=None, use_focal=False, gamma=1.0):   # 新增 gamma 参数
    model = model.to(device)

    if use_focal:
        criterion = FocalLoss(alpha=class_weights.to(device) if class_weights is not None else None, gamma=gamma)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)

        scheduler.step()
        avg_loss = total_loss / len(train_loader.dataset)

        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1:03d} | Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

    return model

def evaluate(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
    return accuracy_score(all_labels, all_preds)

# -------------------- 主流程 --------------------
def main():
    SEQ_LEN = 64
    NUM_CLASSES = 6
    BATCH_SIZE = 32
    HIDDEN_DIM = 16
    NUM_LAYERS = 1
    DROPOUT = 0.2
    EPOCHS = 100
    LR = 2e-3
    JSON_PATH = "byd_2026_5_26.json"

    sequences, labels = load_data_from_json(JSON_PATH, seq_len=SEQ_LEN)

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )

    feat_mean, feat_std = compute_feature_stats(X_train_raw)

    train_dataset = RoadShapeFixedLenDataset(
        X_train_raw, y_train, feat_mean, feat_std, augment=True
    )
    val_dataset = RoadShapeFixedLenDataset(
        X_val_raw, y_val, feat_mean, feat_std, augment=False
    )

    # ========== 关键修改1：使用平方根倒数权重替代 balanced 权重 ==========
    class_counts = np.bincount(y_train)
    sqrt_weights = 1.0 / np.sqrt(class_counts)
    sqrt_weights = sqrt_weights / sqrt_weights.sum() * len(class_counts)  # 平均权重≈1
    class_weights = torch.tensor(sqrt_weights, dtype=torch.float32)
    print("类别权重 (平方根倒数):", class_weights.numpy())

    # ========== 关键修改2：关闭 WeightedRandomSampler，使用普通随机打乱 ==========
    use_weighted_sampler = False
    if use_weighted_sampler:
        sample_weights = class_weights[torch.tensor(y_train, dtype=torch.long)]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_fixed)
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fixed)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fixed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GRUFixedLenClassifier(
        input_dim=5, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES, dropout=DROPOUT
    )

    print("\n--- 初始 GRU 参数范数 ---")
    for name, param in model.gru.named_parameters():
        print(f"{name:30}  norm: {param.norm().item():.6f}")

    # ========== 关键修改3：启用 Focal Loss，gamma=1.0 ==========
    trained_model = train_model(
        model, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=device,
        class_weights=class_weights, use_focal=True, gamma=1.0
    )
    trained_model = train_model(
        model, train_loader, val_loader, epochs=EPOCHS, lr=LR/2.0, device=device,
        class_weights=class_weights, use_focal=True, gamma=1.0
    )
    trained_model = train_model(
        model, train_loader, val_loader, epochs=EPOCHS, lr=LR/4.0, device=device,
        class_weights=class_weights, use_focal=True, gamma=1.0
    )
    trained_model = train_model(
        model, train_loader, val_loader, epochs=EPOCHS, lr=LR/8.0, device=device,
        class_weights=class_weights, use_focal=True, gamma=1.0
    )
    final_acc = evaluate(trained_model, val_loader, device)
    print(f"\nFinal Validation Accuracy: {final_acc:.4f}")
    print("\n分类报告（验证集）：")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
    print(classification_report(all_labels, all_preds,
                                target_names=[f"class_{i}" for i in range(NUM_CLASSES)],
                                zero_division=0))

    print("\n--- 诊断：隐藏状态前5位 (两个训练样本) ---")
    with torch.no_grad():
        for i, (x, y) in enumerate(train_loader):
            out, hn = model.gru(x.to(device))
            print(hn[-1][:2, :5])
            break

    print("\n--- 训练后 GRU 参数范数 ---")
    for name, param in model.gru.named_parameters():
        print(f"{name:30}  norm: {param.norm().item():.6f}")

    torch.save(trained_model.state_dict(), "road_shape_gru_64.pth")
    np.savez('feat_stats.npz', mean=feat_mean, std=feat_std)
    print("\n模型已保存为 road_shape_gru_64.pth，统计量保存为 feat_stats.npz")

if __name__ == "__main__":
    main()