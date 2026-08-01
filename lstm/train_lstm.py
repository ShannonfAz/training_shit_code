# LSTM 版训练（自由设计版，不受要求.md约束）
# 环境变量: LSTM_SEED / LSTM_HIDDEN / LSTM_EPOCHS / LSTM_STEP / LSTM_STAGES
# 每次训练结束自动计时（随机64坐标，完整推理管线 ns）
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

CFG = {
    'seed':   int(os.environ.get('LSTM_SEED', '55')),
    'hidden': int(os.environ.get('LSTM_HIDDEN', '8')),
    'epochs': int(os.environ.get('LSTM_EPOCHS', '100')),
    'step':   int(os.environ.get('LSTM_STEP', '30')),
    'stages': [float(x) for x in os.environ.get('LSTM_STAGES', '1.0,0.9,0.8,0.7,0.6,0.5').split(',')],
}
print("LSTM 配置:", CFG)
torch.manual_seed(CFG['seed'])
np.random.seed(CFG['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

JSON_PATH = os.environ.get('LSTM_JSON', '../2026_8_1_v1.json')
SEQ_LEN, NUM_CLASSES, BATCH_SIZE = 64, 6, 32
LR = 4e-2

# -------------------- 特征工程（沿用验证过的 4 特征） --------------------
def add_motion_features(seq):
    diff = np.diff(seq, axis=0)
    diff = np.vstack([diff, diff[-1]])
    norm = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-8
    dir_vec = diff / norm
    dir_diff = np.diff(dir_vec, axis=0)
    dir_diff = np.vstack([dir_diff, dir_diff[-1]])
    return np.column_stack([diff, dir_diff])          # (64,4)

def compute_feature_stats(sequences):
    all_feats = np.concatenate([add_motion_features(s) for s in sequences], axis=0)
    return all_feats.mean(axis=0), all_feats.std(axis=0) + 1e-8

# -------------------- 数据集 --------------------
class RoadShapeDataset(Dataset):
    def __init__(self, sequences, labels, fmean, fstd, augment=False):
        self.sequences, self.labels = sequences, labels
        self.fmean = torch.as_tensor(fmean, dtype=torch.float32)
        self.fstd = torch.as_tensor(fstd, dtype=torch.float32)
        self.augment = augment

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx].copy()
        if self.augment:
            angle = np.random.uniform(-5, 5) * np.pi / 180
            rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            seq = seq @ rot.T
            seq[:, 0] += np.random.uniform(-0.02 * 320, 0.02 * 320)
            seq[:, 1] += np.random.uniform(-0.02 * 240, 0.02 * 240)
            seq += np.random.normal(0, 0.5, seq.shape)
        feat = add_motion_features(seq)
        feat = torch.as_tensor(feat, dtype=torch.float32)
        feat = (feat - self.fmean) / self.fstd
        return feat, torch.tensor(self.labels[idx], dtype=torch.long)

def collate_fixed(batch):
    seqs, labels = zip(*batch)
    return torch.stack(seqs, dim=0), torch.stack(labels, dim=0)

# -------------------- LSTM 模型 --------------------
class LSTMFixedLenClassifier(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=8, num_layers=1, num_classes=6, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        feat = out.mean(dim=1)
        feat = self.ln(feat)
        return self.fc(feat)

# -------------------- 数据加载 --------------------
LABEL_MAP = {"zhixian": 0, "shizi": 1, "daodazuohuandao": 2,
             "jinruzuohuandao": 3, "daodayouhuandao": 4, "jinruyouhuandao": 5}

def load_data_from_json(json_path, seq_len=64):
    with open(json_path, 'r') as f:
        raw_data = json.load(f)
    sequences, labels, indices = [], [], []
    skipped = 0
    for idx, item in enumerate(raw_data):
        pts = np.array(item["point"], dtype=np.float32)
        if pts.shape[0] != seq_len:
            skipped += 1
            continue
        sequences.append(pts)
        labels.append(LABEL_MAP[item["label"]])
        indices.append(idx)
    print(f"加载数据：有效样本 {len(sequences)}，跳过 {skipped}。")
    return sequences, labels, indices

# -------------------- 训练与评估 --------------------
def train_model(model, train_loader, val_loader, epochs, lr, device, class_weights,
                select_loader=None):
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=CFG['step'], gamma=0.5)
    best_acc, best_quality, best_state = 0.0, -1.0, None
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        scheduler.step()
        val_acc = evaluate(model, val_loader, device)
        if select_loader is not None:
            select_acc, sel_margin, _, _ = evaluate_with_quality(model, select_loader, device)
            if select_acc > best_acc or (select_acc == best_acc and sel_margin > best_quality):
                best_acc, best_quality = select_acc, sel_margin
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"Epoch {epoch+1:03d} | Loss {total_loss/len(train_loader.dataset):.4f} | Val {val_acc:.4f} | Full {select_acc:.4f}")
        else:
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"Epoch {epoch+1:03d} | Loss {total_loss/len(train_loader.dataset):.4f} | Val {val_acc:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  -> 本阶段最佳准确率 {best_acc:.4f} (最小margin {best_quality:.4f})")
    return model

def evaluate(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            all_preds.extend(torch.argmax(model(inputs), dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return accuracy_score(all_labels, all_preds)

def evaluate_with_quality(model, data_loader, device):
    model.eval()
    margins, confs, correct, n = [], [], 0, 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            probs = torch.softmax(model(inputs), dim=1)
            top2 = torch.topk(probs, 2, dim=1)
            margins.extend((top2.values[:, 0] - top2.values[:, 1]).tolist())
            confs.extend(top2.values[:, 0].tolist())
            correct += (top2.indices[:, 0] == labels).sum().item()
            n += labels.size(0)
    worst = int(np.argmin(margins))
    return correct / n, float(min(margins)), float(min(confs)), worst

# -------------------- 推理计时（每训必测） --------------------
def bench_inference(model, feat_mean, feat_std, device, N=3000):
    model.eval()
    rng = np.random.default_rng(42)
    sample = rng.uniform(0, 320, (64, 2)).astype(np.float32)
    for _ in range(20):   # 预热
        feat = add_motion_features(sample)
        feat = torch.as_tensor(feat, dtype=torch.float32)
        feat = (feat - feat_mean) / feat_std
        with torch.no_grad():
            model(feat.unsqueeze(0).to(device))
    t0 = time.perf_counter_ns()
    for _ in range(N):
        feat = add_motion_features(sample)
        feat = torch.as_tensor(feat, dtype=torch.float32)
        feat = (feat - feat_mean) / feat_std
        with torch.no_grad():
            out = model(feat.unsqueeze(0).to(device))
    t1 = time.perf_counter_ns()
    return (t1 - t0) / N

# -------------------- 主流程 --------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sequences, labels, idxs = load_data_from_json(JSON_PATH, seq_len=SEQ_LEN)
    X_tr, X_va, y_tr, y_va, idx_tr, idx_va = train_test_split(
        sequences, labels, idxs, test_size=0.2, random_state=42, stratify=labels)
    fmean, fstd = compute_feature_stats(X_tr)
    train_ds = RoadShapeDataset(X_tr, y_tr, fmean, fstd, augment=True)
    val_ds = RoadShapeDataset(X_va, y_va, fmean, fstd, augment=False)
    full_ds = RoadShapeDataset(sequences, labels, fmean, fstd, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fixed)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fixed)
    full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fixed)

    class_counts = np.bincount(y_tr)
    w = 1.0 / np.sqrt(class_counts)
    class_weights = torch.tensor(w / w.sum() * len(class_counts), dtype=torch.float32)

    model = LSTMFixedLenClassifier(input_dim=4, hidden_dim=CFG['hidden']).to(device)

    global_best_acc, global_best_quality, global_best_state = 0.0, -1.0, None
    for factor in CFG['stages']:
        model = train_model(model, train_loader, val_loader, CFG['epochs'], LR * factor,
                            device, class_weights, select_loader=full_loader)
        acc, margin, _, _ = evaluate_with_quality(model, full_loader, device)
        if acc > global_best_acc or (acc == global_best_acc and margin > global_best_quality):
            global_best_acc, global_best_quality = acc, margin
            global_best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if global_best_state is not None:
        model.load_state_dict(global_best_state)
    print(f"全局最佳: 全量 {global_best_acc:.4f} (最小margin {global_best_quality:.4f})")

    # 最终验证
    full_acc, min_margin, min_conf, worst_idx = evaluate_with_quality(model, full_loader, device)
    val_acc = evaluate(model, val_loader, device)
    print(f"最终验证准确率: {val_acc:.4f}，最终全量准确率: {full_acc:.4f}")
    feat, label = full_ds[worst_idx]
    with torch.no_grad():
        prob = torch.softmax(model(feat.unsqueeze(0).to(device)), dim=1).squeeze()
    inv = {v: k for k, v in LABEL_MAP.items()}
    print(f"最弱样本: JSON#{worst_idx}，真实 {inv[int(label)]}，预测 {inv[int(prob.argmax())]}，"
          f"margin {min_margin:.4f}")

    # 计时
    ns = bench_inference(model, fmean, fstd, device)
    print(f"推理耗时: {ns:,.0f} ns/样本 ({ns/1000:,.1f} µs)  [预算: <600µs 目标, <1000µs 放宽]")

    torch.save(model.state_dict(), 'road_shape_lstm.pth')
    np.savez('feat_stats_lstm.npz', mean=fmean, std=fstd)
    print("模型与统计量已保存。")

if __name__ == "__main__":
    main()
