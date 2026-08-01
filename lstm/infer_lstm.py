# LSTM 版推理（与 train_lstm.py 管线一致），输出统计 + margin + 最弱样本
import json
import time
import numpy as np
import torch
import torch.nn as nn

from train_lstm import add_motion_features, LSTMFixedLenClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMFixedLenClassifier(input_dim=4, hidden_dim=8)
model.load_state_dict(torch.load('road_shape_lstm.pth', map_location=device))
model.to(device)
model.eval()

stats = np.load('feat_stats_lstm.npz')
feat_mean = torch.as_tensor(stats['mean'], dtype=torch.float32)
feat_std = torch.as_tensor(stats['std'], dtype=torch.float32)

idx_to_label = {0: "zhixian", 1: "shizi", 2: "daodazuohuandao", 3: "jinruzuohuandao",
                4: "daodayouhuandao", 5: "jinruyouhuandao"}

def predict(pts):
    feat = add_motion_features(pts)
    feat = torch.as_tensor(feat, dtype=torch.float32)
    feat = (feat - feat_mean) / feat_std
    with torch.no_grad():
        probs = torch.softmax(model(feat.unsqueeze(0).to(device)), dim=1).squeeze().cpu().numpy()
    return probs, int(np.argmax(probs))

if __name__ == "__main__":
    json_path = "../2026_8_1_v1.json"
    with open(json_path, 'r') as f:
        raw = json.load(f)
    label_stats = {name: [0, 0] for name in idx_to_label.values()}
    errors, worst_margin, worst_info = [], 1.0, None
    for i, d in enumerate(raw):
        pts = np.array(d['point'], dtype=np.float32)
        probs, pred_id = predict(pts)
        pred_name = idx_to_label[pred_id]
        label_stats[d['label']][1] += 1
        sorted_p = np.sort(probs)
        margin = float(sorted_p[-1] - sorted_p[-2])
        if margin < worst_margin:
            worst_margin, worst_info = margin, (i, d['label'], pred_name, float(probs[pred_id]))
        if pred_name == d['label']:
            label_stats[d['label']][0] += 1
        else:
            errors.append((i, d['label'], pred_name, float(probs[pred_id])))
    total = len(raw)
    print(f"总样本数: {total}，预测正确: {total - len(errors)}，错误: {len(errors)}，总体准确率: {(total-len(errors))/total:.4f}")
    for name in idx_to_label.values():
        c, n = label_stats[name]
        print(f"  {name:16s} {c:3d}/{n:<3d}  准确率 {c/n:.4f}" if n else f"  {name}: 0")
    if errors:
        for i, t, p, c in errors:
            print(f"  JSON#{i}: {t} -> {p} (置信度 {c:.4f})")
    else:
        print("全部预测正确！")
    i, t, p, c = worst_info
    print(f"最弱样本: JSON#{i}，真实 {t}，预测 {p}，置信度 {c:.4f}，margin {worst_margin:.4f}")

    # 计时（随机64坐标）
    model.cpu()
    rng = np.random.default_rng(42)
    sample = rng.uniform(0, 320, (64, 2)).astype(np.float32)
    N = 3000
    for _ in range(20):
        predict(sample)
    t0 = time.perf_counter_ns()
    for _ in range(N):
        predict(sample)
    t1 = time.perf_counter_ns()
    print(f"推理耗时: {(t1-t0)/N:,.0f} ns/样本 ({(t1-t0)/N/1000:,.1f} µs)")
